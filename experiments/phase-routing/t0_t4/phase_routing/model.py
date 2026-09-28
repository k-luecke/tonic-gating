"""Decoder-only Transformer with a pluggable layer-3 MoE router."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class RouterOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    indices: torch.Tensor
    gates: torch.Tensor
    auxiliary_loss: torch.Tensor


class Router(Protocol):
    def forward(
        self,
        hidden: torch.Tensor,
        *,
        tokens: torch.Tensor | None = None,
    ) -> RouterOutput: ...


class LearnedRouter(nn.Module):
    """Linear hidden-to-expert router with normalized top-k gates."""

    def __init__(self, width: int, n_experts: int = 8, top_k: int = 2) -> None:
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError("top_k must be in [1, n_experts]")
        self.n_experts = n_experts
        self.top_k = top_k
        self.projection = nn.Linear(width, n_experts)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        tokens: torch.Tensor | None = None,
    ) -> RouterOutput:
        del tokens
        logits = self.projection(hidden)
        probabilities = F.softmax(logits, dim=-1)
        top_probabilities, indices = probabilities.topk(self.top_k, dim=-1)
        gates = top_probabilities / top_probabilities.sum(dim=-1, keepdim=True)

        # Top-k generalization of Switch's E * sum(load_e * importance_e).
        flat_indices = indices.reshape(-1)
        load = F.one_hot(flat_indices, self.n_experts).float().mean(dim=0)
        importance = probabilities.reshape(-1, self.n_experts).mean(dim=0)
        auxiliary_loss = self.n_experts * torch.sum(load.detach() * importance)
        return RouterOutput(logits, probabilities, indices, gates, auxiliary_loss)


class FeedForward(nn.Module):
    def __init__(self, width: int, hidden_width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, width),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        width: int,
        hidden_width: int,
        router: Router,
        n_experts: int = 8,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.router = router
        self.experts = nn.ModuleList(
            FeedForward(width, hidden_width) for _ in range(n_experts)
        )
        # Evaluation-only interventions configured by the T4 harness.
        self.eval_noise_std = 0.0
        self.eval_shift: torch.Tensor | None = None
        self.eval_dropped_experts: frozenset[int] = frozenset()

    def forward(
        self, hidden: torch.Tensor, *, tokens: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, RouterOutput]:
        routed_hidden = hidden
        if not self.training:
            if self.eval_shift is not None:
                routed_hidden = routed_hidden + self.eval_shift.to(
                    device=hidden.device, dtype=hidden.dtype
                )
            if self.eval_noise_std:
                routed_hidden = routed_hidden + torch.randn_like(routed_hidden) * self.eval_noise_std
        route = self.router(routed_hidden, tokens=tokens)
        if self.eval_dropped_experts:
            dropped = torch.zeros_like(route.gates, dtype=torch.bool)
            for expert in self.eval_dropped_experts:
                dropped |= route.indices == expert
            surviving = route.gates.masked_fill(dropped, 0.0)
            denominator = surviving.sum(dim=-1, keepdim=True)
            surviving = torch.where(denominator > 0, surviving / denominator, surviving)
            indices = torch.where(surviving > 0, route.indices, route.indices.new_full((), -1))
            route = RouterOutput(route.logits, route.probabilities, indices, surviving, route.auxiliary_loss)
        flat_hidden = routed_hidden.reshape(-1, routed_hidden.shape[-1])
        flat_indices = route.indices.reshape(-1, route.indices.shape[-1])
        flat_gates = route.gates.reshape(-1, route.gates.shape[-1])
        output = torch.zeros_like(flat_hidden)
        # Compute only dispatched token/expert pairs. Batch all routing slots for
        # one expert into one invocation; variable-active routers otherwise cause
        # repeated tiny expert calls solely because slot order differs by token.
        for expert_id, expert in enumerate(self.experts):
            positions, slots = torch.nonzero(
                flat_indices == expert_id, as_tuple=True
            )
            if positions.numel():
                contribution = expert(flat_hidden.index_select(0, positions))
                contribution = contribution * flat_gates[positions, slots, None]
                output.index_add_(0, positions, contribution)
        return output.view_as(hidden), route


class DecoderBlock(nn.Module):
    def __init__(
        self,
        width: int,
        n_heads: int,
        hidden_width: int,
        moe: MoEFeedForward | None = None,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, n_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = moe if moe is not None else FeedForward(width, hidden_width)
        self.is_moe = moe is not None

    def forward(
        self,
        hidden: torch.Tensor,
        causal_mask: torch.Tensor,
        *,
        tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RouterOutput | None]:
        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized, normalized, normalized, attn_mask=causal_mask, need_weights=False
        )
        hidden = hidden + attended
        normalized = self.ffn_norm(hidden)
        if self.is_moe:
            fed, route = self.ffn(normalized, tokens=tokens)
        else:
            fed, route = self.ffn(normalized), None
        return hidden + fed, route


class MoETransformer(nn.Module):
    """4-layer decoder; the third layer (index 2) uses an 8-expert FFN."""

    def __init__(
        self,
        vocab_size: int = 64,
        max_seq_len: int = 128,
        width: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_width: int = 256,
        n_experts: int = 8,
        top_k: int = 2,
        router_cls: type[nn.Module] = LearnedRouter,
    ) -> None:
        super().__init__()
        if n_layers != 4:
            raise ValueError("T1 baseline is fixed at four layers")
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, width)
        # Known task identity is broadcast through the same shared path for all arms.
        self.task_embedding = nn.Embedding(8, width)
        self.position_embedding = nn.Embedding(max_seq_len, width)
        router = router_cls(width, n_experts, top_k)
        moe = MoEFeedForward(width, ffn_width, router, n_experts)
        self.blocks = nn.ModuleList([
            DecoderBlock(width, n_heads, ffn_width, moe if layer == 2 else None)
            for layer in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size, bias=False)
        self.last_router_output: RouterOutput | None = None

    @staticmethod
    def resolve_task_ids(tokens: torch.Tensor, n_tasks: int = 8) -> torch.Tensor:
        marker = (tokens >= 0) & (tokens < n_tasks)
        positions = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
        marker_positions = torch.where(marker, positions, positions.new_full((), -1))
        latest = marker_positions.cummax(dim=1).values
        if torch.any(latest < 0):
            raise ValueError("each sequence must begin with a task marker token 0..7")
        return tokens.gather(1, latest)

    def configure_eval_disturbance(
        self,
        *,
        noise_std: float = 0.0,
        shift: torch.Tensor | None = None,
        dropped_experts: tuple[int, ...] = (),
    ) -> None:
        moe = self.blocks[2].ffn
        moe.eval_noise_std = float(noise_std)
        moe.eval_shift = shift
        moe.eval_dropped_experts = frozenset(dropped_experts)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds {self.max_seq_len}")
        positions = torch.arange(seq_len, device=tokens.device)
        task_ids = self.resolve_task_ids(tokens)
        hidden = (
            self.token_embedding(tokens)
            + self.position_embedding(positions)[None]
            + self.task_embedding(task_ids)
        )
        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=tokens.device).triu(1)
        auxiliary_loss = hidden.new_zeros(())
        route = None
        for block in self.blocks:
            hidden, block_route = block(hidden, causal_mask, tokens=tokens)
            if block_route is not None:
                route = block_route
                auxiliary_loss = block_route.auxiliary_loss
        self.last_router_output = route
        return self.output(self.final_norm(hidden)), auxiliary_loss


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
