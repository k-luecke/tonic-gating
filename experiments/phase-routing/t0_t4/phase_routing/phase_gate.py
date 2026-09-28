"""Task-marker-controlled half-wave cosine Phase Gate router."""
from __future__ import annotations

import math

import torch
from torch import nn

from .model import RouterOutput


class PhaseGateRouter(nn.Module):
    """Route tokens by learnable task reference phases and fixed expert phases.

    Marker tokens 0..7 select the task reference phase.  A marker updates routing
    at its own position and all following positions, which handles the switching
    split while reducing stationary sequences to their first-token marker.
    """

    def __init__(
        self,
        width: int,
        n_experts: int = 8,
        top_k: int | None = None,
        *,
        n_tasks: int = 8,
    ) -> None:
        super().__init__()
        del width, top_k  # Accepted for the common router constructor interface.
        if n_experts != 8 or n_tasks != 8:
            raise ValueError("T3 Phase Gate is fixed at 8 experts and 8 tasks")
        self.n_experts = n_experts
        self.n_tasks = n_tasks
        phases = torch.arange(n_experts, dtype=torch.float32) * (2.0 * math.pi / n_experts)
        self.register_buffer("expert_phases", phases)
        # Start each task aligned to its corresponding expert; references remain learnable.
        self.task_phases = nn.Parameter(phases.clone())

    def resolve_task_ids(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return the most recent marker task at every token position."""
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        marker = (tokens >= 0) & (tokens < self.n_tasks)
        positions = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
        marker_positions = torch.where(marker, positions, positions.new_full((), -1))
        latest_positions = marker_positions.cummax(dim=1).values
        if torch.any(latest_positions < 0):
            raise ValueError("each sequence must begin with a task marker token 0..7")
        return tokens.gather(1, latest_positions)

    def gains_for_phase(self, reference_phase: torch.Tensor) -> torch.Tensor:
        """Return unnormalized max(0, cos(phi_e - phi_ref)) gains."""
        expert_phases = self.expert_phases.to(device=reference_phase.device, dtype=reference_phase.dtype)
        cosine = torch.cos(expert_phases - reference_phase.unsqueeze(-1))
        gains = torch.clamp_min(cosine, 0.0)
        # cos(pi/2) is a tiny signed residual in floating point; geometrically it is zero.
        return torch.where(cosine.abs() <= 1e-7, torch.zeros_like(gains), gains)

    def forward(
        self, hidden: torch.Tensor, *, tokens: torch.Tensor | None = None
    ) -> RouterOutput:
        if tokens is None:
            raise ValueError("PhaseGateRouter requires tokens for task-marker routing")
        if tokens.shape != hidden.shape[:2]:
            raise ValueError("tokens must match hidden batch and sequence dimensions")
        task_ids = self.resolve_task_ids(tokens)
        reference_phases = self.task_phases[task_ids].to(dtype=hidden.dtype)
        gains = self.gains_for_phase(reference_phases)
        normalizer = gains.sum(dim=-1, keepdim=True)
        probabilities = gains / normalizer

        # A half-wave cosine over eight evenly spaced phases has at most four
        # positive entries. Keep only those slots and mark exact-zero slots -1,
        # so MoEFeedForward never invokes an expert for a zero gain.
        selected_gains, indices = gains.topk(self.n_experts // 2, dim=-1)
        active = selected_gains > 0
        indices = torch.where(active, indices, indices.new_full((), -1))
        gates = selected_gains / normalizer
        auxiliary_loss = hidden.new_zeros(())
        return RouterOutput(gains, probabilities, indices, gates, auxiliary_loss)
