"""Parameter-free deterministic token hash routing baselines."""
from __future__ import annotations

import torch
from torch import nn

from .model import RouterOutput


class HashRouter(nn.Module):
    """Route each token with one fixed hash (top-1) or two fixed hashes (top-2)."""

    def __init__(self, width: int, n_experts: int = 8, top_k: int = 1) -> None:
        super().__init__()
        del width
        if n_experts != 8 or top_k not in (1, 2):
            raise ValueError("HashRouter supports 8 experts and top_k in {1, 2}")
        self.n_experts = n_experts
        self.top_k = top_k

    def forward(
        self, hidden: torch.Tensor, *, tokens: torch.Tensor | None = None
    ) -> RouterOutput:
        if tokens is None or tokens.shape != hidden.shape[:2]:
            raise ValueError("HashRouter requires tokens matching hidden [batch, sequence]")
        first = (5 * tokens + 1).remainder(self.n_experts)
        indices = first.unsqueeze(-1)
        if self.top_k == 2:
            # This hash cannot collide with the first modulo 8: equality would
            # require 2*token == 1 (mod 8).
            second = (3 * tokens + 2).remainder(self.n_experts)
            indices = torch.stack((first, second), dim=-1)
        gates = hidden.new_full(indices.shape, 1.0 / self.top_k)
        probabilities = hidden.new_zeros(*tokens.shape, self.n_experts)
        probabilities.scatter_add_(-1, indices, gates)
        logits = probabilities
        return RouterOutput(logits, probabilities, indices, gates, hidden.new_zeros(()))
