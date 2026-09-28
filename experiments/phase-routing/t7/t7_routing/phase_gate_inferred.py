"""Inferred φ_ref Phase Gate — estimates reference phase from routing hidden."""
from __future__ import annotations

import math

import torch
from torch import nn

from .model import RouterOutput


class PhaseGateInferredRouter(nn.Module):
    """Half-wave cosine phase gate with φ_ref inferred from the hidden state.

    Expert phases are fixed at 2π e / 8. A small Linear(width → 2) head maps the
    routing hidden to (coŝ, sin̂); φ_ref = atan2(sin̂, coŝ). Gains use the same
    max(0, cos(φ_e − φ_ref)) half-wave as the supplied PhaseGateRouter.

    **Forbidden for routing:** marker → task_phases[task_id] lookup. Tokens are
    accepted for API parity but ignored for φ_ref (routing is hidden-only).
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
        del top_k, n_tasks  # Accepted for common router constructor interface.
        if n_experts != 8:
            raise ValueError("T7 PhaseGateInferred is fixed at 8 experts")
        self.n_experts = n_experts
        self.width = width
        phases = torch.arange(n_experts, dtype=torch.float32) * (2.0 * math.pi / n_experts)
        self.register_buffer("expert_phases", phases)
        self.phase_head = nn.Linear(width, 2)

    def gains_for_phase(self, reference_phase: torch.Tensor) -> torch.Tensor:
        """Return unnormalized max(0, cos(phi_e - phi_ref)) gains."""
        expert_phases = self.expert_phases.to(
            device=reference_phase.device, dtype=reference_phase.dtype
        )
        cosine = torch.cos(expert_phases - reference_phase.unsqueeze(-1))
        gains = torch.clamp_min(cosine, 0.0)
        return torch.where(cosine.abs() <= 1e-7, torch.zeros_like(gains), gains)

    def infer_phi_ref(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map hidden → continuous φ_ref via unit-circle parameterization."""
        xy = self.phase_head(hidden)
        # atan2(sin, cos) — continuous phase from Linear(width→2) outputs.
        return torch.atan2(xy[..., 1], xy[..., 0])

    def forward(
        self, hidden: torch.Tensor, *, tokens: torch.Tensor | None = None
    ) -> RouterOutput:
        # tokens deliberately ignored for φ_ref (API parity only).
        del tokens
        reference_phases = self.infer_phi_ref(hidden)
        gains = self.gains_for_phase(reference_phases)
        normalizer = gains.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probabilities = gains / normalizer

        selected_gains, indices = gains.topk(self.n_experts // 2, dim=-1)
        active = selected_gains > 0
        indices = torch.where(active, indices, indices.new_full((), -1))
        gates = selected_gains / normalizer
        auxiliary_loss = hidden.new_zeros(())
        return RouterOutput(gains, probabilities, indices, gates, auxiliary_loss)
