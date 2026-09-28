"""T7 harness: costs, disturbance placement, noise hits inferred router."""
from __future__ import annotations

import math

import torch

from harness import dominant_sets, switching_time
from t7_routing.costs import (
    DISTURBANCE_GRID,
    EXPERT_MACS_PER_EXPERT,
    LR_ROUTE_MACS_PER_TOKEN,
    PG_INFERRED_COSINE_OPS,
    PG_INFERRED_PHASE_HEAD_MACS,
    expert_compute_cost,
    routing_decision_cost,
)
from t7_routing.factory import make_model
from t7_routing.model import MoEFeedForward, RouterOutput
from t7_routing.phase_gate_inferred import PhaseGateInferredRouter


def test_disturbance_grid() -> None:
    assert DISTURBANCE_GRID == (0.0, 0.1, 0.3, 1.0)


def test_routing_costs_include_pg_inferred() -> None:
    lr = routing_decision_cost("lr")
    assert lr["C_route_MACs_per_token"] == LR_ROUTE_MACS_PER_TOKEN == 1024

    goal = routing_decision_cost("lr_goal")
    assert goal["C_route_MACs_per_token"] == 2048

    pgi = routing_decision_cost("pg_inferred")
    assert pgi["C_route_MACs_per_token"] == PG_INFERRED_PHASE_HEAD_MACS == 256
    assert pgi["C_route_ops"] == PG_INFERRED_COSINE_OPS == 8
    # Honest: NOT claiming C_route=8 alone.
    assert "256" in pgi["C_route_definition"]
    assert "NOT C_route=8" in pgi["C_route_definition"] or "not C_route=8" in pgi["C_route_definition"].lower()

    assert expert_compute_cost(2.0)["C_exp_MACs_per_token"] == 2.0 * EXPERT_MACS_PER_EXPERT


class _CaptureRouter(torch.nn.Module):
    def __init__(self, n_experts: int = 4) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.last_hidden: torch.Tensor | None = None

    def forward(self, hidden: torch.Tensor, *, tokens: torch.Tensor | None = None) -> RouterOutput:
        del tokens
        self.last_hidden = hidden.detach().clone()
        batch, seq, _ = hidden.shape
        indices = torch.zeros(batch, seq, 1, dtype=torch.long, device=hidden.device)
        gates = torch.ones(batch, seq, 1, device=hidden.device, dtype=hidden.dtype)
        probs = torch.zeros(batch, seq, self.n_experts, device=hidden.device, dtype=hidden.dtype)
        probs[..., 0] = 1.0
        return RouterOutput(probs, probs, indices, gates, hidden.new_zeros(()))


def test_noise_hits_routing_only_for_inferred_path() -> None:
    torch.manual_seed(0)
    router = _CaptureRouter(4)
    ff = MoEFeedForward(width=8, hidden_width=16, router=router, n_experts=4)
    ff.eval()
    hidden = torch.randn(2, 5, 8)
    shift = torch.arange(8, dtype=hidden.dtype)
    expert_inputs: list[torch.Tensor] = []
    real_forward = ff.experts[0].forward

    def capture_expert(x: torch.Tensor) -> torch.Tensor:
        expert_inputs.append(x.detach().clone())
        return real_forward(x)

    ff.experts[0].forward = capture_expert  # type: ignore[method-assign]

    ff.eval_shift = None
    ff.eval_noise_std = 0.5
    expert_inputs.clear()
    torch.manual_seed(11)
    expected_noise = torch.randn_like(hidden) * 0.5
    torch.manual_seed(11)
    ff(hidden)
    assert router.last_hidden is not None
    assert torch.allclose(router.last_hidden, hidden + expected_noise)
    assert expert_inputs and torch.allclose(expert_inputs[0], hidden.reshape(-1, 8))

    ff.eval_shift = shift
    ff.eval_noise_std = 0.25
    expert_inputs.clear()
    torch.manual_seed(22)
    expected_noise = torch.randn_like(hidden) * 0.25
    torch.manual_seed(22)
    ff(hidden)
    assert torch.allclose(router.last_hidden, hidden + shift + expected_noise)
    assert torch.allclose(expert_inputs[0], (hidden + shift).reshape(-1, 8))


def test_noise_changes_pg_inferred_phi_ref() -> None:
    """Unlike supplied PG, noise on routing hidden must change inferred φ_ref."""
    torch.manual_seed(42)
    router = PhaseGateInferredRouter(width=16)
    hidden = torch.randn(1, 3, 16)
    phi_clean = router.infer_phi_ref(hidden)
    noise = torch.randn_like(hidden) * 0.5
    phi_noisy = router.infer_phi_ref(hidden + noise)
    assert not torch.allclose(phi_clean, phi_noisy)


def test_pg_inferred_model_noise_changes_gates() -> None:
    torch.manual_seed(1)
    model = make_model("pg_inferred", width=32, n_heads=4, ffn_width=64)
    model.eval()
    tokens = torch.randint(8, 64, (1, 12))
    tokens[:, 0] = 3
    model.configure_eval_disturbance()
    model(tokens)
    gates_clean = model.last_router_output.gates.detach().clone()
    model.configure_eval_disturbance(noise_std=1.0)
    torch.manual_seed(99)
    model(tokens)
    gates_noisy = model.last_router_output.gates.detach().clone()
    # With strong noise, gates should typically differ (not by-construction invariant).
    assert not torch.allclose(gates_clean, gates_noisy)
