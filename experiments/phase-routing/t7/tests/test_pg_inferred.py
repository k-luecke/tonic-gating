from __future__ import annotations

import torch

from t7_routing.factory import make_model, router_class
from t7_routing.phase_gate_inferred import PhaseGateInferredRouter


def test_phi_ref_changes_with_hidden_independent_of_markers() -> None:
    """Same hidden + different markers → identical routing; different hidden → different φ_ref."""
    torch.manual_seed(0)
    router = PhaseGateInferredRouter(width=32)
    hidden = torch.randn(1, 5, 32)
    tokens_a = torch.tensor([[0, 10, 11, 12, 13]])
    tokens_b = torch.tensor([[7, 20, 21, 22, 23]])
    out_a = router(hidden, tokens=tokens_a)
    out_b = router(hidden, tokens=tokens_b)
    torch.testing.assert_close(out_a.logits, out_b.logits)
    torch.testing.assert_close(out_a.gates, out_b.gates)
    assert torch.equal(out_a.indices, out_b.indices)

    phi1 = router.infer_phi_ref(hidden)
    phi2 = router.infer_phi_ref(hidden + 1.0)
    assert not torch.allclose(phi1, phi2)


def test_gains_half_wave_like_parent() -> None:
    router = PhaseGateInferredRouter(width=16)
    for expert in range(8):
        gains = router.gains_for_phase(router.expert_phases[expert])
        torch.testing.assert_close(gains[expert], torch.tensor(1.0))
        torch.testing.assert_close(gains[(expert - 2) % 8], torch.tensor(0.0))
        torch.testing.assert_close(gains[(expert + 2) % 8], torch.tensor(0.0))


def test_pg_inferred_model_runs_and_phase_head_gets_gradient() -> None:
    torch.manual_seed(9)
    model = make_model("pg_inferred", width=32, n_heads=4, ffn_width=64)
    assert router_class("pg_inferred") is PhaseGateInferredRouter
    tokens = torch.randint(8, 64, (2, 16))
    tokens[:, 0] = torch.tensor([0, 7])
    logits, auxiliary = model(tokens)
    assert logits.shape == (2, 16, 64)
    assert float(auxiliary) == 0.0
    logits.mean().backward()
    router = model.blocks[2].ffn.router
    assert router.phase_head.weight.grad is not None
    # No task_phases parameter on inferred router.
    assert not hasattr(router, "task_phases")


def test_tokens_ignored_for_routing_api_parity() -> None:
    router = PhaseGateInferredRouter(width=16)
    hidden = torch.randn(2, 4, 16)
    # tokens=None is allowed (unlike supplied PG).
    out = router(hidden, tokens=None)
    assert out.indices.shape == (2, 4, 4)
    torch.testing.assert_close(out.gates.sum(-1), torch.ones(2, 4), atol=1e-5, rtol=1e-5)
