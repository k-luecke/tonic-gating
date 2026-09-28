from __future__ import annotations

import torch

from phase_routing.factory import make_model
from phase_routing.phase_gate import PhaseGateRouter


def test_exact_expert_phase_has_unit_gain_and_zero_boundary_neighbors() -> None:
    router = PhaseGateRouter(width=16)
    for expert in range(8):
        gains = router.gains_for_phase(router.expert_phases[expert])
        torch.testing.assert_close(gains[expert], torch.tensor(1.0))
        # The two experts at +/- pi/2 are the half-wave gate boundaries.
        torch.testing.assert_close(gains[(expert - 2) % 8], torch.tensor(0.0))
        torch.testing.assert_close(gains[(expert + 2) % 8], torch.tensor(0.0))
        torch.testing.assert_close(gains[(expert - 3) % 8], torch.tensor(0.0))
        torch.testing.assert_close(gains[(expert + 3) % 8], torch.tensor(0.0))
        torch.testing.assert_close(gains[(expert + 4) % 8], torch.tensor(0.0))


def test_marker_switch_updates_reference_phase_per_token() -> None:
    router = PhaseGateRouter(width=16)
    hidden = torch.randn(1, 7, 16)
    tokens = torch.tensor([[1, 18, 19, 6, 20, 21, 22]])
    result = router(hidden, tokens=tokens)
    expected_tasks = torch.tensor([[1, 1, 1, 6, 6, 6, 6]])
    assert torch.equal(router.resolve_task_ids(tokens), expected_tasks)
    expected = router.gains_for_phase(router.task_phases[expected_tasks])
    torch.testing.assert_close(result.logits, expected)
    torch.testing.assert_close(result.gates.sum(dim=-1), torch.ones(1, 7))
    assert torch.all(result.indices[result.gates == 0] == -1)


def test_phase_gate_model_runs_and_phase_table_gets_gradient() -> None:
    torch.manual_seed(9)
    model = make_model("phase_gate", width=32, n_heads=4, ffn_width=64)
    tokens = torch.randint(8, 64, (2, 16))
    tokens[:, 0] = torch.tensor([0, 7])
    logits, auxiliary = model(tokens)
    assert logits.shape == (2, 16, 64)
    assert float(auxiliary) == 0.0
    logits.mean().backward()
    router = model.blocks[2].ffn.router
    assert router.task_phases.grad is not None
