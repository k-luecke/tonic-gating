from __future__ import annotations

import torch

from phase_routing.model import LearnedRouter, MoETransformer, count_parameters


def test_learned_router_shapes_and_normalization() -> None:
    router = LearnedRouter(width=128, n_experts=8, top_k=2)
    hidden = torch.randn(3, 17, 128)
    result = router(hidden)
    assert result.logits.shape == (3, 17, 8)
    assert result.probabilities.shape == (3, 17, 8)
    assert result.indices.shape == (3, 17, 2)
    assert result.gates.shape == (3, 17, 2)
    assert result.auxiliary_loss.shape == ()
    torch.testing.assert_close(result.probabilities.sum(-1), torch.ones(3, 17))
    torch.testing.assert_close(result.gates.sum(-1), torch.ones(3, 17))
    assert torch.all((result.indices >= 0) & (result.indices < 8))


def test_model_shapes_gradient_and_parameter_budget() -> None:
    torch.manual_seed(7)
    model = MoETransformer()
    tokens = torch.randint(8, 64, (2, 16))
    tokens[:, 0] = torch.tensor([0, 7])
    logits, auxiliary = model(tokens)
    assert logits.shape == (2, 16, 64)
    assert auxiliary.shape == ()
    assert model.last_router_output is not None
    (logits.mean() + 0.01 * auxiliary).backward()
    assert model.blocks[2].ffn.router.projection.weight.grad is not None
    assert 1_000_000 <= count_parameters(model) <= 2_000_000


def test_shared_task_embedding_tracks_latest_marker() -> None:
    tokens = torch.tensor([[2, 9, 10, 6, 11, 12]])
    expected = torch.tensor([[2, 2, 2, 6, 6, 6]])
    assert torch.equal(MoETransformer.resolve_task_ids(tokens), expected)
