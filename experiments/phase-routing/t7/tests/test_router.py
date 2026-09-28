from __future__ import annotations

import torch

from t7_routing.factory import make_model
from t7_routing.model import LearnedRouter, MoETransformer, count_parameters


def test_learned_router_shapes() -> None:
    router = LearnedRouter(width=128, n_experts=8, top_k=2)
    hidden = torch.randn(3, 17, 128)
    result = router(hidden)
    assert result.indices.shape == (3, 17, 2)
    torch.testing.assert_close(result.gates.sum(-1), torch.ones(3, 17))


def test_model_shapes_and_budget() -> None:
    torch.manual_seed(7)
    model = MoETransformer()
    tokens = torch.randint(8, 64, (2, 16))
    tokens[:, 0] = torch.tensor([0, 7])
    logits, auxiliary = model(tokens)
    assert logits.shape == (2, 16, 64)
    (logits.mean() + 0.01 * auxiliary).backward()
    assert 1_000_000 <= count_parameters(model) <= 2_000_000


def test_factory_all_primary_arms() -> None:
    for arm, k in (("lr", 2), ("hr1", 1), ("lr_goal", 2), ("pg_inferred", 4), ("pg", 4)):
        model = make_model(arm, width=32, n_heads=4, ffn_width=64)
        tokens = torch.randint(8, 64, (2, 12))
        tokens[:, 0] = torch.tensor([0, 7])
        logits, _ = model(tokens)
        assert logits.shape == (2, 12, 64)
        assert model.last_router_output.indices.shape[-1] == k
