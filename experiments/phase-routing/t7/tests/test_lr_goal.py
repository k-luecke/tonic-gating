from __future__ import annotations

import torch

from t7_routing.factory import make_model, router_class
from t7_routing.model import LearnedRouterGoal, count_parameters


def test_lr_goal_uses_goal_channel() -> None:
    router = LearnedRouterGoal(width=128, n_experts=8, top_k=2)
    hidden = torch.randn(2, 9, 128)
    tokens = torch.randint(8, 64, (2, 9))
    tokens[:, 0] = torch.tensor([3, 5])
    result = router(hidden, tokens=tokens)
    assert result.logits.shape == (2, 9, 8)
    assert result.indices.shape == (2, 9, 2)
    torch.testing.assert_close(result.gates.sum(-1), torch.ones(2, 9))
    assert router.projection.in_features == 256
    assert router.goal_embedding.weight.shape == (8, 128)


def test_factory_lr_goal_extra_params() -> None:
    lr = make_model("lr")
    goal = make_model("lr_goal")
    assert router_class("lr_goal") is LearnedRouterGoal
    assert count_parameters(goal) - count_parameters(lr) == 2048


def test_different_goals_change_logits() -> None:
    torch.manual_seed(3)
    router = LearnedRouterGoal(width=32, n_experts=8, top_k=2)
    hidden = torch.randn(1, 4, 32)
    tokens_a = torch.tensor([[0, 10, 11, 12]])
    tokens_b = torch.tensor([[7, 10, 11, 12]])
    out_a = router(hidden, tokens=tokens_a)
    out_b = router(hidden, tokens=tokens_b)
    assert not torch.allclose(out_a.logits, out_b.logits)
