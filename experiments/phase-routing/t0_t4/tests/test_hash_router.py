import torch

from phase_routing.factory import make_model
from phase_routing.hash_router import HashRouter


def test_hash_router_is_parameter_free_and_deterministic() -> None:
    tokens = torch.tensor([[0, 8, 9, 63]])
    hidden = torch.randn(1, 4, 16)
    for k in (1, 2):
        router = HashRouter(16, top_k=k)
        first = router(hidden, tokens=tokens)
        second = router(hidden * 7, tokens=tokens)
        assert sum(p.numel() for p in router.parameters()) == 0
        assert torch.equal(first.indices, second.indices)
        torch.testing.assert_close(first.gates.sum(-1), torch.ones(1, 4))
        assert float(first.auxiliary_loss) == 0.0
        if k == 2:
            assert torch.all(first.indices[..., 0] != first.indices[..., 1])


def test_factory_supports_all_four_arms() -> None:
    for arm, k in (("lr", 2), ("hr", 1), ("hr2", 2), ("pg", 4)):
        model = make_model(arm, width=32, n_heads=4, ffn_width=64)
        tokens = torch.randint(8, 64, (2, 12)); tokens[:, 0] = torch.tensor([0, 7])
        logits, _ = model(tokens)
        assert logits.shape == (2, 12, 64)
        assert model.last_router_output.indices.shape[-1] == k
