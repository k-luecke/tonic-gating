import torch

from t7_routing.factory import make_model
from t7_routing.hash_router import HashRouter


def test_hash_router_parameter_free() -> None:
    tokens = torch.tensor([[0, 8, 9, 63]])
    hidden = torch.randn(1, 4, 16)
    router = HashRouter(16, top_k=1)
    first = router(hidden, tokens=tokens)
    second = router(hidden * 7, tokens=tokens)
    assert sum(p.numel() for p in router.parameters()) == 0
    assert torch.equal(first.indices, second.indices)
