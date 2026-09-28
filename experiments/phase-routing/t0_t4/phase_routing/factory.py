"""Small router/model factory used by phase-routing experiments."""
from __future__ import annotations

from torch import nn

from .model import LearnedRouter, MoETransformer
from .hash_router import HashRouter
from .phase_gate import PhaseGateRouter


def router_class(name: str) -> type[nn.Module]:
    normalized = name.lower().replace("-", "_")
    if normalized in {"lr", "learned", "learned_router"}:
        return LearnedRouter
    if normalized in {"hr", "hr1", "hash", "hash_router", "hr2", "hash2"}:
        return HashRouter
    if normalized in {"pg", "phase", "phase_gate", "t3"}:
        return PhaseGateRouter
    raise ValueError(f"unknown router: {name}")


def make_model(router: str = "learned", **model_kwargs: object) -> MoETransformer:
    """Construct a model while retaining MoETransformer's existing defaults."""
    normalized = router.lower().replace("-", "_")
    if normalized in {"hr", "hr1", "hash", "hash_router"}:
        model_kwargs["top_k"] = 1
    elif normalized in {"hr2", "hash2"}:
        model_kwargs["top_k"] = 2
    return MoETransformer(router_cls=router_class(router), **model_kwargs)
