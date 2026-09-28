"""Router/model factory for T7 arms."""
from __future__ import annotations

from torch import nn

from .hash_router import HashRouter
from .model import LearnedRouter, LearnedRouterGoal, MoETransformer
from .phase_gate import PhaseGateRouter
from .phase_gate_inferred import PhaseGateInferredRouter


def router_class(name: str) -> type[nn.Module]:
    normalized = name.lower().replace("-", "_")
    if normalized in {"lr", "learned", "learned_router"}:
        return LearnedRouter
    if normalized in {"lr_goal", "learned_router_goal", "lrgoal"}:
        return LearnedRouterGoal
    if normalized in {"hr", "hr1", "hash", "hash_router", "hr2", "hash2"}:
        return HashRouter
    if normalized in {"pg", "phase", "phase_gate", "pg_supplied"}:
        return PhaseGateRouter
    if normalized in {"pg_inferred", "pgi", "phase_gate_inferred", "inferred"}:
        return PhaseGateInferredRouter
    raise ValueError(f"unknown router: {name}")


def make_model(router: str = "learned", **model_kwargs: object) -> MoETransformer:
    normalized = router.lower().replace("-", "_")
    if normalized in {"hr", "hr1", "hash", "hash_router"}:
        model_kwargs["top_k"] = 1
    elif normalized in {"hr2", "hash2"}:
        model_kwargs["top_k"] = 2
    return MoETransformer(router_cls=router_class(router), **model_kwargs)
