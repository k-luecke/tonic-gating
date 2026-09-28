"""T7 — inferred φ_ref phase gate on harder order-3 noisy recurrence."""
from .factory import make_model, router_class
from .hash_router import HashRouter
from .model import LearnedRouter, LearnedRouterGoal, MoETransformer, count_parameters
from .phase_gate import PhaseGateRouter
from .phase_gate_inferred import PhaseGateInferredRouter

__all__ = [
    "HashRouter",
    "LearnedRouter",
    "LearnedRouterGoal",
    "MoETransformer",
    "PhaseGateRouter",
    "PhaseGateInferredRouter",
    "count_parameters",
    "make_model",
    "router_class",
]
