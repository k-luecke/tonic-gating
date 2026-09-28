"""T7 routing/expert cost definitions (native units, never collapsed)."""
from __future__ import annotations

from typing import Any

WIDTH = 128
EXPERT_HIDDEN = 256
N_EXPERTS = 8
EXPERT_MACS_PER_EXPERT = WIDTH * EXPERT_HIDDEN + EXPERT_HIDDEN * WIDTH  # 65536

LR_ROUTE_MACS_PER_TOKEN = WIDTH * N_EXPERTS  # 1024
LR_GOAL_ROUTE_MACS_PER_TOKEN = (2 * WIDTH) * N_EXPERTS  # 2048

# Supplied PG (secondary): 8 cosine comparisons.
PG_ROUTE_OPS = N_EXPERTS  # 8

# PG-inferred: Linear(width→2) = 2*width MACs + 8 cosine comparisons.
# Document honestly — do NOT claim C_route=8.
PG_INFERRED_PHASE_HEAD_MACS = 2 * WIDTH  # 256
PG_INFERRED_COSINE_OPS = N_EXPERTS  # 8


def routing_decision_cost(arm: str) -> dict[str, Any]:
    normalized = arm.lower().replace("-", "_")
    if normalized == "lr":
        return {
            "C_route_MACs_per_token": float(LR_ROUTE_MACS_PER_TOKEN),
            "C_route_unit": "MACs",
            "C_route_ops": None,
            "C_route_hash_ops": None,
            "C_route_definition": (
                "LR routing decision = hidden_width × E = 128×8 = 1024 MACs/token"
            ),
        }
    if normalized in {"lr_goal", "lrgoal"}:
        return {
            "C_route_MACs_per_token": float(LR_GOAL_ROUTE_MACS_PER_TOKEN),
            "C_route_unit": "MACs",
            "C_route_ops": None,
            "C_route_hash_ops": None,
            "C_route_definition": (
                "LR-goal routing decision = (2×hidden_width) × E = 256×8 = 2048 MACs/token; "
                "concat(hidden, router-only goal_embedding[task_id]) into the "
                "router linear map only"
            ),
        }
    if normalized in {"hr1", "hr"}:
        return {
            "C_route_MACs_per_token": 0.0,
            "C_route_unit": "MACs",
            "C_route_ops": None,
            "C_route_hash_ops": 1,
            "C_route_definition": (
                "HR1 routing decision = table/hash lookup; 0 MACs"
            ),
        }
    if normalized == "hr2":
        return {
            "C_route_MACs_per_token": 0.0,
            "C_route_unit": "MACs",
            "C_route_ops": None,
            "C_route_hash_ops": 2,
            "C_route_definition": (
                "HR2 routing decision = two table/hash lookups; 0 MACs"
            ),
        }
    if normalized in {"pg", "pg_supplied"}:
        return {
            "C_route_MACs_per_token": None,
            "C_route_unit": "primitive_cosine_comparisons",
            "C_route_ops": float(PG_ROUTE_OPS),
            "C_route_hash_ops": None,
            "C_route_definition": (
                "Supplied PG (secondary): E=8 cosine comparisons + marker→phase table; "
                "NOT by-construction for T7 primary (secondary disclosure only)"
            ),
        }
    if normalized in {"pg_inferred", "pgi"}:
        return {
            "C_route_MACs_per_token": float(PG_INFERRED_PHASE_HEAD_MACS),
            "C_route_unit": "MACs_plus_cosine_ops",
            "C_route_ops": float(PG_INFERRED_COSINE_OPS),
            "C_route_hash_ops": None,
            "C_route_phase_head_MACs": float(PG_INFERRED_PHASE_HEAD_MACS),
            "C_route_definition": (
                "PG-inferred routing decision = Linear(width→2) = 2×128 = 256 MACs/token "
                "for the phase head, plus 8 half-wave cosine comparisons + normalize/top-4. "
                "NOT C_route=8. Cost clause informational if ops aren't comparable to LR MACs; "
                "report native units, never combine. Noise/H2 are NOT by-construction "
                "(φ_ref comes from noised hidden)."
            ),
        }
    raise ValueError(f"unknown arm for routing cost: {arm}")


def expert_compute_cost(mean_active_experts_per_token: float) -> dict[str, Any]:
    mean_active = float(mean_active_experts_per_token)
    return {
        "mean_active_experts_per_token": mean_active,
        "expert_MACs_per_expert": float(EXPERT_MACS_PER_EXPERT),
        "C_exp_MACs_per_token": mean_active * float(EXPERT_MACS_PER_EXPERT),
        "C_exp_definition": (
            "mean_active_experts_per_token × expert_MACs_per_expert "
            f"({EXPERT_MACS_PER_EXPERT} = 128×256 + 256×128; activation ignored)"
        ),
    }


DISTURBANCE_GRID = (0.0, 0.1, 0.3, 1.0)
PRIMARY_ROBUSTNESS_UNITS = 0.3
