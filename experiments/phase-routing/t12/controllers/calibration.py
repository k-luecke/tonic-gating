"""T12 T3 calibration locks — ε / ACT priors / Convergence threshold.

Target: C̄ = 7 ± 0.25 on a balanced easy/hard smoke batch.
Locks match T1 T* budgets: easy=4, hard=10 (no soft-hard-to-8 deviation).
Equal-mix mean of 4/10 = 7 → Fixed T=7 matched-mean baseline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

# --- Locked from T1 (measurement; read-only reference) ---
T_STAR_EASY = 4
T_STAR_HARD = 10
T_STAR_GAP = 6

# --- Calibration target ---
C_BAR_TARGET = 7.0
C_BAR_TOL = 0.25

# --- ACT priors (T3 lock; amended 2026-09-23 PT — recalibrate to T1) ---
ACT_T_MAX = 12
ACT_PRIOR_MEAN = 7.0  # matched-mean spirit / Fixed T=7 baseline
ACT_EASY_CENTER = 4.0  # = T*_easy
ACT_HARD_CENTER = 10.0  # = T*_hard (soft-to-8 removed)

# --- Convergence threshold (T3 lock) ---
# Halt when rel Δh < ε. Synthetic tau priors → depths ≈ easy 4 / hard 10
# under exp(-t/tau) < ε  ⇒  t ≈ -tau * ln(ε)
CONV_T_MAX = 12
CONV_EPS = 0.05
CONV_TAU_EASY = 1.335  # ≈4 steps at ε=0.05
CONV_TAU_HARD = 3.337  # ≈10 steps at ε=0.05
CONV_TAU_DEFAULT = 2.336  # ≈7 — near prior_mean spirit

# --- Fixed baseline ---
FIXED_T = 7


@dataclass(frozen=True)
class CalibrationLock:
    c_bar_target: float = C_BAR_TARGET
    c_bar_tol: float = C_BAR_TOL
    act_t_max: int = ACT_T_MAX
    act_prior_mean: float = ACT_PRIOR_MEAN
    act_easy_center: float = ACT_EASY_CENTER
    act_hard_center: float = ACT_HARD_CENTER
    conv_t_max: int = CONV_T_MAX
    conv_eps: float = CONV_EPS
    conv_tau_easy: float = CONV_TAU_EASY
    conv_tau_hard: float = CONV_TAU_HARD
    conv_tau_default: float = CONV_TAU_DEFAULT
    fixed_t: int = FIXED_T
    t_star_easy: int = T_STAR_EASY
    t_star_hard: int = T_STAR_HARD
    t_star_gap: int = T_STAR_GAP
    note: str = (
        "T3 lock amended 2026-09-23 PT: C̄=7±0.25; ACT easy/hard centers 4/10 "
        "(T1 T*); CONV ε=0.05 tau_e/h=1.335/3.337; Fixed T=7. "
        "Soft-hard-to-8 removed. Affinity/C₃/hum untouched."
    )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


LOCK = CalibrationLock()


def within_c_bar(c_bar: float) -> bool:
    return abs(float(c_bar) - C_BAR_TARGET) <= C_BAR_TOL
