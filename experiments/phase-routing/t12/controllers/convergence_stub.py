"""Convergence-threshold halt — stateless per-item stub (T1; deepened T3).

I/O mirrors future trained arm. Uses T3 locks: ε=0.05, tau_easy/hard for
depths ≈4 / ≈10. Under silencing, difficulty is None → default tau.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from controllers.calibration import (
    CONV_EPS,
    CONV_T_MAX,
    CONV_TAU_DEFAULT,
    CONV_TAU_EASY,
    CONV_TAU_HARD,
)

T_MAX = CONV_T_MAX
EPS_DEFAULT = CONV_EPS


@dataclass
class ConvergenceDecision:
    depth: int
    eps: float
    rel_deltas: np.ndarray
    halted: bool
    stub: bool = True


class ConvergenceThresholdStub:
    """Stateless per-item convergence-threshold baseline stub."""

    name = "convergence_threshold"
    kind = "stateless_per_item_stub"

    def __init__(
        self,
        t_max: int = CONV_T_MAX,
        eps: float = CONV_EPS,
        tau_easy: float = CONV_TAU_EASY,
        tau_hard: float = CONV_TAU_HARD,
        tau_default: float = CONV_TAU_DEFAULT,
    ):
        self.t_max = int(t_max)
        self.eps = float(eps)
        self.tau_easy = float(tau_easy)
        self.tau_hard = float(tau_hard)
        self.tau_default = float(tau_default)

    def reset(self) -> None:
        return None

    def decide(
        self,
        item: Optional[Mapping[str, Any]] = None,
        *,
        difficulty: Optional[str] = None,
        rel_deltas: Optional[Sequence[float]] = None,
        silenced: bool = False,
        **_: Any,
    ) -> ConvergenceDecision:
        """Halt at first t where rel Δh < ε; if no series, use difficulty prior."""
        if rel_deltas is not None:
            series = np.asarray(rel_deltas, dtype=np.float64).reshape(-1)
            if series.size == 0:
                series = np.ones(self.t_max, dtype=np.float64)
        else:
            if silenced or difficulty is None:
                tau = self.tau_default
            elif difficulty == "easy":
                tau = self.tau_easy
            elif difficulty == "hard":
                tau = self.tau_hard
            else:
                tau = self.tau_default
            t = np.arange(1, self.t_max + 1, dtype=np.float64)
            series = np.exp(-t / tau)

        halt_at = self.t_max
        halted = False
        for i, d in enumerate(series[: self.t_max]):
            if float(d) < self.eps:
                halt_at = i + 1
                halted = True
                break
        return ConvergenceDecision(
            depth=int(halt_at),
            eps=self.eps,
            rel_deltas=np.asarray(series[: self.t_max], dtype=np.float64),
            halted=halted,
            stub=True,
        )


def dry_call(**kwargs: Any) -> ConvergenceDecision:
    return ConvergenceThresholdStub().decide(**kwargs)
