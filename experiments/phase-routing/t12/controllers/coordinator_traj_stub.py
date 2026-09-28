"""Coordinator regime+traj stub — regime budget + within-example trajectory nudge.

H2 arm. Same decide() I/O as other stubs. Affinity/C₃/hum untouched.
Under H4 silence: prior_mean (no live estimate / traj at controller input).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from controllers.calibration import (
    ACT_EASY_CENTER,
    ACT_HARD_CENTER,
    ACT_PRIOR_MEAN,
    ACT_T_MAX,
)


@dataclass
class CoordinatorTrajDecision:
    depth: int
    mean_compute: float
    used_regime: bool
    used_traj: bool
    stub: bool = True
    name: str = "coordinator_regime_traj"


class CoordinatorRegimeTrajStub:
    name = "coordinator_regime_traj"
    kind = "regime_traj_budget_stub"

    def __init__(
        self,
        t_max: int = ACT_T_MAX,
        prior_mean: float = ACT_PRIOR_MEAN,
        easy_center: float = ACT_EASY_CENTER,
        hard_center: float = ACT_HARD_CENTER,
        traj_nudge_max: float = 1.0,
    ):
        self.t_max = int(t_max)
        self.prior_mean = float(prior_mean)
        self.easy_center = float(easy_center)
        self.hard_center = float(hard_center)
        self.traj_nudge_max = float(traj_nudge_max)

    def reset(self) -> None:
        return None

    def decide(
        self,
        item: Optional[Mapping[str, Any]] = None,
        *,
        difficulty: Optional[str] = None,
        features: Optional[Sequence[float]] = None,
        traj_features: Optional[Sequence[float]] = None,
        rel_deltas: Optional[Sequence[float]] = None,
        silenced: bool = False,
        **_: Any,
    ) -> CoordinatorTrajDecision:
        if silenced or difficulty is None:
            depth = int(round(self.prior_mean))
            return CoordinatorTrajDecision(
                depth=int(np.clip(depth, 1, self.t_max)),
                mean_compute=float(self.prior_mean),
                used_regime=False,
                used_traj=False,
            )
        if difficulty == "easy":
            center = self.easy_center
            used_regime = True
        elif difficulty == "hard":
            center = self.hard_center
            used_regime = True
        elif features is not None:
            f = np.asarray(features, dtype=np.float64).reshape(-1)
            hard_mass = float(f[1]) if f.size > 1 else float(np.mean(f))
            center = (1.0 - hard_mass) * self.easy_center + hard_mass * self.hard_center
            used_regime = True
        else:
            center = self.prior_mean
            used_regime = False

        # Trajectory nudge from rel_deltas / traj_features (within-example), clipped ±1
        used_traj = False
        nudge = 0.0
        series = None
        if traj_features is not None:
            series = np.asarray(traj_features, dtype=np.float64).reshape(-1)
        elif rel_deltas is not None:
            series = np.asarray(rel_deltas, dtype=np.float64).reshape(-1)
        if series is not None and series.size > 0:
            # Slow convergence (high late Δ) → spend more; fast → spend less
            late = float(np.mean(series[-max(1, series.size // 3) :]))
            nudge = float(np.clip((late - 0.1) * 5.0, -self.traj_nudge_max, self.traj_nudge_max))
            used_traj = True
        center = float(np.clip(center + nudge, 1.0, self.t_max))
        depth = int(np.clip(round(center), 1, self.t_max))
        return CoordinatorTrajDecision(
            depth=depth,
            mean_compute=float(center),
            used_regime=used_regime,
            used_traj=used_traj,
        )


def dry_call(**kwargs: Any) -> CoordinatorTrajDecision:
    return CoordinatorRegimeTrajStub().decide(**kwargs)
