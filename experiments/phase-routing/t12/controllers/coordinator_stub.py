"""Coordinator regime stub — budgets depth from regime estimate (when not silenced).

Under H4 silence, falls back to prior_mean (does not see live estimate).
Same decide() I/O shape as other arms.
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
class CoordinatorDecision:
    depth: int
    mean_compute: float
    used_regime: bool
    stub: bool = True
    name: str = "coordinator_regime"


class CoordinatorRegimeStub:
    name = "coordinator_regime"
    kind = "regime_budget_stub"

    def __init__(
        self,
        t_max: int = ACT_T_MAX,
        prior_mean: float = ACT_PRIOR_MEAN,
        easy_center: float = ACT_EASY_CENTER,
        hard_center: float = ACT_HARD_CENTER,
    ):
        self.t_max = int(t_max)
        self.prior_mean = float(prior_mean)
        self.easy_center = float(easy_center)
        self.hard_center = float(hard_center)

    def reset(self) -> None:
        return None

    def decide(
        self,
        item: Optional[Mapping[str, Any]] = None,
        *,
        difficulty: Optional[str] = None,
        features: Optional[Sequence[float]] = None,
        silenced: bool = False,
        **_: Any,
    ) -> CoordinatorDecision:
        if silenced or difficulty is None:
            # H4: no live estimate at controller input → prior mean hold
            depth = int(round(self.prior_mean))
            return CoordinatorDecision(
                depth=int(np.clip(depth, 1, self.t_max)),
                mean_compute=float(self.prior_mean),
                used_regime=False,
            )
        if difficulty == "easy":
            center = self.easy_center
        elif difficulty == "hard":
            center = self.hard_center
        elif features is not None:
            f = np.asarray(features, dtype=np.float64).reshape(-1)
            # channel[1] ≈ hard mass when unsilenced
            hard_mass = float(f[1]) if f.size > 1 else float(np.mean(f))
            center = (1.0 - hard_mass) * self.easy_center + hard_mass * self.hard_center
        else:
            center = self.prior_mean
        depth = int(np.clip(round(center), 1, self.t_max))
        return CoordinatorDecision(
            depth=depth, mean_compute=float(center), used_regime=True
        )


def dry_call(**kwargs: Any) -> CoordinatorDecision:
    return CoordinatorRegimeStub().decide(**kwargs)
