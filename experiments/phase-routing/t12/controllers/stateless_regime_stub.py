"""Stateless regime arm — H4 control; instantaneous readout, no persistence.

Under silence cells it still reads whatever is at the controller input
(zeros/hold), so it cannot recover the live regime — SD compares coordinator
persistence vs this instantaneous read.
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
class StatelessDecision:
    depth: int
    mean_compute: float
    used_features: bool
    stub: bool = True
    name: str = "stateless_regime"


class StatelessRegimeStub:
    name = "stateless_regime"
    kind = "stateless_regime_stub"

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
    ) -> StatelessDecision:
        # Purely instantaneous: prefer features channel; under zero-silence → prior
        if features is not None:
            f = np.asarray(features, dtype=np.float64).reshape(-1)
            if silenced and float(np.sum(np.abs(f))) < 1e-12:
                depth = int(round(self.prior_mean))
                return StatelessDecision(
                    depth=int(np.clip(depth, 1, self.t_max)),
                    mean_compute=float(self.prior_mean),
                    used_features=True,
                )
            hard_mass = float(f[1]) if f.size > 1 else float(np.clip(np.mean(f), 0, 1))
            center = (1.0 - hard_mass) * self.easy_center + hard_mass * self.hard_center
            depth = int(np.clip(round(center), 1, self.t_max))
            return StatelessDecision(
                depth=depth, mean_compute=float(center), used_features=True
            )
        if difficulty == "easy":
            return StatelessDecision(
                depth=int(self.easy_center),
                mean_compute=float(self.easy_center),
                used_features=False,
            )
        if difficulty == "hard":
            return StatelessDecision(
                depth=int(self.hard_center),
                mean_compute=float(self.hard_center),
                used_features=False,
            )
        d = int(round(self.prior_mean))
        return StatelessDecision(
            depth=d, mean_compute=float(self.prior_mean), used_features=False
        )


def dry_call(**kwargs: Any) -> StatelessDecision:
    return StatelessRegimeStub().decide(**kwargs)
