"""ACT (Adaptive Computation Time) — stateless per-item stub (T1; deepened T3).

Same I/O as future trained ACT arm: decide(...) → depth.
No training. Uses T3 calibration locks (easy=4, hard=10, prior_mean=7).
Under silencing, difficulty is None → prior_mean (not live estimate).
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

T_MAX = ACT_T_MAX
PRIOR_MEAN = ACT_PRIOR_MEAN


@dataclass
class ACTDecision:
    depth: int
    halt_probs: np.ndarray  # length T_MAX
    mean_compute: float
    stub: bool = True


class ACTStub:
    """Stateless per-item ACT baseline stub."""

    name = "act"
    kind = "stateless_per_item_stub"

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
    ) -> ACTDecision:
        """Emit halt distribution + chosen depth (no recurrence / no state)."""
        if silenced or difficulty is None:
            if features is not None and not silenced:
                center = float(np.clip(np.mean(features) * self.t_max, 1.0, self.t_max))
            elif features is not None and silenced:
                f = np.asarray(features, dtype=np.float64).reshape(-1)
                if float(np.sum(np.abs(f))) < 1e-12:
                    center = self.prior_mean
                else:
                    hard_mass = float(f[1]) if f.size > 1 else float(np.mean(f))
                    center = (1.0 - hard_mass) * self.easy_center + hard_mass * self.hard_center
            else:
                center = self.prior_mean
        elif difficulty == "easy":
            center = self.easy_center
        elif difficulty == "hard":
            center = self.hard_center
        elif features is not None:
            center = float(np.clip(np.mean(features) * self.t_max, 1.0, self.t_max))
        else:
            center = self.prior_mean
        ts = np.arange(1, self.t_max + 1, dtype=np.float64)
        logits = -0.5 * ((ts - center) / 1.5) ** 2
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        depth = int(ts[int(np.argmax(probs))])
        mean_c = float(np.sum(ts * probs))
        return ACTDecision(depth=depth, halt_probs=probs, mean_compute=mean_c, stub=True)


def dry_call(**kwargs: Any) -> ACTDecision:
    return ACTStub().decide(**kwargs)
