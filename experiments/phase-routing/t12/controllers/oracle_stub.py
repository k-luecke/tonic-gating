"""Oracle arm — report-only ceiling (ideal depth from true difficulty).

Not a support arm. Uses T1 T* (easy=4, hard=10). Affinity untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from controllers.calibration import ACT_EASY_CENTER, ACT_HARD_CENTER, ACT_PRIOR_MEAN, ACT_T_MAX
import numpy as np


@dataclass
class OracleDecision:
    depth: int
    mean_compute: float
    stub: bool = True
    name: str = "oracle"
    report_only: bool = True


class OracleStub:
    name = "oracle"
    kind = "report_only_ceiling"

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
        *,
        difficulty: Optional[str] = None,
        difficulty_true: Optional[str] = None,
        silenced: bool = False,
        **_: Any,
    ) -> OracleDecision:
        # Oracle may use true difficulty even on silence cells (report-only ceiling)
        d = difficulty_true or difficulty
        if d == "easy":
            center = self.easy_center
        elif d == "hard":
            center = self.hard_center
        else:
            center = self.prior_mean
        depth = int(np.clip(round(center), 1, self.t_max))
        return OracleDecision(depth=depth, mean_compute=float(center))


def dry_call(**kwargs: Any) -> OracleDecision:
    return OracleStub().decide(**kwargs)
