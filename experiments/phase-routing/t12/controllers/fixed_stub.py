"""Fixed-depth stub (T=7 matched-mean baseline + arbitrary fixed)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from controllers.calibration import FIXED_T, LOCK


@dataclass
class FixedDecision:
    depth: int
    mean_compute: float
    stub: bool = True
    name: str = "fixed"


class FixedStub:
    name = "fixed"
    kind = "stateless_per_item_stub"

    def __init__(self, depth: int = FIXED_T):
        self.depth = int(depth)

    def reset(self) -> None:
        return None

    def decide(self, **_: Any) -> FixedDecision:
        d = self.depth
        return FixedDecision(depth=d, mean_compute=float(d), stub=True, name=f"fixed_t{d}")


def dry_call(depth: int = FIXED_T, **kwargs: Any) -> FixedDecision:
    return FixedStub(depth=depth).decide(**kwargs)
