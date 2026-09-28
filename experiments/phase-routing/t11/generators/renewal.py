"""Renewal: regime A, then B long enough to demote A, then A again."""

from __future__ import annotations

from typing import List

try:
    from .common import (
        Family,
        Segment,
        Sequence,
        make_family_segment,
        rng_for,
    )
except ImportError:
    from common import (
        Family,
        Segment,
        Sequence,
        make_family_segment,
        rng_for,
    )

FAMILIES: List[Family] = ["browser", "structured", "general"]


def generate_renewal(seed: int) -> Sequence:
    """Pattern: A → B → B → A (return) → trailing C."""
    rng = rng_for(seed, "Renewal")
    shuffled = FAMILIES.copy()
    rng.shuffle(shuffled)
    a, b, c = shuffled[0], shuffled[1], shuffled[2]
    segments: List[Segment] = [
        make_family_segment(a, rng, notes="renewal: regime A start", use_miniwob=True),
        make_family_segment(b, rng, notes="renewal: regime B (demote A toward warm/cold)", use_miniwob=True),
        make_family_segment(b, rng, notes="renewal: regime B continues", use_miniwob=True),
        make_family_segment(a, rng, notes="renewal: regime A returns", use_miniwob=True),
        make_family_segment(c, rng, notes="renewal: trailing unrelated regime", use_miniwob=True),
    ]
    return Sequence(seed=seed, sequence_type="Renewal", segments=segments)
