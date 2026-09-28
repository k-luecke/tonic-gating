"""Ambiguous sequences: task clear only after first 30–60% of the segment."""

from __future__ import annotations

from typing import List

try:
    from .common import (
        Family,
        Segment,
        Sequence,
        make_family_segment,
        rng_for,
        segment_count_for,
    )
except ImportError:
    from common import (
        Family,
        Segment,
        Sequence,
        make_family_segment,
        rng_for,
        segment_count_for,
    )

FAMILIES: List[Family] = ["browser", "structured", "general"]


def generate_ambiguous(seed: int) -> Sequence:
    rng = rng_for(seed, "Ambiguous")
    n = segment_count_for(seed, 4, 6)
    order = FAMILIES.copy()
    rng.shuffle(order)
    families = [order[i % len(order)] for i in range(n)]
    segments: List[Segment] = [
        make_family_segment(
            fam,
            rng,
            notes=f"ambiguous: {fam} cue buried mid-segment",
            ambiguous=True,
            use_miniwob=False,
        )
        for fam in families
    ]
    return Sequence(seed=seed, sequence_type="Ambiguous", segments=segments)
