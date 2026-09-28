"""Clean sequences: task clear in first tokens; one family per segment."""

from __future__ import annotations

from typing import List

try:
    from .common import (
        FAMILY_TO_SPECIALIST,
        Family,
        Segment,
        Sequence,
        make_family_segment,
        rng_for,
        segment_count_for,
    )
except ImportError:
    from common import (
        FAMILY_TO_SPECIALIST,
        Family,
        Segment,
        Sequence,
        make_family_segment,
        rng_for,
        segment_count_for,
    )

FAMILIES: List[Family] = ["browser", "structured", "general"]


def generate_clean(seed: int) -> Sequence:
    """4–6 segments; one family per segment; cues in first tokens."""
    rng = rng_for(seed, "Clean")
    n = segment_count_for(seed, 4, 6)
    order = FAMILIES.copy()
    rng.shuffle(order)
    # Extend to n by cycling shuffled order
    families = [order[i % len(order)] for i in range(n)]
    segments: List[Segment] = [
        make_family_segment(
            fam,
            rng,
            notes="clean: cue words in first tokens",
            ambiguous=False,
            use_miniwob=True,
        )
        for fam in families
    ]
    # Identity check
    for seg in segments:
        assert seg.oracle_specialist == FAMILY_TO_SPECIALIST[seg.family]
    return Sequence(seed=seed, sequence_type="Clean", segments=segments)
