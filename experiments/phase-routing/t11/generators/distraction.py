"""Distraction: short off-task span inside a segment; correct action is stay.

T11.2 Rule 19: inserts MUST carry other-family keyword cues so a stateless
arm that re-scores mid-span can thrash (switch away and back). Neutral
weather/clock inserts (legacy DISTRACTION_INSERTS) do not tempt and are
not used for T11.2 counting candidates.
"""

from __future__ import annotations

from typing import List, Tuple

try:
    from .common import (
        DISTRACTION_OTHER_FAMILY,
        Family,
        Segment,
        Sequence,
        make_family_segment,
        rng_for,
        segment_count_for,
    )
except ImportError:
    from common import (
        DISTRACTION_OTHER_FAMILY,
        Family,
        Segment,
        Sequence,
        make_family_segment,
        rng_for,
        segment_count_for,
    )

FAMILIES: List[Family] = ["browser", "structured", "general"]


def _other_families(fam: Family) -> List[Family]:
    return [f for f in FAMILIES if f != fam]


def _inject(base: str, insert: str, at_frac: float = 0.4) -> Tuple[str, dict]:
    words = base.split()
    at = max(1, int(len(words) * at_frac))
    left = words[:at]
    right = words[at:]
    insert_words = insert.strip().split()
    text = " ".join(left + insert_words + right)
    span = {
        "start_token": at,
        "end_token": at + len(insert_words),
        "n_tokens": len(insert_words),
        "other_family_cues": True,
    }
    return text, span


def generate_distraction(seed: int) -> Sequence:
    rng = rng_for(seed, "Distraction")
    n = segment_count_for(seed, 4, 6)
    order = FAMILIES.copy()
    rng.shuffle(order)
    families = [order[i % len(order)] for i in range(n)]
    segments: List[Segment] = []
    for fam in families:
        base = make_family_segment(
            fam,
            rng,
            notes=f"distraction: stay on {fam}; other-family cues in span",
            ambiguous=False,
            use_miniwob=(fam == "browser"),
        )
        other = rng.choice(_other_families(fam))
        insert = rng.choice(DISTRACTION_OTHER_FAMILY[other])
        text, span = _inject(base.text, insert)
        span["cue_family"] = other
        # Sealed distraction spans are 6–9 tokens ({6,8,9}); all well inside T_hold(n=3)=65.
        if not (6 <= span["n_tokens"] <= 9):
            raise ValueError(f"distraction span outside sealed 6–9 range: {span['n_tokens']}")
        segments.append(
            Segment(
                family=base.family,
                oracle_specialist=base.oracle_specialist,
                text=text,
                notes=base.notes,
                distraction_span=span,
                browser_episode=base.browser_episode,
            )
        )
    return Sequence(seed=seed, sequence_type="Distraction", segments=segments)
