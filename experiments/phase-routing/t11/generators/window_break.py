"""Window-break: harness carry metadata; break not planted in text."""

from __future__ import annotations

from typing import List

try:
    from .common import (
        CARRY_MODES,
        CarryMode,
        Placement,
        Segment,
        Sequence,
        WindowBreakMeta,
        make_family_segment,
        rng_for,
    )
except ImportError:
    from common import (
        CARRY_MODES,
        CarryMode,
        Placement,
        Segment,
        Sequence,
        WindowBreakMeta,
        make_family_segment,
        rng_for,
    )


def generate_window_break(
    seed: int,
    carry_mode: CarryMode = "cold",
    placement: Placement = "at_task_boundary",
) -> Sequence:
    """4 segments; break after index `at` (before next call).

    carry_mode and placement live only in window_break harness metadata.
    inside_task: break mid-segment conceptually; texts unchanged (metadata-only
    for this sealed batch — harness splits observation at score time).
    """
    if carry_mode not in CARRY_MODES:
        raise ValueError(f"unknown carry_mode {carry_mode!r}; expected one of {CARRY_MODES}")

    rng = rng_for(seed, "Window-break", carry_mode, placement)
    families = ["browser", "structured", "general", "browser"]
    notes = [
        "pre-break segment",
        "pre-break / break-adjacent",
        "first post-break segment (score M6a)",
        "second post-break segment",
    ]
    segments: List[Segment] = [
        make_family_segment(fam, rng, notes=note, use_miniwob=True)  # type: ignore[arg-type]
        for fam, note in zip(families, notes)
    ]

    # at=1 → break after second segment (task boundary Structured→General)
    # inside_task: same at index for sealed batch; placement flag distinguishes M6d
    at = 1
    if placement == "inside_task":
        # Mark break as occurring inside the segment at index `at` (harness splits).
        segments[at].notes = (segments[at].notes or "") + "; inside_task break target"
    wb = WindowBreakMeta(at=at, carry_mode=carry_mode, placement=placement)
    return Sequence(
        seed=seed,
        sequence_type="Window-break",
        segments=segments,
        window_break=wb,
    )
