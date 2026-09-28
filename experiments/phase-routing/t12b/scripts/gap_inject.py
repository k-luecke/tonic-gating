#!/usr/bin/env python3
"""T12b evidence gap / lag injection + G10 in-gap spurious flip.

Rule 19 G1: gap steps are zero affinity EXCEPT the locked G10 single-frame
mid-gap spurious ScoreAffinity flip (wrong specialist peak one frame).
Does not retune C₃/hum; does not rewrite sealed generators.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

GAP_LENGTHS = (3, 6, 12, 24)
LAG_LENGTHS = (2, 4, 8)


@dataclass(frozen=True)
class GapSpec:
    start: int
    length: int

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass(frozen=True)
class GapPerturbation:
    """G10: single-frame spurious ScoreAffinity flip mid-gap."""

    offset: int  # index within gap [0, length)
    wrong_spec_index: int  # 0..2
    peak: float = 1.0


def _rng(*parts: int) -> np.random.Generator:
    h = hashlib.sha256()
    for p in parts:
        h.update(int(p).to_bytes(8, "little", signed=False))
    seed = int.from_bytes(h.digest()[:8], "little")
    return np.random.default_rng(seed)


def choose_gap_start(
    n_steps: int,
    gap_len: int,
    *,
    seed: int,
    seq_id: int,
    seg_index: int,
    evidence_point: int = 0,
) -> Optional[int]:
    lo = max(0, int(evidence_point))
    hi = int(n_steps) - int(gap_len)
    if hi < lo:
        return None
    rng = _rng(seed, seq_id, seg_index, gap_len, 436480)
    return int(rng.integers(lo, hi + 1))


def mid_gap_offset(gap_len: int) -> int:
    """Primary G10 lock: single frame at mid-gap."""
    return int(gap_len) // 2


def choose_spurious_spec(
    *,
    seed: int,
    seq_id: int,
    true_spec_index: Optional[int] = None,
) -> int:
    """Wrong specialist index ≠ true (if known)."""
    rng = _rng(seed, seq_id, 91011)
    choices = [0, 1, 2]
    if true_spec_index is not None and int(true_spec_index) in choices:
        choices = [c for c in choices if c != int(true_spec_index)]
    return int(rng.choice(choices))


def inject_gap(
    affinities: Sequence[Sequence[float]],
    gap: GapSpec,
    *,
    assert_zero: bool = True,
    perturbation: Optional[GapPerturbation] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Zero affinity in gap; optionally apply G10 single-frame flip.

    Returns (modified, gap_mask). When perturbation set, that one frame is
    non-zero (wrong specialist peak); surrounding gap frames stay zero.
    """
    arr = np.asarray(affinities, dtype=np.float64).copy()
    if arr.ndim != 2:
        raise ValueError(f"affinities must be (T, D); got shape {arr.shape}")
    T, D = arr.shape
    mask = np.zeros(T, dtype=bool)
    s, e = max(0, gap.start), min(T, gap.end)
    if e <= s:
        return arr, mask
    mask[s:e] = True
    arr[mask] = 0.0
    if perturbation is not None:
        off = int(perturbation.offset)
        if off < 0 or off >= (e - s):
            raise ValueError(f"perturbation offset {off} out of gap length {e - s}")
        t = s + off
        vec = np.zeros(D, dtype=np.float64)
        idx = int(perturbation.wrong_spec_index) % max(D, 1)
        if D >= 3:
            vec[idx] = float(perturbation.peak)
        else:
            vec[0] = float(perturbation.peak)
        arr[t] = vec[:D]
    if assert_zero:
        assert_gaps_zero(arr, mask, allow_perturbation_at=(
            None if perturbation is None else (s + int(perturbation.offset),)
        ))
    return arr, mask


def inject_gap_with_g10(
    affinities: Sequence[Sequence[float]],
    gap: GapSpec,
    *,
    seed: int,
    seq_id: int,
    true_spec_index: Optional[int] = None,
    peak: float = 1.0,
    assert_zero: bool = True,
) -> Tuple[np.ndarray, np.ndarray, GapPerturbation]:
    """Convenience: mid-gap spurious flip (G10 primary)."""
    pert = GapPerturbation(
        offset=mid_gap_offset(gap.length),
        wrong_spec_index=choose_spurious_spec(
            seed=seed, seq_id=seq_id, true_spec_index=true_spec_index
        ),
        peak=peak,
    )
    mod, mask = inject_gap(affinities, gap, assert_zero=assert_zero, perturbation=pert)
    return mod, mask, pert


def assert_gaps_zero(
    affinities: np.ndarray,
    gap_mask: np.ndarray,
    *,
    allow_perturbation_at: Optional[Sequence[int]] = None,
) -> None:
    """G1 — gap steps zero except optional G10 frame indices."""
    if not gap_mask.any():
        return
    allow = set(int(i) for i in (allow_perturbation_at or ()))
    idxs = np.where(gap_mask)[0]
    for i in idxs:
        if int(i) in allow:
            # must be NON-zero (spurious peak present)
            if float(np.abs(affinities[i]).sum()) <= 0.0:
                raise AssertionError(f"G10 frame {i} is zero — expected spurious peak")
            continue
        if not np.all(affinities[i] == 0.0):
            bad = float(np.max(np.abs(affinities[i])))
            raise AssertionError(f"gap affinity not zero at {i} (max |aff|={bad})")


def apply_lag(
    affinities: Sequence[Sequence[float]],
    lag: int,
    *,
    zero_pad: bool = True,
) -> np.ndarray:
    arr = np.asarray(affinities, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"affinities must be (T, D); got shape {arr.shape}")
    if lag <= 0:
        return arr.copy()
    out = np.zeros_like(arr)
    if zero_pad:
        out[lag:] = arr[:-lag]
    else:
        out[lag:] = arr[:-lag]
        out[:lag] = arr[0]
    return out


def self_test() -> None:
    T, D = 40, 3
    base = np.ones((T, D), dtype=np.float64)
    start = choose_gap_start(T, 6, seed=20260923, seq_id=1, seg_index=0, evidence_point=10)
    assert start is not None and start >= 10
    gap = GapSpec(start=start, length=6)
    mod, mask = inject_gap(base, gap, assert_zero=True)
    assert mask.sum() == 6
    assert np.all(mod[mask] == 0.0)
    assert np.all(mod[~mask] == 1.0)
    # G10 mid-gap flip
    mod2, mask2, pert = inject_gap_with_g10(
        base, gap, seed=20260923, seq_id=1, true_spec_index=0, peak=1.0
    )
    assert mask2.sum() == 6
    t_flip = gap.start + pert.offset
    assert float(np.abs(mod2[t_flip]).sum()) > 0.0
    assert int(np.argmax(mod2[t_flip])) == pert.wrong_spec_index
    assert pert.wrong_spec_index != 0  # true was 0
    # surrounding gap zeros
    for i in np.where(mask2)[0]:
        if int(i) == t_flip:
            continue
        assert np.all(mod2[i] == 0.0)
    # non-zero non-G10 inside gap must fail
    bad = mod.copy()
    bad[start] = 0.5
    try:
        assert_gaps_zero(bad, mask)
        raise SystemExit("expected AssertionError for non-zero gap")
    except AssertionError:
        pass
    lagged = apply_lag(base, 4, zero_pad=True)
    assert np.all(lagged[:4] == 0.0)
    assert np.all(lagged[4:] == 1.0)
    for g in GAP_LENGTHS:
        assert g in (3, 6, 12, 24)
    for L in LAG_LENGTHS:
        assert L in (2, 4, 8)
    print("gap_inject self-test PASS (incl. G10 mid-gap spurious flip)")
    print(f"  sample gap start={start} len=6 flip_t={t_flip} wrong_spec={pert.wrong_spec_index}")
    print(f"  GAP_LENGTHS={GAP_LENGTHS} LAG_LENGTHS={LAG_LENGTHS}")


def main() -> int:
    ap = argparse.ArgumentParser(description="T12b gap/lag inject utilities")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
