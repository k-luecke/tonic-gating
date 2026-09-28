"""T12 T2 disturbance grid — clean + noise + shift + silence cells (H4).

Shift cell carries T8 regression-lock spirit (δ reference 0.3).
Does not touch Affinity/C₃/hum. Smoke-scale only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np

from generators.easy_hard import (
    CONTENT_OFFSET,
    CONTENT_SIZE,
    GenConfig,
    build_split,
)

CellKind = Literal["clean", "noise", "shift", "silence"]

# Draft cell params (exact σ/δ free at T5 freeze; smoke locks below)
NOISE_SIGMA = 0.15  # fraction of content tokens flipped
SHIFT_DELTA = 0.3  # T8 regression-lock δ reference (content modular shift scale)
SHIFT_TOKENS = max(1, int(round(SHIFT_DELTA * CONTENT_SIZE)))  # discrete shift


@dataclass(frozen=True)
class DisturbanceCell:
    kind: CellKind
    sigma: float = 0.0
    delta: float = 0.0
    silence_mode: str = "none"  # none|zero|hold — for H4 silence cells
    note: str = ""


GRID: Dict[str, DisturbanceCell] = {
    "clean": DisturbanceCell(kind="clean", note="unmodified sequences"),
    "noise": DisturbanceCell(
        kind="noise", sigma=NOISE_SIGMA, note=f"content flip rate σ={NOISE_SIGMA}"
    ),
    "shift": DisturbanceCell(
        kind="shift",
        delta=SHIFT_DELTA,
        note=f"T8 regression-lock path δ={SHIFT_DELTA} → token shift {SHIFT_TOKENS}",
    ),
    "silence": DisturbanceCell(
        kind="silence",
        silence_mode="zero",
        note="H4 silence cell: sequences clean; estimate zeroed at controller input",
    ),
}


def apply_noise(sequences: np.ndarray, seed: int, sigma: float = NOISE_SIGMA) -> np.ndarray:
    """Flip a fraction of content tokens (task id at pos 0 preserved)."""
    rng = np.random.Generator(np.random.PCG64(seed + 17))
    out = np.array(sequences, dtype=np.int64, copy=True)
    N, L = out.shape
    for n in range(N):
        for t in range(1, L):
            if out[n, t] < CONTENT_OFFSET:
                continue
            if rng.random() < sigma:
                out[n, t] = CONTENT_OFFSET + int(rng.integers(CONTENT_SIZE))
    return out


def apply_shift(
    sequences: np.ndarray, delta: float = SHIFT_DELTA, tokens: Optional[int] = None
) -> np.ndarray:
    """Modular content shift (T8 shift regression lock path)."""
    k = int(tokens) if tokens is not None else max(1, int(round(delta * CONTENT_SIZE)))
    out = np.array(sequences, dtype=np.int64, copy=True)
    content = out >= CONTENT_OFFSET
    vals = out[content] - CONTENT_OFFSET
    out[content] = CONTENT_OFFSET + ((vals + k) % CONTENT_SIZE)
    return out


def apply_cell(
    sequences: np.ndarray,
    cell: DisturbanceCell,
    *,
    seed: int,
) -> Tuple[np.ndarray, Dict]:
    meta = {
        "kind": cell.kind,
        "sigma": cell.sigma,
        "delta": cell.delta,
        "silence_mode": cell.silence_mode,
        "note": cell.note,
    }
    if cell.kind == "clean":
        return np.array(sequences, dtype=np.int64, copy=True), meta
    if cell.kind == "noise":
        return apply_noise(sequences, seed=seed, sigma=cell.sigma), meta
    if cell.kind == "shift":
        return apply_shift(sequences, delta=cell.delta), meta
    if cell.kind == "silence":
        # sequences stay clean; silencing applied at controller gate (not here)
        return np.array(sequences, dtype=np.int64, copy=True), meta
    raise ValueError(f"unknown cell kind: {cell.kind}")


def build_grid_batch(
    seed: int,
    *,
    difficulties: Sequence[str] = ("easy", "hard"),
    cells: Optional[Iterable[str]] = None,
    n_per_task: int = 4,
    seq_len: int = 32,
) -> Dict[str, Dict[str, Dict]]:
    """Parity-ready grid: same seed → same base sequences; cells are transforms.

    Returns nested dict: difficulty → cell_name → {sequences, tasks, meta}
    """
    cell_names = list(cells) if cells is not None else list(GRID.keys())
    out: Dict[str, Dict[str, Dict]] = {}
    for diff in difficulties:
        base_seqs, tasks = build_split(
            GenConfig(
                seed=seed,
                n_per_task=n_per_task,
                seq_len=seq_len,
                difficulty=diff,  # type: ignore[arg-type]
            )
        )
        out[diff] = {}
        for name in cell_names:
            cell = GRID[name]
            seqs, meta = apply_cell(base_seqs, cell, seed=seed)
            out[diff][name] = {
                "sequences": seqs,
                "tasks": tasks.copy(),
                "meta": meta,
                "seed": seed,
                "difficulty": diff,
            }
    return out


def cell_list() -> List[str]:
    return list(GRID.keys())
