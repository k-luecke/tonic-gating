"""Parity harness — same seed → same sequences for all arms; matched init family.

T2 acceptance: arms share batch draws and init family; only controller policy differs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from controllers.silencing import SilencingGate, apply_silence_to_decide_kwargs
from generators.disturbance import GRID, build_grid_batch, cell_list

# Draft five-seed list (T8/T11 family intent); smoke may use 1–2
DRAFT_SEEDS: List[int] = [20260921, 20260922, 20260923, 20260924, 20260925]
INIT_FAMILY = "pcg64_t12_v1"
ARMS_DEFAULT = (
    "fixed_t7",
    "act",
    "convergence",
    "coordinator_regime",
    "stateless_regime",
)


@dataclass
class ArmBatch:
    arm: str
    seed: int
    difficulty: str
    cell: str
    sequences: np.ndarray
    tasks: np.ndarray
    decide_kwargs: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)


class ParityHarness:
    """Build matched batches for every arm under a seed × difficulty × cell grid."""

    def __init__(
        self,
        *,
        arms: Sequence[str] = ARMS_DEFAULT,
        init_family: str = INIT_FAMILY,
        n_per_task: int = 4,
        seq_len: int = 32,
        silence_mode_for_silence_cell: str = "zero",
    ):
        self.arms = list(arms)
        self.init_family = init_family
        self.n_per_task = int(n_per_task)
        self.seq_len = int(seq_len)
        self.silence_mode_for_silence_cell = silence_mode_for_silence_cell

    def grid(
        self, seed: int, cells: Optional[Sequence[str]] = None
    ) -> Dict[str, Dict[str, Dict]]:
        return build_grid_batch(
            seed,
            cells=cells,
            n_per_task=self.n_per_task,
            seq_len=self.seq_len,
        )

    def iter_arm_batches(
        self,
        seed: int,
        *,
        cells: Optional[Sequence[str]] = None,
        difficulties: Sequence[str] = ("easy", "hard"),
    ) -> List[ArmBatch]:
        grid = self.grid(seed, cells=cells)
        batches: List[ArmBatch] = []
        for diff in difficulties:
            for cell_name, payload in grid[diff].items():
                silence = cell_name == "silence"
                gate = SilencingGate(
                    mode=self.silence_mode_for_silence_cell if silence else "none"
                )
                ctrl_in = gate.prepare(difficulty=diff)
                base_kwargs = apply_silence_to_decide_kwargs(ctrl_in)
                base_kwargs["difficulty_true"] = diff  # disclosure / scoring only
                for arm in self.arms:
                    batches.append(
                        ArmBatch(
                            arm=arm,
                            seed=seed,
                            difficulty=diff,
                            cell=cell_name,
                            sequences=payload["sequences"],
                            tasks=payload["tasks"],
                            decide_kwargs=dict(base_kwargs),
                            meta={
                                "init_family": self.init_family,
                                "cell_meta": payload["meta"],
                                "silenced": silence,
                                "parity_key": f"{seed}:{diff}:{cell_name}",
                            },
                        )
                    )
        return batches

    def parity_check(self, seed: int) -> Dict[str, Any]:
        """Verify same seed → identical sequences across arms for each cell."""
        batches = self.iter_arm_batches(seed)
        by_key: Dict[str, List[ArmBatch]] = {}
        for b in batches:
            by_key.setdefault(b.meta["parity_key"], []).append(b)
        ok = True
        details = []
        for key, group in by_key.items():
            ref = group[0].sequences
            matched = all(np.array_equal(ref, g.sequences) for g in group)
            arms = [g.arm for g in group]
            details.append(
                {
                    "parity_key": key,
                    "arms": arms,
                    "sequences_match": matched,
                    "shape": list(ref.shape),
                }
            )
            ok = ok and matched
        # also: clean vs silence sequences identical (silence is controller-side)
        clean_easy = [b for b in batches if b.difficulty == "easy" and b.cell == "clean"]
        sil_easy = [b for b in batches if b.difficulty == "easy" and b.cell == "silence"]
        silence_seq_ok = bool(
            clean_easy
            and sil_easy
            and np.array_equal(clean_easy[0].sequences, sil_easy[0].sequences)
        )
        # noise/shift differ from clean
        noise_easy = [b for b in batches if b.difficulty == "easy" and b.cell == "noise"]
        shift_easy = [b for b in batches if b.difficulty == "easy" and b.cell == "shift"]
        noise_differs = bool(
            noise_easy and not np.array_equal(clean_easy[0].sequences, noise_easy[0].sequences)
        )
        shift_differs = bool(
            shift_easy and not np.array_equal(clean_easy[0].sequences, shift_easy[0].sequences)
        )
        return {
            "pass": bool(ok and silence_seq_ok and noise_differs and shift_differs),
            "init_family": self.init_family,
            "seed": seed,
            "n_batches": len(batches),
            "cells": cell_list(),
            "arms": self.arms,
            "per_key": details,
            "silence_sequences_match_clean": silence_seq_ok,
            "noise_differs_from_clean": noise_differs,
            "shift_differs_from_clean": shift_differs,
            "grid_meta": {k: GRID[k].note for k in GRID},
        }
