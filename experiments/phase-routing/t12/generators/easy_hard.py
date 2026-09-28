"""T12 easy/hard generator adapted from recovered T7 imports.

Easy  = order-1 affine (phase_routing data.py MULTIPLIERS/OFFSETS)
Hard  = order-3 affine (t7_routing freeze data.py A/B/C/D coeffs)

Read-only adaptation: coeffs copied as constants (same values as recovered
sources). No Affinity/C₃/hum touch. Smoke-scale batches only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np

Difficulty = Literal["easy", "hard"]

VOCAB_SIZE = 64
N_TASKS = 8
CONTENT_OFFSET = N_TASKS
CONTENT_SIZE = VOCAB_SIZE - CONTENT_OFFSET
SEQ_LEN = 64  # smoke length (full T7 uses 128)

# Easy — order-1 (from imports/t7_phase_routing_recovered/data.py)
EASY_MULT = (1, 3, 5, 9, 11, 13, 15, 17)
EASY_OFF = (1, 5, 9, 13, 17, 23, 29, 37)

# Hard — order-3 (from imports/t7_routing_freeze_recovered/data.py)
HARD_A = (1, 1, 1, 1, 1, 1, 1, 1)
HARD_B = (1, 1, 1, 1, 2, 2, 2, 2)
HARD_C = (1, 1, 2, 2, 1, 1, 2, 2)
HARD_D = (1, 5, 9, 13, 17, 23, 29, 37)


@dataclass(frozen=True)
class GenConfig:
    seed: int = 20260923
    n_per_task: int = 16
    seq_len: int = SEQ_LEN
    difficulty: Difficulty = "easy"


def next_easy(symbol: int, task: int) -> int:
    return (EASY_MULT[task] * symbol + EASY_OFF[task]) % CONTENT_SIZE


def next_hard(x_t: int, x_tm1: int, x_tm2: int, task: int) -> int:
    return (
        HARD_A[task] * x_t
        + HARD_B[task] * x_tm1
        + HARD_C[task] * x_tm2
        + HARD_D[task]
    ) % CONTENT_SIZE


def _seq_easy(rng: np.random.Generator, task: int, seq_len: int) -> np.ndarray:
    out = np.empty(seq_len, dtype=np.uint8)
    out[0] = task
    symbol = int(rng.integers(CONTENT_SIZE))
    out[1] = CONTENT_OFFSET + symbol
    for pos in range(2, seq_len):
        symbol = next_easy(symbol, task)
        out[pos] = CONTENT_OFFSET + symbol
    return out


def _seq_hard(rng: np.random.Generator, task: int, seq_len: int) -> np.ndarray:
    out = np.empty(seq_len, dtype=np.uint8)
    out[0] = task
    s0 = int(rng.integers(CONTENT_SIZE))
    s1 = int(rng.integers(CONTENT_SIZE))
    s2 = int(rng.integers(CONTENT_SIZE))
    out[1] = CONTENT_OFFSET + s0
    out[2] = CONTENT_OFFSET + s1
    out[3] = CONTENT_OFFSET + s2
    x_tm2, x_tm1, x_t = s0, s1, s2
    for pos in range(4, seq_len):
        symbol = next_hard(x_t, x_tm1, x_tm2, task)
        out[pos] = CONTENT_OFFSET + symbol
        x_tm2, x_tm1, x_t = x_tm1, x_t, symbol
    return out


def build_split(config: GenConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Return (sequences [N,L], task_ids [N]). Stationary only (smoke)."""
    rng = np.random.Generator(np.random.PCG64(config.seed))
    seqs = []
    tasks = []
    for task in range(N_TASKS):
        for _ in range(config.n_per_task):
            if config.difficulty == "easy":
                seqs.append(_seq_easy(rng, task, config.seq_len))
            else:
                seqs.append(_seq_hard(rng, task, config.seq_len))
            tasks.append(task)
    return np.stack(seqs), np.asarray(tasks, dtype=np.int64)


def build_easy_hard(
    seed: int = 20260923,
    n_per_task: int = 16,
    seq_len: int = SEQ_LEN,
) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for diff in ("easy", "hard"):
        seqs, tasks = build_split(
            GenConfig(seed=seed, n_per_task=n_per_task, seq_len=seq_len, difficulty=diff)  # type: ignore[arg-type]
        )
        out[diff] = {"sequences": seqs, "tasks": tasks}
    return out


if __name__ == "__main__":
    data = build_easy_hard(n_per_task=2, seq_len=32)
    for k, v in data.items():
        print(k, v["sequences"].shape, "tasks", v["tasks"][:4].tolist())
