from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from phase_routing.data import (
    CONTENT_OFFSET,
    DataConfig,
    N_TASKS,
    build_arrays,
    next_symbol,
    write_dataset,
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generation_is_bit_deterministic(tmp_path: Path) -> None:
    config = DataConfig(seed=20260921, train_per_task=4, heldout_per_task=2, switching_sequences=5)
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert write_dataset(first, config) == write_dataset(second, config)
    for name in ("train.npy", "heldout.npy", "switching.npy", "metadata.json", "MANIFEST.sha256"):
        assert file_hash(first / name) == file_hash(second / name)


def test_stationary_sequences_follow_task_rule() -> None:
    arrays = build_arrays(DataConfig(train_per_task=2, heldout_per_task=1, switching_sequences=1))
    for row in arrays["train"]:
        task = int(row[0])
        assert 0 <= task < N_TASKS
        for current, following in zip(row[1:-1], row[2:]):
            expected = CONTENT_OFFSET + next_symbol(int(current) - CONTENT_OFFSET, task)
            assert int(following) == expected


def test_switching_split_contains_two_distinct_markers() -> None:
    switching = build_arrays(DataConfig(train_per_task=1, heldout_per_task=1, switching_sequences=16))["switching"]
    for row in switching:
        marker_positions = np.flatnonzero(row < CONTENT_OFFSET)
        assert marker_positions.size == 2
        assert row[marker_positions[0]] != row[marker_positions[1]]
        assert 24 <= marker_positions[1] <= len(row) - 24
