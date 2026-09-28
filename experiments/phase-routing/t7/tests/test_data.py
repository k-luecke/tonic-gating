from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from t7_routing.data import (
    A_COEFFS,
    B_COEFFS,
    C_COEFFS,
    CONTENT_OFFSET,
    CONTENT_SIZE,
    D_COEFFS,
    FLIP_OFFSETS,
    P_FLIP,
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


def test_order3_rule_and_optional_flip_entropy() -> None:
    """Default p_flip=0: exact order-3. Explicit p_flip>0: flip rate near p."""
    clean = build_arrays(
        DataConfig(seed=1, train_per_task=8, heldout_per_task=1, switching_sequences=1, p_flip=0.0)
    )
    for row in clean["train"]:
        task = int(row[0])
        assert 0 <= task < N_TASKS
        xs = [int(row[i]) - CONTENT_OFFSET for i in range(1, 4)]
        for position in range(4, len(row)):
            expected = next_symbol(xs[2], xs[1], xs[0], task)
            assert int(row[position]) - CONTENT_OFFSET == expected
            xs = [xs[1], xs[2], expected]

    noisy = build_arrays(
        DataConfig(
            seed=20260921,
            train_per_task=64,
            heldout_per_task=8,
            switching_sequences=1,
            p_flip=0.15,
        )
    )
    n_check = n_flip = 0
    for row in noisy["train"]:
        task = int(row[0])
        xs = [int(row[i]) - CONTENT_OFFSET for i in range(1, 4)]
        for position in range(4, len(row)):
            det = next_symbol(xs[2], xs[1], xs[0], task)
            observed = int(row[position]) - CONTENT_OFFSET
            n_check += 1
            if observed != det:
                n_flip += 1
                assert observed == (det + FLIP_OFFSETS[task]) % CONTENT_SIZE
            xs = [xs[1], xs[2], observed]
    rate = n_flip / n_check
    assert 0.10 < rate < 0.20, f"flip rate {rate} outside expected band around 0.15"


def test_frozen_coeffs_and_default_p_flip() -> None:
    assert A_COEFFS == tuple([1] * 8)
    assert B_COEFFS == (1, 1, 1, 1, 2, 2, 2, 2)
    assert C_COEFFS == (1, 1, 2, 2, 1, 1, 2, 2)
    assert D_COEFFS == (1, 5, 9, 13, 17, 23, 29, 37)
    assert P_FLIP == 0.0
    assert len(FLIP_OFFSETS) == 8


def test_switching_split_contains_two_distinct_markers() -> None:
    switching = build_arrays(
        DataConfig(train_per_task=1, heldout_per_task=1, switching_sequences=16, p_flip=0.0)
    )["switching"]
    for row in switching:
        marker_positions = np.flatnonzero(row < CONTENT_OFFSET)
        assert marker_positions.size == 2
        assert row[marker_positions[0]] != row[marker_positions[1]]
        assert 24 <= marker_positions[1] <= len(row) - 28
