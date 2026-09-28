"""Deterministic synthetic multi-task language-model data."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

VOCAB_SIZE = 64
N_TASKS = 8
MARKER_TOKENS = tuple(range(N_TASKS))
CONTENT_OFFSET = N_TASKS
CONTENT_SIZE = VOCAB_SIZE - CONTENT_OFFSET
SEQ_LEN = 128
# All multipliers are units modulo 56, making each task rule a permutation.
MULTIPLIERS = (1, 3, 5, 9, 11, 13, 15, 17)
OFFSETS = (1, 5, 9, 13, 17, 23, 29, 37)
GENERATED_FILES = ("train.npy", "heldout.npy", "switching.npy", "metadata.json")


@dataclass(frozen=True)
class DataConfig:
    seed: int = 20260921
    train_per_task: int = 256
    heldout_per_task: int = 64
    switching_sequences: int = 512
    seq_len: int = SEQ_LEN


def next_symbol(symbol: int, task: int) -> int:
    """Apply task's affine recurrence to a zero-based content symbol."""
    return (MULTIPLIERS[task] * symbol + OFFSETS[task]) % CONTENT_SIZE


def _stationary_sequence(rng: np.random.Generator, task: int, seq_len: int) -> np.ndarray:
    out = np.empty(seq_len, dtype=np.uint8)
    out[0] = task
    symbol = int(rng.integers(CONTENT_SIZE))
    out[1] = CONTENT_OFFSET + symbol
    for position in range(2, seq_len):
        symbol = next_symbol(symbol, task)
        out[position] = CONTENT_OFFSET + symbol
    return out


def _switching_sequence(rng: np.random.Generator, seq_len: int) -> np.ndarray:
    """Create one sequence with a new task marker at a random interior position."""
    first = int(rng.integers(N_TASKS))
    second = int(rng.integers(N_TASKS - 1))
    if second >= first:
        second += 1
    # Leave enough predictable tokens on both sides of the change.
    switch_position = int(rng.integers(24, seq_len - 23))
    out = np.empty(seq_len, dtype=np.uint8)
    out[0] = first
    symbol = int(rng.integers(CONTENT_SIZE))
    out[1] = CONTENT_OFFSET + symbol
    for position in range(2, switch_position):
        symbol = next_symbol(symbol, first)
        out[position] = CONTENT_OFFSET + symbol
    out[switch_position] = second
    symbol = int(rng.integers(CONTENT_SIZE))
    out[switch_position + 1] = CONTENT_OFFSET + symbol
    for position in range(switch_position + 2, seq_len):
        symbol = next_symbol(symbol, second)
        out[position] = CONTENT_OFFSET + symbol
    return out


def build_arrays(config: DataConfig) -> dict[str, np.ndarray]:
    """Build arrays using one explicitly seeded PCG64 stream."""
    rng = np.random.Generator(np.random.PCG64(config.seed))
    train = np.stack([
        _stationary_sequence(rng, task, config.seq_len)
        for task in range(N_TASKS)
        for _ in range(config.train_per_task)
    ])
    heldout = np.stack([
        _stationary_sequence(rng, task, config.seq_len)
        for task in range(N_TASKS)
        for _ in range(config.heldout_per_task)
    ])
    switching = np.stack([
        _switching_sequence(rng, config.seq_len)
        for _ in range(config.switching_sequences)
    ])
    return {"train": train, "heldout": heldout, "switching": switching}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset(output_dir: Path, config: DataConfig) -> str:
    """Write deterministic NPY artifacts and return the manifest SHA-256."""
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = build_arrays(config)
    for name, array in arrays.items():
        with (output_dir / f"{name}.npy").open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
    metadata = {
        "config": asdict(config),
        "vocab_size": VOCAB_SIZE,
        "marker_tokens": list(MARKER_TOKENS),
        "content_tokens": [CONTENT_OFFSET, VOCAB_SIZE - 1],
        "rule": "x_next = (multiplier[task] * x + offset[task]) mod 56",
        "multipliers": list(MULTIPLIERS),
        "offsets": list(OFFSETS),
        "split_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "rng": "numpy.random.PCG64",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    included = [*GENERATED_FILES]
    changelog = output_dir / "CHANGELOG.txt"
    if changelog.exists():
        included.append("CHANGELOG.txt")
    lines = [f"{_sha256(output_dir / name)}  data/{name}" for name in sorted(included)]
    manifest = output_dir / "MANIFEST.sha256"
    manifest.write_text("\n".join(lines) + "\n", encoding="ascii")
    return _sha256(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-per-task", type=int, default=256)
    parser.add_argument("--heldout-per-task", type=int, default=64)
    parser.add_argument("--switching-sequences", type=int, default=512)
    args = parser.parse_args()
    config = DataConfig(
        seed=args.seed,
        train_per_task=args.train_per_task,
        heldout_per_task=args.heldout_per_task,
        switching_sequences=args.switching_sequences,
    )
    manifest_hash = write_dataset(args.output_dir, config)
    print(f"seed={config.seed}")
    print(f"manifest_sha256={manifest_hash}")


if __name__ == "__main__":
    main()
