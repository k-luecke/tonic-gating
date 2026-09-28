"""Harder order-3 affine recurrence for T7 (calibrated from single-arm LR smoke)."""
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

# Task-specific order-3: x_{t+1} = (a*x_t + b*x_{t-1} + c*x_{t-2} + d) mod 56
# Coeffs frozen from single-arm LR smoke so clean heldout lands ~0.5–1 nat with
# specialization ≥4/8 at 1000 steps. Large original multipliers + p_flip=0.15 were
# unlearnable on this backbone (heldout stuck ≳2.2); see PREREG §disclosure.
A_COEFFS = (1, 1, 1, 1, 1, 1, 1, 1)
B_COEFFS = (1, 1, 1, 1, 2, 2, 2, 2)
C_COEFFS = (1, 1, 2, 2, 1, 1, 2, 2)
D_COEFFS = (1, 5, 9, 13, 17, 23, 29, 37)
# Flip machinery retained; default p_flip=0 after smoke calibration.
FLIP_OFFSETS = (7, 11, 13, 17, 19, 23, 29, 31)
P_FLIP = 0.0

GENERATED_FILES = ("train.npy", "heldout.npy", "switching.npy", "metadata.json")


@dataclass(frozen=True)
class DataConfig:
    seed: int = 20260921
    train_per_task: int = 256
    heldout_per_task: int = 64
    switching_sequences: int = 512
    seq_len: int = SEQ_LEN
    p_flip: float = P_FLIP


def next_symbol(x_t: int, x_tm1: int, x_tm2: int, task: int) -> int:
    """Deterministic order-3 affine step (zero-based content symbols)."""
    return (
        A_COEFFS[task] * x_t
        + B_COEFFS[task] * x_tm1
        + C_COEFFS[task] * x_tm2
        + D_COEFFS[task]
    ) % CONTENT_SIZE


def maybe_flip(det: int, task: int, rng: np.random.Generator, p_flip: float) -> int:
    """With probability p_flip, replace det with (det + flip_offset[task]) mod 56."""
    if p_flip > 0.0 and float(rng.random()) < p_flip:
        return (det + FLIP_OFFSETS[task]) % CONTENT_SIZE
    return det


def _stationary_sequence(
    rng: np.random.Generator, task: int, seq_len: int, p_flip: float
) -> np.ndarray:
    out = np.empty(seq_len, dtype=np.uint8)
    out[0] = task
    s0 = int(rng.integers(CONTENT_SIZE))
    s1 = int(rng.integers(CONTENT_SIZE))
    s2 = int(rng.integers(CONTENT_SIZE))
    out[1] = CONTENT_OFFSET + s0
    out[2] = CONTENT_OFFSET + s1
    out[3] = CONTENT_OFFSET + s2
    x_tm2, x_tm1, x_t = s0, s1, s2
    for position in range(4, seq_len):
        det = next_symbol(x_t, x_tm1, x_tm2, task)
        symbol = maybe_flip(det, task, rng, p_flip)
        out[position] = CONTENT_OFFSET + symbol
        x_tm2, x_tm1, x_t = x_tm1, x_t, symbol
    return out


def _switching_sequence(
    rng: np.random.Generator, seq_len: int, p_flip: float
) -> np.ndarray:
    first = int(rng.integers(N_TASKS))
    second = int(rng.integers(N_TASKS - 1))
    if second >= first:
        second += 1
    switch_position = int(rng.integers(24, seq_len - 27))
    out = np.empty(seq_len, dtype=np.uint8)
    out[0] = first
    s0 = int(rng.integers(CONTENT_SIZE))
    s1 = int(rng.integers(CONTENT_SIZE))
    s2 = int(rng.integers(CONTENT_SIZE))
    out[1] = CONTENT_OFFSET + s0
    out[2] = CONTENT_OFFSET + s1
    out[3] = CONTENT_OFFSET + s2
    x_tm2, x_tm1, x_t = s0, s1, s2
    for position in range(4, switch_position):
        det = next_symbol(x_t, x_tm1, x_tm2, first)
        symbol = maybe_flip(det, first, rng, p_flip)
        out[position] = CONTENT_OFFSET + symbol
        x_tm2, x_tm1, x_t = x_tm1, x_t, symbol
    out[switch_position] = second
    s0 = int(rng.integers(CONTENT_SIZE))
    s1 = int(rng.integers(CONTENT_SIZE))
    s2 = int(rng.integers(CONTENT_SIZE))
    out[switch_position + 1] = CONTENT_OFFSET + s0
    out[switch_position + 2] = CONTENT_OFFSET + s1
    out[switch_position + 3] = CONTENT_OFFSET + s2
    x_tm2, x_tm1, x_t = s0, s1, s2
    for position in range(switch_position + 4, seq_len):
        det = next_symbol(x_t, x_tm1, x_tm2, second)
        symbol = maybe_flip(det, second, rng, p_flip)
        out[position] = CONTENT_OFFSET + symbol
        x_tm2, x_tm1, x_t = x_tm1, x_t, symbol
    return out


def build_arrays(config: DataConfig) -> dict[str, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(config.seed))
    train = np.stack([
        _stationary_sequence(rng, task, config.seq_len, config.p_flip)
        for task in range(N_TASKS)
        for _ in range(config.train_per_task)
    ])
    heldout = np.stack([
        _stationary_sequence(rng, task, config.seq_len, config.p_flip)
        for task in range(N_TASKS)
        for _ in range(config.heldout_per_task)
    ])
    switching = np.stack([
        _switching_sequence(rng, config.seq_len, config.p_flip)
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
        "rule": (
            "x_{t+1} = (a[t]*x_t + b[t]*x_{t-1} + c[t]*x_{t-2} + d[t]) mod 56; "
            f"with probability p_flip={config.p_flip} replace det with "
            "(det + flip_offset[t]) mod 56"
        ),
        "a_coeffs": list(A_COEFFS),
        "b_coeffs": list(B_COEFFS),
        "c_coeffs": list(C_COEFFS),
        "d_coeffs": list(D_COEFFS),
        "flip_offsets": list(FLIP_OFFSETS),
        "p_flip": config.p_flip,
        "order": 3,
        "calibration_note": (
            "p_flip default 0.0 after single-arm smoke: p_flip=0.15 left LR heldout "
            "stuck ≳2.2 nats on this backbone; order-3 + C-tier coeffs provide hardness "
            "vs primary (~0.0013). See PREREG disclosure."
        ),
        "split_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "rng": "numpy.random.PCG64",
        "experiment": "T7",
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
    parser.add_argument("--p-flip", type=float, default=P_FLIP)
    args = parser.parse_args()
    config = DataConfig(
        seed=args.seed,
        train_per_task=args.train_per_task,
        heldout_per_task=args.heldout_per_task,
        switching_sequences=args.switching_sequences,
        p_flip=args.p_flip,
    )
    manifest_hash = write_dataset(args.output_dir, config)
    print(f"seed={config.seed}")
    print(f"p_flip={config.p_flip}")
    print(f"manifest_sha256={manifest_hash}")


if __name__ == "__main__":
    main()
