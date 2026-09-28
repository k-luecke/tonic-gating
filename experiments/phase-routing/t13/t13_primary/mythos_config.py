#!/usr/bin/env python3
"""T13 mythos_tiny locks — read-only pins from frozen PREREG (do not amend)."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np

T13 = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
RESULTS = T13 / "results"
CKPT_DIR = RESULTS / "checkpoints"
PREREG_MD = T13 / "PREREG.md"
PREREG_SHA_PATH = T13 / "PREREG.sha256"
AFFINITY_PATH = ROOT / "t11" / "imports" / "hum_measure" / "affinity_inputs.py"

EXPECTED_PREREG_SHA = (
    "3176eb5ea3458047b820ff5511644d20a697a7a9485d5118560ff7b71e1f981d"
)
EXPECTED_AFFINITY_SHA = (
    "7dae877a3975c1ba4ca264f03f572d6749cce39a114ea3adbcffc010c7852c50"
)
EXPECTED_DRAFT_CONTENT_SHA = (
    "1e3f7fe80c4763b827e1bb9cb566adf83551255c5bd4aad11073b91639ea0119"
)

# mythos_tiny DRAFT locks (frozen)
K = 4
W = 64
L_MAX = 4
S_FULL = 64
S_E = 16  # S_full / K
VOCAB = 256
TRAIN_STEPS = 2000
BATCH = 32
LR = 5e-2
LEAK = 0.25  # leaky-integrator mix
SEED_INTERVAL = 0.05
LOCKED_SEEDS: List[int] = [20260928, 20260929, 20260930, 20261001, 20261002]

# G9
GRAD_EXPLODE = 1e4
LOSS_EXPLODE = 1e6

# G13
G13_TOL = 0.1
G13_GRID_N = 36

# G17
G17_N = 1000
G17_P0 = 0.25
G17_THRESH = 0.273

# iso reference (PREREG §3.2)
C_MATCH_REF = 4_210_688


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def read_prereg_sha() -> str:
    return PREREG_SHA_PATH.read_text(encoding="utf-8").split()[0].strip()


def assert_frozen_pins() -> dict:
    """Refuse to train if Affinity / PREREG pins drift. Never mutates those files."""
    got_prereg = sha256_file(PREREG_MD)
    got_affinity = sha256_file(AFFINITY_PATH)
    file_sha = read_prereg_sha()
    ok = (
        got_prereg == EXPECTED_PREREG_SHA
        and file_sha == EXPECTED_PREREG_SHA
        and got_affinity == EXPECTED_AFFINITY_SHA
    )
    return {
        "ok": ok,
        "prereg_sha": got_prereg,
        "prereg_sha_file": file_sha,
        "affinity_sha": got_affinity,
        "expected_prereg": EXPECTED_PREREG_SHA,
        "expected_affinity": EXPECTED_AFFINITY_SHA,
    }


def hashed_phases(prereg_sha: str | None = None, k: int = K) -> np.ndarray:
    """G14: K expert phases fixed and hashed from frozen PREREG digest."""
    salt = prereg_sha or EXPECTED_PREREG_SHA
    phases = np.zeros(k, dtype=np.float64)
    for e in range(k):
        digest = hashlib.sha256(f"{salt}|t13|expert_phase|{e}".encode("utf-8")).digest()
        u = int.from_bytes(digest[:8], "big") / float(1 << 64)
        phases[e] = 2.0 * math.pi * u
    return phases


def phases_sha(phases: np.ndarray) -> str:
    blob = ",".join(f"{p:.12f}" for p in phases.tolist())
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def pairwise_delta_phis(phases: np.ndarray) -> np.ndarray:
    """Unordered pairwise phase gaps on the circle, folded into [0, π]."""
    gaps = []
    k = len(phases)
    for i in range(k):
        for j in range(i + 1, k):
            d = abs(float(phases[i] - phases[j])) % (2.0 * math.pi)
            if d > math.pi:
                d = 2.0 * math.pi - d
            gaps.append(d)
    return np.asarray(gaps, dtype=np.float64)


def g13_delta_phi_set(phases: np.ndarray, n_grid: int = G13_GRID_N) -> np.ndarray:
    """LOCKED: pairwise hashed phases ∪ N=36 uniform grid on [0, 2π)."""
    pairwise = pairwise_delta_phis(phases)
    # also include raw signed pairwise for medium (full circle gaps)
    raw = []
    k = len(phases)
    for i in range(k):
        for j in range(i + 1, k):
            d = (float(phases[j] - phases[i])) % (2.0 * math.pi)
            raw.append(d)
    grid = np.linspace(0.0, 2.0 * math.pi, num=n_grid, endpoint=False)
    combined = np.unique(np.concatenate([np.asarray(raw, dtype=np.float64), grid]))
    return combined


def c_match_moe_slices(l_e: int = L_MAX, s_e: int = S_E, w: int = W, k: int = K) -> int:
    # sum_e L_e * (W*S_e + 4*S_e*W^2) with equal L_e
    per = w * s_e + 4 * s_e * w * w
    return int(k * l_e * per)


def slice_bounds(e: int, s_full: int = S_FULL, s_e: int = S_E) -> Tuple[int, int]:
    start = e * s_e
    return start, start + s_e
