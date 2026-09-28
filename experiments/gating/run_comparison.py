#!/usr/bin/env python3
"""
T5 — Gate vs router comparison harness (PREREG §3–§6).

Default: smoke-fast subset (wiring check; non-preregistered).
--full: 200 symbols × 10 seeds × full 4×4×4 grid + wobble stretch.
--switch: dedicated switching-latency protocol (also included in --full).

Does not retrain the router; loads frozen weights/readouts from results/.
Gate path trains oscillatory-sheet readouts once if missing
(results/gate_readouts.npz), then freezes them.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from content import N_STREAMS, multiplex_streams, stream_phase
from lockin import (
    CYCLES_PER_SYMBOL,
    DT,
    F0,
    FS,
    PILOT_AMP,
    SAMPLES_PER_SYMBOL,
    per_cycle_phase_jitter,
)
from router import (
    DEFAULT_READOUT_PATH,
    DEFAULT_WEIGHTS_PATH,
    PatchReadouts,
    load_frozen_readouts,
    load_frozen_router,
    router_features,
    sha256_file,
)
from sheet import Control, N_PATCHES, N_UNITS, PATCH_UNITS, Sheet
from stencil import StencilController

# ---------------------------------------------------------------------------
# Paths / grids
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "comparison"
PRIMARY_RESULTS = ROOT / "results" / "comparison_primary"
GATE_READOUT_PATH = ROOT / "results" / "gate_readouts.npz"
ROUTER_PILOT_READOUT_PATH = ROOT / "results" / "router_pilot_readouts.npz"
PREREG_PATH = ROOT / "PREREG.md"
PREREG_HASH_PATH = ROOT / "results" / "PREREG.sha256"

SIGMA_F_GRID = (0.0, 0.02, 0.05, 0.10)
SIGMA_J_GRID = (0.0, 0.02, 0.05, 0.10)
SIGMA_C_GRID = (0.0, 0.01, 0.03, 0.05)

WOBBLE_HZ = 5.0
WOBBLE_AMP = 0.3  # rad

# T1 slots/cycle @ SNR+10 dB (for C_ctl gate)
SLOTS_BY_SIGMA_J = {
    0.0: 15.65,
    0.02: 11.82,
    0.05: 8.35,
    0.10: 4.30,
}

SAMPLES_PER_CYCLE = int(round(FS / F0))  # 100

# Smoke defaults (non-preregistered)
SMOKE_N_SYMBOLS = 40
SMOKE_N_SEEDS = 3
SMOKE_SEED0 = 0
# Subset: clean, each-axis high, mid corner, plus wobble stretch
SMOKE_CELLS: list[tuple[float, float, float, bool]] = [
    (0.0, 0.0, 0.0, False),
    (0.10, 0.0, 0.0, False),
    (0.0, 0.10, 0.0, False),
    (0.0, 0.0, 0.05, False),
    (0.05, 0.05, 0.03, False),
    (0.0, 0.0, 0.0, True),  # wobble stretch on clean mismatch/jitter/drift
]

# Primary registered grid (denser than smoke; prereg trial counts).
# Used when --full cannot finish under ~15–20 min. Includes H1 cells
# (clean + σ_c=0.05) and axis/corner coverage + wobble.
PRIMARY_CELLS: list[tuple[float, float, float, bool]] = [
    (0.0, 0.0, 0.0, False),
    (0.0, 0.0, 0.01, False),
    (0.0, 0.0, 0.03, False),
    (0.0, 0.0, 0.05, False),  # H1 drift clause
    (0.02, 0.0, 0.0, False),
    (0.05, 0.0, 0.0, False),
    (0.10, 0.0, 0.0, False),
    (0.0, 0.02, 0.0, False),
    (0.0, 0.05, 0.0, False),
    (0.0, 0.10, 0.0, False),
    (0.05, 0.05, 0.03, False),
    (0.10, 0.10, 0.05, False),
    (0.05, 0.0, 0.05, False),
    (0.0, 0.05, 0.05, False),
    (0.0, 0.0, 0.0, True),  # wobble stretch
]

FULL_N_SYMBOLS = 200
FULL_N_SEEDS = 10
FULL_SEED0 = 0
N_SWITCHES = 50


# ---------------------------------------------------------------------------
# Condition helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Condition:
    sigma_f: float
    sigma_j: float
    sigma_c: float
    wobble: bool = False

    def key(self) -> str:
        tag = "wobble" if self.wobble else "grid"
        return (
            f"{tag}_sf{self.sigma_f:.2f}_sj{self.sigma_j:.2f}_sc{self.sigma_c:.2f}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "sigma_f": self.sigma_f,
            "sigma_j": self.sigma_j,
            "sigma_c": self.sigma_c,
            "wobble": self.wobble,
            "key": self.key(),
        }


def full_grid_conditions() -> list[Condition]:
    cells = [
        Condition(sf, sj, sc, False)
        for sf in SIGMA_F_GRID
        for sj in SIGMA_J_GRID
        for sc in SIGMA_C_GRID
    ]
    cells.append(Condition(0.0, 0.0, 0.0, True))
    return cells


def smoke_conditions() -> list[Condition]:
    return [Condition(*c) for c in SMOKE_CELLS]


def primary_conditions() -> list[Condition]:
    return [Condition(*c) for c in PRIMARY_CELLS]


# ---------------------------------------------------------------------------
# Disturbance / content generation (shared by both systems)
# ---------------------------------------------------------------------------
def sigma_c_rad_per_sample(sigma_c_rad_per_cycle: float) -> float:
    """
    Convert PREREG σ_c (radians per carrier cycle) to RW step std per sample.

    samples_per_cycle = FS/F0 (= 100 at 4 kHz / 40 Hz).
    step_std = σ_c_rad_per_cycle / samples_per_cycle.

    Legacy smoke used σ_c as rad/sample directly (100× more aggressive at
    the same numeric grid). See PREREG §3.
    """
    return float(sigma_c_rad_per_cycle) / float(SAMPLES_PER_CYCLE)


def shared_phase_trajectory(
    n: int,
    *,
    sigma_c: float,
    wobble: bool,
    t0: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Cumulative shared phase (radians), length n.

    σ_c: PREREG units = **radians per carrier cycle**. Random-walk increment
    per sample ~ N(0, σ_c / samples_per_cycle). Optional 5 Hz sinusoidal
    wobble amplitude 0.3 rad.
    """
    phase = np.zeros(n, dtype=np.float64)
    if sigma_c > 0.0:
        step = sigma_c_rad_per_sample(sigma_c)
        increments = rng.normal(0.0, step, size=n)
        phase = np.cumsum(increments)
    if wobble:
        t = t0 + np.arange(n, dtype=np.float64) * DT
        phase = phase + WOBBLE_AMP * np.sin(2.0 * np.pi * WOBBLE_HZ * t)
    return phase


def make_symbol(
    bits: np.ndarray,
    *,
    sigma_j: float,
    sigma_c: float,
    wobble: bool,
    t0: float,
    rng: np.random.Generator,
    n_samples: int = SAMPLES_PER_SYMBOL,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build multiplexed symbol window + shared phase trajectory.

    Jitter: per-cycle phase offset (same draw applied to content).
    Shared drift/wobble applied identically to all streams.
    """
    shared = shared_phase_trajectory(
        n_samples, sigma_c=sigma_c, wobble=wobble, t0=t0, rng=rng
    )
    jitter = per_cycle_phase_jitter(n_samples, sigma_j, rng)
    phase = shared + jitter
    active = {d: float(bits[d]) for d in range(N_STREAMS)}
    window = multiplex_streams(n_samples, active, shared_phase=phase)
    return window, phase


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------
def selectivity_from_A(A: np.ndarray, k_star: int) -> float:
    tot = float(A.sum())
    if tot < 1e-12:
        return 0.0
    return float(A[int(k_star)] / tot)


def leakage_from_A(A: np.ndarray, k_star: int, A0: float = 0.0) -> float:
    k_star = int(k_star)
    Ak = float(A[k_star])
    if Ak < 1e-12:
        return 0.0
    others = [max(0.0, float(A[k]) - A0) / Ak for k in range(len(A)) if k != k_star]
    return float(np.mean(others)) if others else 0.0


def c_ctl_gate(sigma_j: float) -> float:
    """log2(K) + K * log2(slots) with slots from T1 prior at this σ_j."""
    # nearest tabulated σ_j
    sj = min(SLOTS_BY_SIGMA_J.keys(), key=lambda x: abs(x - sigma_j))
    slots = SLOTS_BY_SIGMA_J[sj]
    return float(np.log2(N_PATCHES) + N_PATCHES * np.log2(max(slots, 1.0)))


def c_ctl_router(macs: int) -> float:
    return float(macs)


def bootstrap_ci(
    values: np.ndarray, *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap 95% CI. values = one number per seed."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(v.mean())
    if v.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return mean, lo, hi


# ---------------------------------------------------------------------------
# Gate readouts (oscillatory sheet activity → bit)
# ---------------------------------------------------------------------------
def pilot_phase_resample(
    drive: np.ndarray,
    pilot_phase: np.ndarray,
    n_bins: int | None = None,
) -> np.ndarray:
    """
    Resample gated drive onto a uniform grid in *open-pilot phase*.

    This is the primary gate bit-feature: activity is referenced to the pilot
    phase, not absolute sample index. Under shared drift the pilot walks with
    the content; absolute-time FIR (legacy) collapses even when the PLL holds.
    Empty phase bins are filled by periodic linear interpolation.
    """
    drive = np.asarray(drive, dtype=np.float64).reshape(-1)
    pilot_phase = np.asarray(pilot_phase, dtype=np.float64).reshape(-1)
    if drive.size != pilot_phase.size:
        raise ValueError("drive and pilot_phase length mismatch")
    if n_bins is None:
        n_bins = int(drive.size)
    n_bins = int(n_bins)
    acc = np.zeros(n_bins, dtype=np.float64)
    cnt = np.zeros(n_bins, dtype=np.float64)
    bins = (np.mod(pilot_phase, 2.0 * np.pi) / (2.0 * np.pi) * n_bins).astype(np.int64) % n_bins
    for b, v in zip(bins, drive):
        acc[b] += float(v)
        cnt[b] += 1.0
    out = np.zeros(n_bins, dtype=np.float64)
    mask = cnt > 0.0
    out[mask] = acc[mask] / cnt[mask]
    if mask.any() and not mask.all():
        x = np.arange(n_bins, dtype=np.float64)
        xp = x[mask]
        fp = out[mask]
        xp2 = np.concatenate([xp - n_bins, xp, xp + n_bins])
        fp2 = np.concatenate([fp, fp, fp])
        out[~mask] = np.interp(x[~mask], xp2, fp2)
    return out


GATE_READOUT_MODE_PILOT_PHASE = "pilot_phase"
GATE_READOUT_MODE_TIME_FIR = "time_fir"  # exploratory / legacy only


@dataclass
class GateReadouts:
    """
    Per-patch ridge bit heads on oscillatory gated drive.

    Primary mode ``pilot_phase``: features = gated drive (mean g·u on the open
    patch) resampled onto a uniform open-pilot-phase grid. References the
    pilot, not absolute time.

    Legacy mode ``time_fir``: ridge on the absolute-time drive series — must
    not be the primary frozen path.
    """

    weights: np.ndarray  # (K, n_bins+1)
    n_samples: int = SAMPLES_PER_SYMBOL
    n_bins: int = SAMPLES_PER_SYMBOL
    A0: float = 0.0
    mode: str = GATE_READOUT_MODE_PILOT_PHASE

    def features(self, drive_ts: np.ndarray, pilot_phase_ts: np.ndarray | None = None) -> np.ndarray:
        drive = np.asarray(drive_ts, dtype=np.float64).reshape(-1)
        if self.mode == GATE_READOUT_MODE_PILOT_PHASE:
            if pilot_phase_ts is None:
                raise ValueError("pilot_phase mode requires pilot_phase_ts")
            return pilot_phase_resample(drive, pilot_phase_ts, n_bins=self.n_bins)
        if self.mode == GATE_READOUT_MODE_TIME_FIR:
            if drive.size != self.n_samples:
                raise ValueError(f"activity length {drive.size} != {self.n_samples}")
            return drive
        raise ValueError(f"unknown gate readout mode {self.mode!r}")

    def predict_bit(
        self,
        activity_ts: np.ndarray,
        patch: int,
        pilot_phase_ts: np.ndarray | None = None,
    ) -> int:
        feat = self.features(activity_ts, pilot_phase_ts)
        xb = np.append(feat, 1.0)
        score = float(xb @ self.weights[int(patch)])
        return 1 if score > 0.5 else 0

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            weights=self.weights,
            n_samples=np.array([self.n_samples]),
            n_bins=np.array([self.n_bins]),
            A0=np.array([self.A0]),
            mode=np.array(self.mode),
        )
        return sha256_file(path)

    @classmethod
    def load(cls, path: str | Path) -> "GateReadouts":
        data = np.load(path, allow_pickle=True)
        mode = (
            str(data["mode"].item())
            if "mode" in data
            else GATE_READOUT_MODE_TIME_FIR  # legacy npz
        )
        n_bins = int(data["n_bins"][0]) if "n_bins" in data else int(data["n_samples"][0])
        return cls(
            weights=np.array(data["weights"], dtype=np.float64, copy=True),
            n_samples=int(data["n_samples"][0]),
            n_bins=n_bins,
            A0=float(data["A0"][0]) if "A0" in data else 0.0,
            mode=mode,
        )


def fit_gate_readouts(
    n_symbols: int = 2000,
    *,
    seed: int = 20260921,
    lam: float = 1e-3,
    sigma_f: float = 0.0,
    mode: str = GATE_READOUT_MODE_PILOT_PHASE,
) -> GateReadouts:
    """
    Train per-patch linear readouts on clean oscillatory-sheet gated drive.

    Primary (pilot_phase): gated drive resampled onto the open-pilot phase
    grid, then ridge → bit. Label: bit of stream k when patch k is open.
    """
    if mode not in (GATE_READOUT_MODE_PILOT_PHASE, GATE_READOUT_MODE_TIME_FIR):
        raise ValueError(mode)
    rng = np.random.default_rng(seed)
    S = SAMPLES_PER_SYMBOL
    Xs: list[list[np.ndarray]] = [[] for _ in range(N_PATCHES)]
    Ys: list[list[float]] = [[] for _ in range(N_PATCHES)]

    for _ in range(n_symbols):
        d_star = int(rng.integers(0, N_STREAMS))
        bits = rng.integers(0, 2, size=N_STREAMS).astype(np.float64)
        window, phase = make_symbol(
            bits, sigma_j=0.0, sigma_c=0.0, wobble=False, t0=0.0, rng=rng
        )
        sheet = Sheet(
            sigma_f=sigma_f, seed=int(rng.integers(0, 2**31 - 1)), gain_mode="oscillatory"
        )
        theta = stream_phase(d_star)
        sheet.phi[:] = theta + sheet.rng.normal(0.0, 0.05, size=N_UNITS)
        stencil = StencilController(
            open_patch=d_star,
            sigma_c=0.0,
            seed=int(rng.integers(0, 2**31 - 1)),
            shared_delta=0.0,
        )
        stencil.control_for_target(d_star, open_patch=d_star)
        drive = np.empty(S, dtype=np.float64)
        pilot_phase = np.empty(S, dtype=np.float64)
        idx = PATCH_UNITS[d_star]
        for i in range(S):
            dpsi = 2.0 * np.pi * stencil.carrier_hz * stencil.dt
            stencil.psi = stencil.psi + dpsi
            stencil.t += stencil.dt
            psi = np.asarray(stencil.psi, dtype=np.float64) + phase[i]
            pilot = PILOT_AMP * np.cos(psi)
            u = float(window[i])
            sheet.step(u, Control(inhibition=stencil.I.copy(), pilot_signal=pilot))
            drive[i] = float(np.mean(sheet.gain[idx] * u))
            pilot_phase[i] = float(np.asarray(psi).reshape(-1)[d_star])
        if mode == GATE_READOUT_MODE_PILOT_PHASE:
            feat = pilot_phase_resample(drive, pilot_phase, n_bins=S)
        else:
            feat = drive
        Xs[d_star].append(feat)
        Ys[d_star].append(float(bits[d_star]))

    feat_dim = S
    W = np.zeros((N_PATCHES, feat_dim + 1), dtype=np.float64)
    eye = np.eye(feat_dim + 1)
    eye[-1, -1] = 0.0
    for k in range(N_PATCHES):
        if not Xs[k]:
            continue
        X = np.stack(Xs[k], axis=0)
        y = np.asarray(Ys[k], dtype=np.float64)
        Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        W[k] = np.linalg.solve(Xb.T @ Xb + lam * eye, Xb.T @ y)
    return GateReadouts(
        weights=W, n_samples=S, n_bins=S, A0=0.0, mode=mode
    )



def load_or_train_gate_readouts(
    path: Path = GATE_READOUT_PATH, *, force: bool = False
) -> GateReadouts:
    if path.exists() and not force:
        gr = GateReadouts.load(path)
        if gr.mode == GATE_READOUT_MODE_PILOT_PHASE:
            return gr
        print(f"Gate readouts at {path} are mode={gr.mode!r}; retraining pilot_phase …")
        force = True
    print(f"Training gate readouts → {path} …")
    gr = fit_gate_readouts(
        n_symbols=800, seed=20260921, mode=GATE_READOUT_MODE_PILOT_PHASE
    )
    digest = gr.save(path)
    print(f"  saved gate_readouts mode={gr.mode} sha256={digest[:16]}…")
    return gr


# ---------------------------------------------------------------------------
# Single-seed evaluation for one condition / system
# ---------------------------------------------------------------------------
def _apply_receiver_jitter(sheet: Sheet, sigma_j: float, rng: np.random.Generator) -> None:
    """Add one-shot per-unit phase jitter of scale σ_j (cycles → rad) at symbol start."""
    if sigma_j <= 0.0:
        return
    sheet.phi += rng.normal(0.0, sigma_j * 2.0 * np.pi, size=N_UNITS)
    sheet.phi = (sheet.phi + np.pi) % (2.0 * np.pi) - np.pi



# ---------------------------------------------------------------------------
# Secondary arm: router + pilot-referenced demod (same alignment ref as gate)
# ---------------------------------------------------------------------------
@dataclass
class RouterPilotReadouts:
    """
    Per-patch ridge on pilot-phase-resampled multiplex window.

    Secondary preregistered arm: router selection (no sheet pilot / gain=1)
    but bit demod uses the *same* open-pilot-phase grid as the gate, isolating
    spatial gating from the alignment reference.
    """

    weights: np.ndarray  # (K, n_bins+1)
    n_bins: int = SAMPLES_PER_SYMBOL
    n_samples: int = SAMPLES_PER_SYMBOL

    def predict_bit(
        self,
        window: np.ndarray,
        patch: int,
        pilot_phase_ts: np.ndarray,
    ) -> int:
        feat = pilot_phase_resample(window, pilot_phase_ts, n_bins=self.n_bins)
        xb = np.append(feat, 1.0)
        score = float(xb @ self.weights[int(patch)])
        return 1 if score > 0.5 else 0

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            weights=self.weights,
            n_bins=np.array([self.n_bins]),
            n_samples=np.array([self.n_samples]),
            mode=np.array("router_pilot_phase"),
        )
        return sha256_file(path)

    @classmethod
    def load(cls, path: str | Path) -> "RouterPilotReadouts":
        data = np.load(path, allow_pickle=True)
        return cls(
            weights=np.array(data["weights"], dtype=np.float64, copy=True),
            n_bins=int(data["n_bins"][0]),
            n_samples=int(data["n_samples"][0]),
        )


def fit_router_pilot_readouts(
    n_symbols: int = 2000,
    *,
    seed: int = 20260921,
    lam: float = 1e-3,
) -> RouterPilotReadouts:
    """Train pilot-phase-resampled window → bit heads (clean data)."""
    rng = np.random.default_rng(seed)
    S = SAMPLES_PER_SYMBOL
    Xs: list[list[np.ndarray]] = [[] for _ in range(N_PATCHES)]
    Ys: list[list[float]] = [[] for _ in range(N_PATCHES)]
    for _ in range(n_symbols):
        bits = rng.integers(0, 2, size=N_STREAMS).astype(np.float64)
        # clean: shared phase = 0; pilot phase for stream k = 2π f0 t + θ_k
        window, _shared = make_symbol(
            bits, sigma_j=0.0, sigma_c=0.0, wobble=False, t0=0.0, rng=rng
        )
        t = np.arange(S, dtype=np.float64) * DT
        for k in range(N_PATCHES):
            pilot_phase = 2.0 * np.pi * F0 * t + stream_phase(k)
            feat = pilot_phase_resample(window, pilot_phase, n_bins=S)
            Xs[k].append(feat)
            Ys[k].append(float(bits[k]))
    W = np.zeros((N_PATCHES, S + 1), dtype=np.float64)
    eye = np.eye(S + 1)
    eye[-1, -1] = 0.0
    for k in range(N_PATCHES):
        X = np.stack(Xs[k], axis=0)
        y = np.asarray(Ys[k], dtype=np.float64)
        Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        W[k] = np.linalg.solve(Xb.T @ Xb + lam * eye, Xb.T @ y)
    return RouterPilotReadouts(weights=W, n_bins=S, n_samples=S)


def load_or_train_router_pilot_readouts(
    path: Path = None, *, force: bool = False
) -> RouterPilotReadouts:
    path = Path(path) if path is not None else ROUTER_PILOT_READOUT_PATH
    if path.exists() and not force:
        return RouterPilotReadouts.load(path)
    print(f"Training router-pilot readouts → {path} …")
    rr = fit_router_pilot_readouts(n_symbols=800, seed=20260921)
    digest = rr.save(path)
    print(f"  saved router_pilot_readouts sha256={digest[:16]}…")
    return rr



def run_gate_seed(
    cond: Condition,
    seed: int,
    n_symbols: int,
    gate_readouts: GateReadouts,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sheet_rng_seed = seed + 17
    S = SAMPLES_PER_SYMBOL
    acc_hits = 0
    S_list, L_list = [], []
    c_act_acc = 0.0
    c_energy_acc = 0.0
    n_steps_total = 0

    for sym_i in range(n_symbols):
        d_star = int(rng.integers(0, N_STREAMS))
        bits = rng.integers(0, 2, size=N_STREAMS).astype(np.float64)
        t0 = sym_i * S * DT
        window, shared_phase = make_symbol(
            bits,
            sigma_j=cond.sigma_j,
            sigma_c=cond.sigma_c,
            wobble=cond.wobble,
            t0=t0,
            rng=rng,
        )
        sheet = Sheet(
            sigma_f=cond.sigma_f, seed=sheet_rng_seed + sym_i, gain_mode="oscillatory"
        )
        theta = stream_phase(d_star)
        sheet.phi[:] = theta + sheet.rng.normal(0.0, 0.1, size=N_UNITS)
        _apply_receiver_jitter(sheet, cond.sigma_j, rng)

        stencil = StencilController(
            open_patch=d_star,
            sigma_c=0.0,
            shared_delta=0.0,
            seed=seed + 1000 + sym_i,
        )
        stencil.control_for_target(d_star, open_patch=d_star)

        drive_ts = np.empty(S, dtype=np.float64)
        pilot_phase_ts = np.empty(S, dtype=np.float64)
        gain_active_sum = 0.0
        energy_sum = 0.0
        idx = PATCH_UNITS[d_star]
        for i in range(S):
            # Advance stencil carrier; inject shared phase into pilot
            dpsi = 2.0 * np.pi * stencil.carrier_hz * stencil.dt
            stencil.psi = stencil.psi + dpsi
            stencil.t += stencil.dt
            psi = np.asarray(stencil.psi, dtype=np.float64) + shared_phase[i]
            pilot = PILOT_AMP * np.cos(psi)
            control = Control(inhibition=stencil.I.copy(), pilot_signal=pilot)
            u = float(window[i])
            sheet.step(u, control)
            g = sheet.gain
            gain_active_sum += float(np.sum(g > 0.1))
            energy_sum += float(np.sum(g * abs(u)))
            drive_ts[i] = float(np.mean(g[idx] * u))
            pilot_phase_ts[i] = float(np.asarray(psi).reshape(-1)[d_star])
            n_steps_total += 1

        A = sheet.patch_activity()
        S_list.append(selectivity_from_A(A, d_star))
        L_list.append(leakage_from_A(A, d_star, gate_readouts.A0))
        # Primary: pilot-phase-referenced features (not absolute-time FIR)
        bit_hat = gate_readouts.predict_bit(
            drive_ts, d_star, pilot_phase_ts=pilot_phase_ts
        )
        acc_hits += int(bit_hat == int(bits[d_star]))
        c_act_acc += gain_active_sum / S
        c_energy_acc += energy_sum

    return {
        "system": "gate",
        "seed": seed,
        "acc": acc_hits / n_symbols,
        "S": float(np.mean(S_list)),
        "L": float(np.mean(L_list)),
        "C_act": c_act_acc / n_symbols,
        "C_ctl": c_ctl_gate(cond.sigma_j),
        "C_energy": c_energy_acc / n_symbols,
        "n_symbols": n_symbols,
    }


def run_router_seed(
    cond: Condition,
    seed: int,
    n_symbols: int,
    router_weights_path: str,
    readout_path: str,
    router_pilot_readout_path: str | None = None,
) -> dict[str, Any]:
    """
    Primary router arm: no pilot; absolute/content phase-hist bit demod.
    Secondary (same trials): pilot-referenced demod (same alignment ref as gate).
    """
    rng = np.random.default_rng(seed)
    router = load_frozen_router(router_weights_path)
    readouts = load_frozen_readouts(readout_path)
    pilot_ro = None
    if router_pilot_readout_path is not None:
        pilot_ro = RouterPilotReadouts.load(router_pilot_readout_path)
    macs = router.macs_per_decision()
    S = SAMPLES_PER_SYMBOL
    acc_hits = 0
    acc_pilot_hits = 0
    S_list, L_list = [], []
    c_act_acc = 0.0
    c_energy_acc = 0.0

    for sym_i in range(n_symbols):
        d_star = int(rng.integers(0, N_STREAMS))
        bits = rng.integers(0, 2, size=N_STREAMS).astype(np.float64)
        t0 = sym_i * S * DT
        window, shared = make_symbol(
            bits,
            sigma_j=cond.sigma_j,
            sigma_c=cond.sigma_c,
            wobble=cond.wobble,
            t0=t0,
            rng=rng,
        )
        feat = router_features(window, d_star)
        k_hat = int(router.select_topk(feat, k=1)[0])
        ctrl = router.make_control(k_hat)

        sheet = Sheet(
            sigma_f=cond.sigma_f, seed=seed + 17 + sym_i, gain_mode="fixed"
        )
        _apply_receiver_jitter(sheet, cond.sigma_j, rng)

        gain_active_sum = 0.0
        energy_sum = 0.0
        for i in range(S):
            u = float(window[i])
            sheet.step(u, ctrl)
            g = sheet.gain
            gain_active_sum += float(np.sum(g > 0.1))
            energy_sum += float(np.sum(g * abs(u)))

        A = sheet.patch_activity()
        S_list.append(selectivity_from_A(A, d_star))
        L_list.append(leakage_from_A(A, d_star, 0.0))
        # Primary ACC: content phase-hist demod (no pilot), require correct route
        bit_hat = readouts.predict_bit(window, d_star)
        if k_hat == d_star and bit_hat == int(bits[d_star]):
            acc_hits += 1
        # Secondary ACC: same pilot-phase reference as gate demod
        if pilot_ro is not None:
            t = t0 + np.arange(S, dtype=np.float64) * DT
            pilot_phase = 2.0 * np.pi * F0 * t + stream_phase(d_star) + shared
            bit_pilot = pilot_ro.predict_bit(window, d_star, pilot_phase)
            if k_hat == d_star and bit_pilot == int(bits[d_star]):
                acc_pilot_hits += 1
        c_act_acc += gain_active_sum / S
        c_energy_acc += energy_sum

    out = {
        "system": "router",
        "seed": seed,
        "acc": acc_hits / n_symbols,
        "S": float(np.mean(S_list)),
        "L": float(np.mean(L_list)),
        "C_act": c_act_acc / n_symbols,
        "C_ctl": c_ctl_router(macs),
        "C_energy": c_energy_acc / n_symbols,
        "n_symbols": n_symbols,
    }
    if pilot_ro is not None:
        out["acc_pilot_ref"] = acc_pilot_hits / n_symbols
    return out



def measure_T_sw(
    system: str,
    cond: Condition,
    *,
    n_switches: int = N_SWITCHES,
    seed: int = 0,
    router_weights_path: str = str(DEFAULT_WEIGHTS_PATH),
) -> dict[str, Any]:
    """
    T_sw = first time (cycles) at which A_k2 > A_k1 and stays so for one full cycle.
    """
    rng = np.random.default_rng(seed)
    latencies: list[float] = []
    max_horizon_cycles = 20.0
    max_steps = int(max_horizon_cycles * SAMPLES_PER_CYCLE)

    router = None
    if system == "router":
        router = load_frozen_router(router_weights_path)

    for sw in range(n_switches):
        k1 = int(rng.integers(0, N_PATCHES))
        k2 = int(rng.integers(0, N_PATCHES - 1))
        if k2 >= k1:
            k2 += 1

        # Continuous ON content for both streams (bits=1) so activity can build
        bits = np.ones(N_STREAMS, dtype=np.float64)
        # Build a long window with disturbances
        n = max_steps + SAMPLES_PER_CYCLE
        t0 = 0.0
        shared = shared_phase_trajectory(
            n, sigma_c=cond.sigma_c, wobble=cond.wobble, t0=t0, rng=rng
        )
        jitter = per_cycle_phase_jitter(n, cond.sigma_j, rng)
        phase = shared + jitter
        active = {d: 1.0 for d in range(N_STREAMS)}
        window = multiplex_streams(n, active, shared_phase=phase)

        if system == "gate":
            sheet = Sheet(
                sigma_f=cond.sigma_f, seed=seed + sw, gain_mode="oscillatory"
            )
            sheet.phi[:] = stream_phase(k1) + sheet.rng.normal(0.0, 0.1, size=N_UNITS)
            stencil = StencilController(
                open_patch=k1, sigma_c=0.0, shared_delta=0.0, seed=seed + 50 + sw
            )
            stencil.control_for_target(k1, open_patch=k1)
            # Warmup on k1
            warm = SAMPLES_PER_SYMBOL
            for i in range(warm):
                dpsi = 2.0 * np.pi * stencil.carrier_hz * stencil.dt
                stencil.psi = stencil.psi + dpsi
                stencil.t += stencil.dt
                pilot = PILOT_AMP * np.cos(stencil.psi + shared[i])
                sheet.step(window[i], Control(inhibition=stencil.I.copy(), pilot_signal=pilot))
            # Switch to k2
            stencil.control_for_target(k2, open_patch=k2)
            hold = 0
            t_sw = None
            for i in range(warm, warm + max_steps):
                dpsi = 2.0 * np.pi * stencil.carrier_hz * stencil.dt
                stencil.psi = stencil.psi + dpsi
                stencil.t += stencil.dt
                pilot = PILOT_AMP * np.cos(stencil.psi + shared[i])
                sheet.step(window[i], Control(inhibition=stencil.I.copy(), pilot_signal=pilot))
                A = sheet.patch_activity()
                if A[k2] > A[k1]:
                    hold += 1
                    if hold >= SAMPLES_PER_CYCLE:
                        t_sw = (i - warm - SAMPLES_PER_CYCLE + 1) / SAMPLES_PER_CYCLE
                        break
                else:
                    hold = 0
            latencies.append(float(t_sw) if t_sw is not None else max_horizon_cycles)
        else:
            assert router is not None
            sheet = Sheet(sigma_f=cond.sigma_f, seed=seed + sw, gain_mode="fixed")
            ctrl1 = router.make_control(k1)
            warm = SAMPLES_PER_SYMBOL
            for i in range(warm):
                sheet.step(window[i], ctrl1)
            # Router re-selects using a window slice at switch; use goal=k2
            feat_win = window[warm : warm + SAMPLES_PER_SYMBOL]
            if feat_win.size < SAMPLES_PER_SYMBOL:
                feat_win = window[-SAMPLES_PER_SYMBOL:]
            ctrl2 = router.control_for_target(k2, window=feat_win, top_k=1)
            hold = 0
            t_sw = None
            for i in range(warm, warm + max_steps):
                sheet.step(window[i], ctrl2)
                A = sheet.patch_activity()
                if A[k2] > A[k1]:
                    hold += 1
                    if hold >= SAMPLES_PER_CYCLE:
                        t_sw = (i - warm - SAMPLES_PER_CYCLE + 1) / SAMPLES_PER_CYCLE
                        break
                else:
                    hold = 0
            latencies.append(float(t_sw) if t_sw is not None else max_horizon_cycles)

    arr = np.asarray(latencies, dtype=np.float64)
    return {
        "system": system,
        "T_sw_mean": float(arr.mean()),
        "T_sw_std": float(arr.std()),
        "T_sw_median": float(np.median(arr)),
        "n_switches": n_switches,
        "latencies": arr.tolist(),
    }


# ---------------------------------------------------------------------------
# Worker entry (picklable) for multiprocessing
# ---------------------------------------------------------------------------
def _worker_gate(args: tuple) -> dict[str, Any]:
    cond_dict, seed, n_symbols, gate_path = args
    cond = Condition(
        cond_dict["sigma_f"],
        cond_dict["sigma_j"],
        cond_dict["sigma_c"],
        cond_dict["wobble"],
    )
    gr = GateReadouts.load(gate_path)
    out = run_gate_seed(cond, seed, n_symbols, gr)
    out["condition"] = cond.as_dict()
    return out


def _worker_router(args: tuple) -> dict[str, Any]:
    # args: cond_dict, seed, n_symbols, weights_path, readout_path [, pilot_readout_path]
    cond_dict, seed, n_symbols, weights_path, readout_path = args[:5]
    pilot_path = args[5] if len(args) > 5 else None
    cond = Condition(
        cond_dict["sigma_f"],
        cond_dict["sigma_j"],
        cond_dict["sigma_c"],
        cond_dict["wobble"],
    )
    out = run_router_seed(
        cond, seed, n_symbols, weights_path, readout_path, pilot_path
    )
    out["condition"] = cond.as_dict()
    return out


def _worker_switch(args: tuple) -> dict[str, Any]:
    system, cond_dict, n_switches, seed, weights_path = args
    cond = Condition(
        cond_dict["sigma_f"],
        cond_dict["sigma_j"],
        cond_dict["sigma_c"],
        cond_dict["wobble"],
    )
    out = measure_T_sw(
        system,
        cond,
        n_switches=n_switches,
        seed=seed,
        router_weights_path=weights_path,
    )
    out["condition"] = cond.as_dict()
    return out


# ---------------------------------------------------------------------------
# Aggregation / I/O / figures / H1
# ---------------------------------------------------------------------------
def aggregate_condition(
    cond: Condition,
    gate_seed_results: list[dict[str, Any]],
    router_seed_results: list[dict[str, Any]],
    switch_gate: dict[str, Any] | None,
    switch_router: dict[str, Any] | None,
) -> dict[str, Any]:
    def pack(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
        vals = np.array([r[metric] for r in rows], dtype=np.float64)
        mean, lo, hi = bootstrap_ci(vals, seed=42)
        return {"mean": mean, "ci95_lo": lo, "ci95_hi": hi, "per_seed": vals.tolist()}

    out: dict[str, Any] = {
        "condition": cond.as_dict(),
        "gate": {
            "ACC": pack(gate_seed_results, "acc"),
            "S": pack(gate_seed_results, "S"),
            "L": pack(gate_seed_results, "L"),
            "C_act": pack(gate_seed_results, "C_act"),
            "C_ctl": gate_seed_results[0]["C_ctl"] if gate_seed_results else None,
            "C_energy": pack(gate_seed_results, "C_energy"),
        },
        "router": {
            "ACC": pack(router_seed_results, "acc"),
            "ACC_pilot_ref": (
                pack(router_seed_results, "acc_pilot_ref")
                if router_seed_results and "acc_pilot_ref" in router_seed_results[0]
                else None
            ),
            "S": pack(router_seed_results, "S"),
            "L": pack(router_seed_results, "L"),
            "C_act": pack(router_seed_results, "C_act"),
            "C_ctl": router_seed_results[0]["C_ctl"] if router_seed_results else None,
            "C_energy": pack(router_seed_results, "C_energy"),
        },
    }
    if switch_gate is not None:
        out["gate"]["T_sw"] = {
            "mean": switch_gate["T_sw_mean"],
            "std": switch_gate["T_sw_std"],
            "median": switch_gate["T_sw_median"],
        }
    if switch_router is not None:
        out["router"]["T_sw"] = {
            "mean": switch_router["T_sw_mean"],
            "std": switch_router["T_sw_std"],
            "median": switch_router["T_sw_median"],
        }
    return out


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "key",
        "sigma_f",
        "sigma_j",
        "sigma_c",
        "wobble",
        "ACC_gate",
        "ACC_gate_lo",
        "ACC_gate_hi",
        "ACC_router",
        "ACC_router_pilot_ref",
        "ACC_router_lo",
        "ACC_router_hi",
        "S_gate",
        "S_router",
        "L_gate",
        "L_router",
        "C_act_gate",
        "C_act_gate_lo",
        "C_act_gate_hi",
        "C_act_router",
        "C_act_router_lo",
        "C_act_router_hi",
        "C_ctl_gate",
        "C_ctl_router",
        "C_energy_gate",
        "C_energy_router",
        "T_sw_gate",
        "T_sw_router",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            c = r["condition"]
            g, rt = r["gate"], r["router"]
            w.writerow(
                {
                    "key": c["key"],
                    "sigma_f": c["sigma_f"],
                    "sigma_j": c["sigma_j"],
                    "sigma_c": c["sigma_c"],
                    "wobble": int(c["wobble"]),
                    "ACC_gate": g["ACC"]["mean"],
                    "ACC_gate_lo": g["ACC"]["ci95_lo"],
                    "ACC_gate_hi": g["ACC"]["ci95_hi"],
                    "ACC_router": rt["ACC"]["mean"],
                    "ACC_router_pilot_ref": (
                        rt["ACC_pilot_ref"]["mean"]
                        if rt.get("ACC_pilot_ref") else ""
                    ),
                    "ACC_router_lo": rt["ACC"]["ci95_lo"],
                    "ACC_router_hi": rt["ACC"]["ci95_hi"],
                    "S_gate": g["S"]["mean"],
                    "S_router": rt["S"]["mean"],
                    "L_gate": g["L"]["mean"],
                    "L_router": rt["L"]["mean"],
                    "C_act_gate": g["C_act"]["mean"],
                    "C_act_gate_lo": g["C_act"]["ci95_lo"],
                    "C_act_gate_hi": g["C_act"]["ci95_hi"],
                    "C_act_router": rt["C_act"]["mean"],
                    "C_act_router_lo": rt["C_act"]["ci95_lo"],
                    "C_act_router_hi": rt["C_act"]["ci95_hi"],
                    "C_ctl_gate": g["C_ctl"],
                    "C_ctl_router": rt["C_ctl"],
                    "C_energy_gate": g["C_energy"]["mean"],
                    "C_energy_router": rt["C_energy"]["mean"],
                    "T_sw_gate": g.get("T_sw", {}).get("mean", ""),
                    "T_sw_router": rt.get("T_sw", {}).get("mean", ""),
                }
            )


def make_figures(summary_rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    # Use non-wobble grid cells for main heat-style plots; also line plots vs σ_c
    grid = [r for r in summary_rows if not r["condition"]["wobble"]]
    if not grid:
        grid = summary_rows

    def _xy(rows, key_g, key_r):
        # plot vs index for smoke; for fuller grids facet by σ_f,σ_j
        xs = np.arange(len(rows))
        labels = [r["condition"]["key"] for r in rows]
        yg = [r["gate"][key_g]["mean"] if isinstance(r["gate"][key_g], dict) else r["gate"][key_g] for r in rows]
        yr = [r["router"][key_r]["mean"] if isinstance(r["router"][key_r], dict) else r["router"][key_r] for r in rows]
        return xs, labels, yg, yr

    # --- ACC ---
    fig, ax = plt.subplots(figsize=(10, 4.5))
    xs, labels, yg, yr = _xy(grid, "ACC", "ACC")
    ax.plot(xs, yg, "o-", label="gate", color="C0")
    ax.plot(xs, yr, "s-", label="router", color="C1")
    # CI whiskers
    for i, r in enumerate(grid):
        ax.vlines(i, r["gate"]["ACC"]["ci95_lo"], r["gate"]["ACC"]["ci95_hi"], color="C0", alpha=0.5)
        ax.vlines(i, r["router"]["ACC"]["ci95_lo"], r["router"]["ACC"]["ci95_hi"], color="C1", alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("ACC")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Task accuracy (primary) — gate vs router")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "fig_ACC.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # --- S / L ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    xs, labels, yg, yr = _xy(grid, "S", "S")
    axes[0].plot(xs, yg, "o-", label="gate")
    axes[0].plot(xs, yr, "s-", label="router")
    axes[0].set_title("Selectivity S")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    xs, labels, yg, yr = _xy(grid, "L", "L")
    axes[1].plot(xs, yg, "o-", label="gate")
    axes[1].plot(xs, yr, "s-", label="router")
    axes[1].set_title("Leakage L")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "fig_S_L.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # --- T_sw ---
    fig, ax = plt.subplots(figsize=(10, 4.5))
    has_tsw = any("T_sw" in r["gate"] for r in grid)
    if has_tsw:
        xs = np.arange(len(grid))
        labels = [r["condition"]["key"] for r in grid]
        yg = [r["gate"].get("T_sw", {}).get("mean", np.nan) for r in grid]
        yr = [r["router"].get("T_sw", {}).get("mean", np.nan) for r in grid]
        ax.plot(xs, yg, "o-", label="gate")
        ax.plot(xs, yr, "s-", label="router")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("T_sw (cycles)")
        ax.set_title("Switching latency")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "T_sw not measured in this run\n(use --switch or --full)", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    p = out_dir / "fig_T_sw.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # --- Cost ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    xs = np.arange(len(grid))
    labels = [r["condition"]["key"] for r in grid]
    for ax, metric, title in zip(
        axes,
        ["C_act", "C_ctl", "C_energy"],
        [r"$C_{act}$", r"$C_{ctl}$", r"$C_{energy}$"],
    ):
        if metric == "C_ctl":
            yg = [r["gate"]["C_ctl"] for r in grid]
            yr = [r["router"]["C_ctl"] for r in grid]
        else:
            yg = [r["gate"][metric]["mean"] for r in grid]
            yr = [r["router"][metric]["mean"] for r in grid]
        ax.plot(xs, yg, "o-", label="gate")
        ax.plot(xs, yr, "s-", label="router")
        ax.set_title(title)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Cost channels (never combined)")
    fig.tight_layout()
    p = out_dir / "fig_cost.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)
    return paths


def h1_verdict(summary_rows: list[dict[str, Any]]) -> str:
    """
    H1 if:
      clean: ACC_gate ≥ ACC_router − 0.02 AND C_act_gate ≤ 0.5 × C_act_router
      AND at σ_c=0.05 (σ_f=σ_j=0): drop_gate smaller than drop_router by more than CI width
    else H0 or mixed.
    """
    def find(sf, sj, sc, wobble=False):
        for r in summary_rows:
            c = r["condition"]
            if (
                abs(c["sigma_f"] - sf) < 1e-12
                and abs(c["sigma_j"] - sj) < 1e-12
                and abs(c["sigma_c"] - sc) < 1e-12
                and bool(c["wobble"]) == wobble
            ):
                return r
        return None

    clean = find(0.0, 0.0, 0.0, False)
    drift = find(0.0, 0.0, 0.05, False)
    lines = ["H1 verdict (PREREG §6)", "=" * 40]
    if clean is None:
        lines.append("Missing clean cell — cannot evaluate. Verdict: mixed")
        return "\n".join(lines) + "\n"

    ag = clean["gate"]["ACC"]["mean"]
    ar = clean["router"]["ACC"]["mean"]
    cg = clean["gate"]["C_act"]["mean"]
    cr = clean["router"]["C_act"]["mean"]
    clean_acc_ok = ag >= (ar - 0.02)
    clean_cost_ok = cg <= 0.5 * cr
    lines.append(f"Clean ACC_gate={ag:.4f} ACC_router={ar:.4f}  (need gate≥router−0.02): {clean_acc_ok}")
    lines.append(f"Clean C_act_gate={cg:.2f} C_act_router={cr:.2f}  (need gate≤0.5×router): {clean_cost_ok}")

    drift_ok = False
    if drift is None:
        lines.append("Missing σ_c=0.05 cell — drift clause not evaluable.")
    else:
        # drop = ACC_clean - ACC_drift
        drop_g = ag - drift["gate"]["ACC"]["mean"]
        drop_r = ar - drift["router"]["ACC"]["mean"]
        # CI width: use max of the two systems' ACC CI half-widths at drift (or clean)
        def halfwidth(cell, sys):
            d = cell[sys]["ACC"]
            return 0.5 * (d["ci95_hi"] - d["ci95_lo"])

        ci_w = max(
            halfwidth(clean, "gate"),
            halfwidth(clean, "router"),
            halfwidth(drift, "gate"),
            halfwidth(drift, "router"),
        )
        # drop_gate smaller than drop_router by more than CI width
        # ⇒ drop_r - drop_g > ci_w
        margin = drop_r - drop_g
        drift_ok = margin > ci_w
        lines.append(
            f"Drift σ_c=0.05: drop_gate={drop_g:.4f} drop_router={drop_r:.4f} "
            f"margin={margin:.4f} CI_width={ci_w:.4f} → drift_ok={drift_ok}"
        )

    if clean_acc_ok and clean_cost_ok and drift_ok:
        verdict = "H1"
    elif not clean_acc_ok and not clean_cost_ok and not drift_ok:
        verdict = "H0"
    else:
        # Partial satisfaction
        if clean_acc_ok and clean_cost_ok and drift is None:
            verdict = "mixed"
        elif (clean_acc_ok and clean_cost_ok) or drift_ok:
            verdict = "mixed"
        else:
            verdict = "H0"
    lines.append(
        "Note: 'more gracefully' = relative drop criterion above (drop_gate < drop_router by > CI width); absolute ACC need not stay near clean."
    )
    lines.append(
        "Secondary arm (router+pilot-ref demod) is reported in summary but is NOT part of the H1 primary criteria."
    )
    lines.append(f"Verdict: {verdict}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def run_comparison(
    *,
    full: bool = False,
    primary: bool = False,
    do_switch: bool = False,
    n_workers: int | None = None,
    out_dir: Path | None = None,
    prereg_sha256: str | None = None,
    start_time_iso: str | None = None,
) -> dict[str, Any]:
    from datetime import datetime

    t_wall0 = time.perf_counter()
    if out_dir is None:
        out_dir = PRIMARY_RESULTS if primary else RESULTS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if start_time_iso is None:
        start_time_iso = datetime.now().astimezone().isoformat(timespec="seconds")

    if full:
        conditions = full_grid_conditions()
        n_symbols = FULL_N_SYMBOLS
        seeds = list(range(FULL_SEED0, FULL_SEED0 + FULL_N_SEEDS))
        do_switch = True
        mode = "full"
    elif primary:
        conditions = primary_conditions()
        n_symbols = FULL_N_SYMBOLS
        seeds = list(range(FULL_SEED0, FULL_SEED0 + FULL_N_SEEDS))
        do_switch = True
        mode = "primary"
    else:
        conditions = smoke_conditions()
        n_symbols = SMOKE_N_SYMBOLS
        seeds = list(range(SMOKE_SEED0, SMOKE_SEED0 + SMOKE_N_SEEDS))
        mode = "smoke"

    print(f"Mode={mode}  conditions={len(conditions)}  symbols={n_symbols}  seeds={len(seeds)}")
    print(f"Switch={do_switch}  workers={n_workers or 'auto'}")
    print(f"start_time_iso={start_time_iso}")

    # Ensure frozen router + gate readouts
    if not Path(DEFAULT_WEIGHTS_PATH).exists():
        raise FileNotFoundError(
            f"Missing frozen router weights at {DEFAULT_WEIGHTS_PATH}; run train_router.py first"
        )
    gate_readouts = load_or_train_gate_readouts(GATE_READOUT_PATH)
    _ = gate_readouts  # trained/loaded on disk for workers

    weights_path = str(DEFAULT_WEIGHTS_PATH)
    readout_path = str(DEFAULT_READOUT_PATH)
    pilot_readout_path = str(ROUTER_PILOT_READOUT_PATH)
    _ = load_or_train_router_pilot_readouts(ROUTER_PILOT_READOUT_PATH)
    gate_path = str(GATE_READOUT_PATH)

    # Schedule jobs
    gate_jobs = []
    router_jobs = []
    for cond in conditions:
        cd = cond.as_dict()
        for seed in seeds:
            # Same seed feeds both systems (identical trial streams)
            gate_jobs.append((cd, seed, n_symbols, gate_path))
            router_jobs.append((cd, seed, n_symbols, weights_path, readout_path, pilot_readout_path))

    switch_jobs = []
    if do_switch:
        for cond in conditions:
            cd = cond.as_dict()
            switch_jobs.append(("gate", cd, N_SWITCHES, 12345, weights_path))
            switch_jobs.append(("router", cd, N_SWITCHES, 12345, weights_path))

    max_workers = n_workers or min(8, (len(gate_jobs) + len(router_jobs) + len(switch_jobs)) or 1)

    gate_results: list[dict[str, Any]] = []
    router_results: list[dict[str, Any]] = []
    switch_results: list[dict[str, Any]] = []

    # Prefer process pool for CPU-bound sheet stepping
    print(f"Running {len(gate_jobs)} gate + {len(router_jobs)} router seed-jobs …")
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {}
        for j in gate_jobs:
            futs[ex.submit(_worker_gate, j)] = ("gate", j[0]["key"], j[1])
        for j in router_jobs:
            futs[ex.submit(_worker_router, j)] = ("router", j[0]["key"], j[1])
        for j in switch_jobs:
            futs[ex.submit(_worker_switch, j)] = ("switch", j[1]["key"], j[0])

        done = 0
        total = len(futs)
        for fut in as_completed(futs):
            kind, key, meta = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"JOB FAIL {kind} {key} {meta}: {e}")
                raise
            if kind == "gate":
                gate_results.append(res)
            elif kind == "router":
                router_results.append(res)
            else:
                switch_results.append(res)
            done += 1
            if done % max(1, total // 10) == 0 or done == total:
                print(f"  progress {done}/{total}")

    # Aggregate per condition
    summary_rows: list[dict[str, Any]] = []
    for cond in conditions:
        g_seeds = [r for r in gate_results if r["condition"]["key"] == cond.key()]
        r_seeds = [r for r in router_results if r["condition"]["key"] == cond.key()]
        sg = next(
            (s for s in switch_results if s["condition"]["key"] == cond.key() and s["system"] == "gate"),
            None,
        )
        sr = next(
            (s for s in switch_results if s["condition"]["key"] == cond.key() and s["system"] == "router"),
            None,
        )
        agg = aggregate_condition(cond, g_seeds, r_seeds, sg, sr)
        summary_rows.append(agg)
        # One JSON per condition
        jpath = out_dir / f"{cond.key()}.json"
        with jpath.open("w") as f:
            json.dump(agg, f, indent=2)

    csv_path = out_dir / "summary.csv"
    write_summary_csv(summary_rows, csv_path)
    fig_paths = make_figures(summary_rows, out_dir)

    verdict_text = h1_verdict(summary_rows)
    verdict_path = out_dir / "H1_verdict.txt"
    verdict_path.write_text(verdict_text)
    print(verdict_text)

    # Nested all-in-one JSON
    all_path = out_dir / "all_conditions.json"
    with all_path.open("w") as f:
        json.dump(
            {
                "mode": mode,
                "n_symbols": n_symbols,
                "seeds": seeds,
                "conditions": summary_rows,
            },
            f,
            indent=2,
        )

    elapsed = time.perf_counter() - t_wall0
    meta = {
        "mode": mode,
        "prereg_sha256": prereg_sha256,
        "start_time_iso": start_time_iso,
        "elapsed_sec": elapsed,
        "n_symbols": n_symbols,
        "n_seeds": len(seeds),
        "n_conditions": len(conditions),
        "do_switch": do_switch,
        "csv": str(csv_path),
        "figures": [str(p) for p in fig_paths],
        "verdict_path": str(verdict_path),
        "verdict_text": verdict_text,
        "router_readout_mode": "phase_hist",
        "out_dir": str(out_dir),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done in {elapsed:.1f}s  → {csv_path}")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description="T5: oscillatory gate vs learned router comparison")
    p.add_argument(
        "--full",
        action="store_true",
        help="Preregistered protocol: 200 symbols × 10 seeds × full 4×4×4 + wobble (+ switches)",
    )
    p.add_argument(
        "--primary",
        action="store_true",
        help=(
            "Primary registered denser-than-smoke grid with prereg trial counts "
            "(200×10); writes results/comparison_primary/. Use after PREREG.sha256."
        ),
    )
    p.add_argument(
        "--switch",
        action="store_true",
        help="Measure T_sw (50 switches/condition). Implied by --full/--primary.",
    )
    p.add_argument("--workers", type=int, default=None, help="Process pool size (default: 8)")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: results/comparison or comparison_primary)",
    )
    p.add_argument(
        "--prereg-sha256",
        type=str,
        default=None,
        help="Prereg content hash to log in run_meta.json",
    )
    args = p.parse_args()
    run_comparison(
        full=args.full,
        primary=args.primary,
        do_switch=args.switch,
        n_workers=args.workers,
        out_dir=args.out,
        prereg_sha256=args.prereg_sha256,
    )


if __name__ == "__main__":
    main()
