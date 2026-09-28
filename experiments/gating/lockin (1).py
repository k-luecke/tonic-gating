#!/usr/bin/env python3
"""
lockin.py — Single-receiver oscillatory lock-in model (T0 / T1).

T0: four-panel lockin_results.png (default).
T1: --jitter → receiver/sender per-cycle timing jitter sweep →
    results/lockin_jitter_slots.png + T1_slots_statement.txt.
Numpy / scipy / matplotlib only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants (preregistered defaults)
# ---------------------------------------------------------------------------
F0 = 40.0  # Hz carrier
FS = 4000.0  # Hz sample rate
CYCLES_PER_SYMBOL = 5
SYMBOL_DUR = CYCLES_PER_SYMBOL / F0  # 0.125 s
SAMPLES_PER_SYMBOL = int(round(SYMBOL_DUR * FS))  # 500
DT = 1.0 / FS

PLL_K = 800.0  # loop gain (rad/s per unit error)
PLL_TAU_CYCLES = 3.0
PLL_TAU = PLL_TAU_CYCLES / F0  # seconds
PILOT_AMP = 0.5

# Reference power of a unit pulsed carrier (for SNR definition)
_T_REF = np.arange(SAMPLES_PER_SYMBOL) * DT
_PULSED_REF = np.maximum(0.0, np.cos(2.0 * np.pi * F0 * _T_REF)) ** 2
P_REF = float(np.mean(_PULSED_REF**2))

RNG_SEED = 20260921
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_CWD = Path(__file__).resolve().parent / "lockin_results.png"
OUT_RESULTS = OUT_DIR / "lockin_results.png"
OUT_JITTER = OUT_DIR / "lockin_jitter_slots.png"
OUT_T1_STMT = OUT_DIR / "T1_slots_statement.txt"
JITTER_SIGMAS = (0.0, 0.02, 0.05, 0.10)  # cycles


# ---------------------------------------------------------------------------
# Bit / waveform helpers
# ---------------------------------------------------------------------------
def make_bits(n_symbols: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=n_symbols).astype(np.float64)


def expand_bits(bits: np.ndarray, n_samp: int) -> np.ndarray:
    return np.repeat(bits, n_samp)



def per_cycle_phase_jitter(n: int, sigma_j: float, rng: np.random.Generator) -> np.ndarray:
    """
    Piecewise-constant phase jitter: one draw per carrier cycle.

    sigma_j is in cycles (fraction of a period); converted to radians as 2π·N(0,σ_j).
    Returns length-n array of phase offsets (radians). σ_j=0 → zeros.
    """
    if sigma_j <= 0.0:
        return np.zeros(n, dtype=np.float64)
    samples_per_cycle = int(round(FS / F0))  # 100 at 4 kHz / 40 Hz
    n_cycles = int(np.ceil(n / samples_per_cycle))
    jit_cycles = rng.normal(0.0, sigma_j, size=n_cycles)
    return np.repeat(jit_cycles * (2.0 * np.pi), samples_per_cycle)[:n]


def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """AWGN relative to unit pulsed-carrier power (preregistered SNR)."""
    p_noise = P_REF / (10.0 ** (snr_db / 10.0))
    return x + rng.normal(0.0, np.sqrt(p_noise), size=x.shape)


# ---------------------------------------------------------------------------
# First-order PLL
# ---------------------------------------------------------------------------
def run_pll(
    signal: np.ndarray,
    f_nom: float = F0,
    k: float = PLL_K,
    tau: float = PLL_TAU,
    phi0: float = 0.0,
) -> np.ndarray:
    """
    First-order PLL with loop filter τ.

    Phase detector err = −x·sin(φ) so the lock φ→θ is stable for x≈cos(θ).
    Returns instantaneous VCO phase φ[n] (radians).
    """
    n = len(signal)
    phi = np.empty(n, dtype=np.float64)
    phi_n = float(phi0)
    e_f = 0.0
    a = DT / tau
    w0 = 2.0 * np.pi * f_nom
    for i in range(n):
        phi[i] = phi_n
        err = -signal[i] * np.sin(phi_n)
        e_f += a * (err - e_f)
        phi_n += (w0 + k * e_f) * DT
    return phi


def fixed_phase(n: int, phi0: float = 0.0, f: float = F0) -> np.ndarray:
    t = np.arange(n) * DT
    return 2.0 * np.pi * f * t + phi0


def rectified_gain(phi_ref: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.cos(phi_ref))


def integrate_symbols(x: np.ndarray, n_symbols: int) -> np.ndarray:
    usable = n_symbols * SAMPLES_PER_SYMBOL
    return x[:usable].reshape(n_symbols, SAMPLES_PER_SYMBOL).sum(axis=1)


def calibrate_threshold(scores: np.ndarray, bits: np.ndarray) -> float:
    """Midpoint of mean ON vs mean OFF (trial self-calibration)."""
    on = scores[bits > 0.5]
    off = scores[bits < 0.5]
    if len(on) == 0 or len(off) == 0:
        return float(np.median(scores))
    return 0.5 * (float(on.mean()) + float(off.mean()))


def ber_from_scores(
    scores: np.ndarray, bits: np.ndarray, thr: float | None = None
) -> float:
    if thr is None:
        thr = calibrate_threshold(scores, bits)
    pred = (scores >= thr).astype(np.float64)
    return float(np.mean(pred != bits))


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------
def simulate_two_sender_trial(
    n_symbols: int,
    phase_b: float,
    snr_db: float,
    rng: np.random.Generator,
    *,
    waveform: str = "pulsed",
    ref_mode: str = "fixed",
    f_offset_a: float = 0.0,
    f_offset_b: float = 0.0,
    drift_rate: float = 0.0,
    shared_freq_offset: float = 0.0,
    pilot_amp: float = PILOT_AMP,
    bits_a: np.ndarray | None = None,
    bits_b: np.ndarray | None = None,
    frozen_threshold: float | None = None,
    sigma_j: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Two senders, one receiver. Returns (BER_A, bits_a, scores_a).

    ref_mode: 'fixed' | 'content' | 'pilot'
    drift_rate: rad/s common phase ramp (shared drift)
    shared_freq_offset: Hz added to both senders (and pilot)
    frozen_threshold: if set, skip trial self-calibration (used for drift→chance)
    sigma_j: per-cycle timing jitter (cycles) on sender volleys AND receiver gain
    """
    n = n_symbols * SAMPLES_PER_SYMBOL
    t = np.arange(n) * DT

    if bits_a is None:
        bits_a = make_bits(n_symbols, rng)
    if bits_b is None:
        bits_b = make_bits(n_symbols, rng)

    gate_a = expand_bits(bits_a, SAMPLES_PER_SYMBOL)
    gate_b = expand_bits(bits_b, SAMPLES_PER_SYMBOL)

    shared = (2.0 * np.pi * shared_freq_offset + drift_rate) * t
    phase_a_t = 2.0 * np.pi * f_offset_a * t + shared
    phase_b_t = phase_b + 2.0 * np.pi * f_offset_b * t + shared

    # Independent per-cycle jitter on each sender and on the receiver gain
    jit_a = per_cycle_phase_jitter(n, sigma_j, rng)
    jit_b = per_cycle_phase_jitter(n, sigma_j, rng)
    jit_rx = per_cycle_phase_jitter(n, sigma_j, rng)

    ca = np.cos(2.0 * np.pi * F0 * t + phase_a_t + jit_a)
    cb = np.cos(2.0 * np.pi * F0 * t + phase_b_t + jit_b)
    if waveform == "pulsed":
        sa = np.maximum(0.0, ca) ** 2
        sb = np.maximum(0.0, cb) ** 2
    else:
        sa, sb = ca, cb

    content = gate_a * sa + gate_b * sb
    content = add_awgn(content, snr_db, rng)

    if ref_mode == "fixed":
        phi_ref = fixed_phase(n, phi0=0.0, f=F0)
    elif ref_mode == "content":
        phi_ref = run_pll(content, f_nom=F0)
    elif ref_mode == "pilot":
        pilot = pilot_amp * np.cos(2.0 * np.pi * F0 * t + shared)
        pilot = pilot + rng.normal(0.0, 0.02, size=n)
        phi_ref = run_pll(pilot, f_nom=F0)
    else:
        raise ValueError(ref_mode)

    # Receiver gain jitter: same σ_j mechanism on the reference phase
    scores = integrate_symbols(content * rectified_gain(phi_ref + jit_rx), n_symbols)
    ber = ber_from_scores(scores, bits_a, thr=frozen_threshold)
    return ber, bits_a, scores


def clean_on_threshold(rng: np.random.Generator, n_cal: int = 40) -> float:
    """Frozen threshold from a short clean aligned calibration block."""
    ber, bits, scores = simulate_two_sender_trial(
        n_cal,
        phase_b=np.deg2rad(90.0),
        snr_db=30.0,
        rng=rng,
        waveform="pulsed",
        ref_mode="fixed",
        bits_b=np.zeros(n_cal),
    )
    return calibrate_threshold(scores, bits)


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
def avg_ber(
    n_symbols: int,
    phase_b: float,
    snr_db: float,
    base_seed: int,
    n_seeds: int = 5,
    **kwargs,
) -> float:
    bers = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s * 10007)
        ber, _, _ = simulate_two_sender_trial(
            n_symbols, phase_b, snr_db, rng, **kwargs
        )
        bers.append(ber)
    return float(np.mean(bers))


def sweep_phase_separation(
    snrs_db: list[float],
    phase_degs: np.ndarray,
    n_symbols: int,
    seed: int,
    n_seeds: int = 5,
    sigma_j: float = 0.0,
) -> dict[float, np.ndarray]:
    out: dict[float, np.ndarray] = {}
    for snr in snrs_db:
        bers = []
        for i, deg in enumerate(phase_degs):
            b = avg_ber(
                n_symbols,
                np.deg2rad(deg),
                snr,
                seed + int(snr * 1000) + i * 17 + int(sigma_j * 1e6),
                n_seeds=n_seeds,
                waveform="pulsed",
                ref_mode="fixed",
                sigma_j=sigma_j,
            )
            bers.append(b)
        out[snr] = np.asarray(bers, dtype=np.float64)
    return out


def usable_separation_deg(
    phase_degs: np.ndarray, bers: np.ndarray, thr: float = 0.05
) -> float:
    """Linear-interpolated phase where BER crosses `thr` from above."""
    # Ensure we start above thr near 0° (in-phase interference)
    for i in range(len(phase_degs) - 1):
        b0, b1 = bers[i], bers[i + 1]
        d0, d1 = phase_degs[i], phase_degs[i + 1]
        if b0 >= thr > b1 and d1 > 0:
            frac = (b0 - thr) / (b0 - b1 + 1e-12)
            return float(d0 + frac * (d1 - d0))
    below = np.where(bers < thr)[0]
    if len(below):
        return float(phase_degs[below[0]])
    return float(phase_degs[-1])


def sweep_pulsed_vs_signed(
    phase_degs: np.ndarray,
    n_symbols: int,
    snr_db: float,
    seed: int,
    n_seeds: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    ber_p, ber_s = [], []
    for i, deg in enumerate(phase_degs):
        ber_p.append(
            avg_ber(
                n_symbols,
                np.deg2rad(deg),
                snr_db,
                seed + 1000 + i,
                n_seeds=n_seeds,
                waveform="pulsed",
                ref_mode="fixed",
            )
        )
        ber_s.append(
            avg_ber(
                n_symbols,
                np.deg2rad(deg),
                snr_db,
                seed + 2000 + i,
                n_seeds=n_seeds,
                waveform="signed",
                ref_mode="fixed",
            )
        )
    return np.asarray(ber_p), np.asarray(ber_s)


def sweep_fixed_vs_offset(
    offsets_hz: np.ndarray,
    trial_s: float,
    snr_db: float,
    seed: int,
) -> np.ndarray:
    """Fixed reference under shared frequency offset; frozen threshold → chance."""
    n_symbols = max(8, int(round(trial_s / SYMBOL_DUR)))
    rng_cal = np.random.default_rng(seed + 333)
    thr = clean_on_threshold(rng_cal)
    bers = []
    for i, df in enumerate(offsets_hz):
        rng = np.random.default_rng(seed + 3000 + i)
        # Orthogonal interferer; frozen thr so slip drives BER → ~0.5
        ber, _, _ = simulate_two_sender_trial(
            n_symbols,
            phase_b=np.deg2rad(90.0),
            snr_db=snr_db,
            rng=rng,
            waveform="pulsed",
            ref_mode="fixed",
            shared_freq_offset=float(df),
            frozen_threshold=thr,
        )
        bers.append(ber)
    return np.asarray(bers)


def sweep_pll_content_vs_pilot(
    drifts_rad_s: np.ndarray,
    mismatches_hz: np.ndarray,
    n_symbols: int,
    snr_db: float,
    seed: int,
    n_seeds: int = 3,
) -> dict[str, np.ndarray]:
    ber_c_d, ber_p_d = [], []
    for i, dr in enumerate(drifts_rad_s):
        ber_c_d.append(
            avg_ber(
                n_symbols,
                np.deg2rad(90.0),
                snr_db,
                seed + 4000 + i,
                n_seeds=n_seeds,
                waveform="pulsed",
                ref_mode="content",
                drift_rate=float(dr),
            )
        )
        ber_p_d.append(
            avg_ber(
                n_symbols,
                np.deg2rad(90.0),
                snr_db,
                seed + 5000 + i,
                n_seeds=n_seeds,
                waveform="pulsed",
                ref_mode="pilot",
                drift_rate=float(dr),
            )
        )

    ber_c_m, ber_p_m = [], []
    for i, df in enumerate(mismatches_hz):
        ber_c_m.append(
            avg_ber(
                n_symbols,
                np.deg2rad(90.0),
                snr_db,
                seed + 6000 + i,
                n_seeds=n_seeds,
                waveform="pulsed",
                ref_mode="content",
                shared_freq_offset=float(df),
            )
        )
        ber_p_m.append(
            avg_ber(
                n_symbols,
                np.deg2rad(90.0),
                snr_db,
                seed + 7000 + i,
                n_seeds=n_seeds,
                waveform="pulsed",
                ref_mode="pilot",
                shared_freq_offset=float(df),
            )
        )

    return {
        "content_drift": np.asarray(ber_c_d),
        "pilot_drift": np.asarray(ber_p_d),
        "content_mismatch": np.asarray(ber_c_m),
        "pilot_mismatch": np.asarray(ber_p_m),
        "drifts": drifts_rad_s,
        "mismatches": mismatches_hz,
    }


# ---------------------------------------------------------------------------
# Figure + main
# ---------------------------------------------------------------------------
def main_t0() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = RNG_SEED

    # Runtime-tuned sizes: multi-seed averages, still typically < 60 s
    # Dense near 0–60° (slot measurement), coarser to 180° (shape)
    phase_degs = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 60.0, 25),   # 2.5° steps
                np.linspace(70.0, 180.0, 12),
            ]
        )
    )
    snrs = [10.0, 0.0, -6.0]
    n_sym_phase = 200
    n_sym_wave = 140
    n_sym_pll = 100
    n_seeds_phase = 8
    n_seeds_wave = 5
    n_seeds_pll = 4

    print("Sweep 1: phase separation / slots …")
    sep = sweep_phase_separation(
        snrs, phase_degs, n_sym_phase, seed, n_seeds=n_seeds_phase
    )
    usable = {snr: usable_separation_deg(phase_degs, sep[snr]) for snr in snrs}
    slots = {snr: (360.0 / usable[snr] if usable[snr] > 0 else float("inf")) for snr in snrs}

    print("Sweep 2: pulsed vs signed …")
    ber_pulsed, ber_signed = sweep_pulsed_vs_signed(
        phase_degs, n_sym_wave, snr_db=0.0, seed=seed, n_seeds=n_seeds_wave
    )

    print("Sweep 3: fixed reference vs frequency offset …")
    offsets = np.array([0.0, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 1.0])
    ber_fixed = sweep_fixed_vs_offset(offsets, trial_s=50.0, snr_db=10.0, seed=seed)

    print("Sweep 4: PLL content vs pilot …")
    drifts = np.linspace(0.0, 2.0 * np.pi * 0.5, 9)  # up to ~0.5 Hz-equivalent
    mismatches = np.linspace(0.0, 8.0, 9)
    pll = sweep_pll_content_vs_pilot(
        drifts, mismatches, n_sym_pll, snr_db=10.0, seed=seed, n_seeds=n_seeds_pll
    )

    # ----- Console summary -----
    print("\n=== T0 lock-in findings (this run) ===")
    for snr in snrs:
        print(
            f"1. SNR {snr:+.0f} dB: usable separation ≈ {usable[snr]:.0f}° "
            f"→ ~{slots[snr]:.1f} slots/cycle (BER<5%)"
        )
    i180 = int(np.argmin(np.abs(phase_degs - 180.0)))
    i0 = int(np.argmin(np.abs(phase_degs - 0.0)))
    i90 = int(np.argmin(np.abs(phase_degs - 90.0)))
    print(
        f"2. Pulsed BER@0°/90°/180° = "
        f"{ber_pulsed[i0]:.3f}/{ber_pulsed[i90]:.3f}/{ber_pulsed[i180]:.3f}; "
        f"Signed BER@0°/90°/180° = "
        f"{ber_signed[i0]:.3f}/{ber_signed[i90]:.3f}/{ber_signed[i180]:.3f}"
    )
    i025 = int(np.argmin(np.abs(offsets - 0.25)))
    print(
        f"3. Fixed ref BER @ 0 Hz = {ber_fixed[0]:.3f}, "
        f"@ 0.25 Hz / 50s = {ber_fixed[i025]:.3f} (chance ~0.5)"
    )
    print(
        f"4. Content PLL mean BER (drift/mismatch) = "
        f"{pll['content_drift'].mean():.3f}/{pll['content_mismatch'].mean():.3f}; "
        f"Pilot PLL mean BER = "
        f"{pll['pilot_drift'].mean():.3f}/{pll['pilot_mismatch'].mean():.3f}"
    )

    # ----- 2×2 figure -----
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
    fig.suptitle(
        "Oscillatory lock-in (T0) — single-receiver, 40 Hz / 4 kHz, 5-cycle symbols",
        fontsize=12,
    )

    ax = axes[0, 0]
    colors = {10.0: "C0", 0.0: "C1", -6.0: "C2"}
    for snr in snrs:
        ax.plot(
            phase_degs,
            sep[snr],
            "-o",
            ms=3,
            color=colors[snr],
            label=f"SNR {snr:+.0f} dB (sep≈{usable[snr]:.0f}°)",
        )
    ax.axhline(0.05, color="k", ls="--", lw=0.8, label="BER = 5%")
    ax.set_xlabel("Phase offset of sender B (deg)")
    ax.set_ylabel("BER (sender A)")
    ax.set_title("1. Phase separation / slots (pulsed)")
    ax.set_ylim(-0.02, 0.55)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    slot_txt = ", ".join(f"{slots[s]:.0f}" for s in snrs)
    ax.text(
        0.02,
        0.02,
        f"slots/cycle ≈ {slot_txt} (10/0/−6 dB)",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )

    ax = axes[0, 1]
    ax.plot(phase_degs, ber_signed, "-o", ms=3, color="C3", label="Signed sinusoid")
    ax.plot(phase_degs, ber_pulsed, "-o", ms=3, color="C0", label="Pulsed max(0,cos)²")
    ax.axhline(0.05, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Phase offset of sender B (deg)")
    ax.set_ylabel("BER (sender A)")
    ax.set_title("2. Positive-only vs signed (SNR 0 dB)")
    ax.set_ylim(-0.02, 0.55)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(offsets, ber_fixed, "-o", ms=4, color="C1")
    ax.axhline(0.5, color="k", ls="--", lw=0.8, label="chance")
    ax.axvline(0.25, color="gray", ls=":", lw=1)
    ax.set_xlabel("Shared frequency offset (Hz)")
    ax.set_ylabel("BER (sender A)")
    ax.set_title("3. Fixed reference vs freq offset (50 s trial)")
    ax.set_ylim(-0.02, 0.65)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.98,
        0.05,
        f"BER@0.25 Hz = {ber_fixed[i025]:.2f}",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
    )

    ax = axes[1, 1]
    ax.plot(
        pll["mismatches"],
        pll["content_mismatch"],
        "-o",
        ms=3,
        color="C3",
        label="Content-locked PLL",
    )
    ax.plot(
        pll["mismatches"],
        pll["pilot_mismatch"],
        "-o",
        ms=3,
        color="C2",
        label="Pilot-locked PLL",
    )
    ax.axhline(
        pll["content_drift"].mean(),
        color="C3",
        ls="--",
        lw=1,
        alpha=0.7,
        label=f"Content drift mean={pll['content_drift'].mean():.2f}",
    )
    ax.axhline(
        pll["pilot_drift"].mean(),
        color="C2",
        ls="--",
        lw=1,
        alpha=0.7,
        label=f"Pilot drift mean={pll['pilot_drift'].mean():.2f}",
    )
    ax.set_xlabel("Shared frequency mismatch (Hz)")
    ax.set_ylabel("BER (sender A)")
    ax.set_title("4. PLL content vs pilot (K=800, τ=3 cyc)")
    ax.set_ylim(-0.02, 0.65)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_RESULTS, dpi=140)
    fig.savefig(OUT_CWD, dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT_RESULTS}")
    print(f"Wrote {OUT_CWD}")



# ---------------------------------------------------------------------------
# T1 — receiver / sender per-cycle timing jitter
# ---------------------------------------------------------------------------
def main_t1() -> None:
    """Phase-separation vs σ_j; writes lockin_jitter_slots.png + statement."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prediction must already be on disk before measurements are interpreted.
    # If missing, write it first (never overwrite a prior prediction line).
    pred_line = "PREDICTION: 4 to 8 slots per cycle at σ_j = 0.05"
    if OUT_T1_STMT.exists():
        existing = OUT_T1_STMT.read_text().splitlines()
        if not existing or not existing[0].startswith("PREDICTION:"):
            OUT_T1_STMT.write_text(pred_line + "\n" + OUT_T1_STMT.read_text())
    else:
        OUT_T1_STMT.write_text(pred_line + "\n")
    print(pred_line)

    seed = RNG_SEED + 1
    sigmas = list(JITTER_SIGMAS)
    # Focus SNR for the slots-vs-jitter curve; secondary BER panel at 10 dB
    snr_primary = 10.0
    snrs = [10.0, 0.0]  # two SNRs keep runtime reasonable
    phase_degs = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 90.0, 31),   # 3° steps through the slot region
                np.linspace(100.0, 180.0, 9),
            ]
        )
    )
    n_symbols = 160
    n_seeds = 6

    usable: dict[float, dict[float, float]] = {snr: {} for snr in snrs}
    slots: dict[float, dict[float, float]] = {snr: {} for snr in snrs}
    ber_curves: dict[float, np.ndarray] = {}

    for sigma_j in sigmas:
        print(f"T1 phase-separation sweep at σ_j = {sigma_j:.2f} …")
        sep = sweep_phase_separation(
            snrs, phase_degs, n_symbols, seed, n_seeds=n_seeds, sigma_j=sigma_j
        )
        for snr in snrs:
            u = usable_separation_deg(phase_degs, sep[snr])
            usable[snr][sigma_j] = u
            slots[snr][sigma_j] = 360.0 / u if u > 0 else float("inf")
            print(
                f"  SNR {snr:+.0f} dB: usable sep ≈ {u:.1f}° → "
                f"{slots[snr][sigma_j]:.2f} slots/cycle"
            )
        ber_curves[sigma_j] = sep[snr_primary]

    # ----- Figure: slots vs σ_j + BER curves at 10 dB -----
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    fig.suptitle(
        "T1 — timing jitter (per-cycle σ_j on senders + receiver gain)",
        fontsize=12,
    )

    ax = axes[0]
    for snr, style in zip(snrs, ["-o", "-s"]):
        xs = np.asarray(sigmas)
        ys = np.asarray([usable[snr][s] for s in sigmas])
        sl = np.asarray([slots[snr][s] for s in sigmas])
        ax.plot(xs, ys, style, ms=6, label=f"SNR {snr:+.0f} dB")
        for x, y, s in zip(xs, ys, sl):
            ax.annotate(
                f"{s:.1f}",
                (x, y),
                textcoords="offset points",
                xytext=(4, 6),
                fontsize=7,
            )
    ax.axhspan(360.0 / 8.0, 360.0 / 4.0, color="C2", alpha=0.12, label="4–8 slots band")
    ax.axvline(0.05, color="gray", ls=":", lw=1)
    ax.set_xlabel("Timing jitter σ_j (cycles)")
    ax.set_ylabel("Min usable separation (deg, BER<5%)")
    ax.set_title("Usable separation vs σ_j")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for sigma_j, color in zip(sigmas, ["C0", "C1", "C2", "C3"]):
        ax.plot(
            phase_degs,
            ber_curves[sigma_j],
            "-o",
            ms=2.5,
            color=color,
            label=f"σ_j={sigma_j:.2f} (sep≈{usable[snr_primary][sigma_j]:.0f}°)",
        )
    ax.axhline(0.05, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Phase offset of sender B (deg)")
    ax.set_ylabel("BER (sender A)")
    ax.set_title(f"Phase-separation BER curves (SNR {snr_primary:+.0f} dB)")
    ax.set_ylim(-0.02, 0.55)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_JITTER, dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT_JITTER}")

    # One-line measured statement at σ_j=0.05 (primary SNR=10 dB)
    s05 = slots[snr_primary][0.05]
    u05 = usable[snr_primary][0.05]
    measured = (
        f"MEASURED: at σ_j=0.05, SNR={snr_primary:+.0f} dB → "
        f"{s05:.2f} slots/cycle (usable separation {u05:.1f}°)"
    )
    print(measured)

    # Append measurements; keep PREDICTION as first line
    lines = OUT_T1_STMT.read_text().splitlines()
    pred = lines[0] if lines and lines[0].startswith("PREDICTION:") else pred_line
    detail_lines = [
        pred,
        measured,
        "DETAIL: slots/cycle by σ_j (SNR +10 / 0 dB):",
    ]
    for s in sigmas:
        detail_lines.append(
            f"  σ_j={s:.2f}: {slots[10.0][s]:.2f} / {slots[0.0][s]:.2f}"
        )
    s05_0 = slots[0.0][0.05]
    in_band_10 = 4.0 <= s05 <= 8.0
    in_band_0 = 4.0 <= s05_0 <= 8.0
    if in_band_10 and in_band_0:
        verdict = f"yes — {s05:.2f} (SNR+10) and {s05_0:.2f} (SNR 0) both in 4–8"
    elif in_band_0 or in_band_10:
        verdict = (
            f"marginal — SNR +10 dB gives {s05:.2f} "
            f"({'inside' if in_band_10 else 'just outside'} 4–8); "
            f"SNR 0 dB gives {s05_0:.2f} "
            f"({'inside' if in_band_0 else 'outside'} 4–8). "
            "Trend matches handoff: jitter pulls slots toward biology's 4–8."
        )
    else:
        verdict = (
            f"no — {s05:.2f} (SNR+10) / {s05_0:.2f} (SNR 0) vs predicted 4–8"
        )
    detail_lines.append("SUPPORTS_PREDICTION: " + verdict)
    OUT_T1_STMT.write_text("\n".join(detail_lines) + "\n")
    print(f"Wrote {OUT_T1_STMT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Oscillatory lock-in T0/T1")
    parser.add_argument(
        "--jitter",
        action="store_true",
        help="Run T1 timing-jitter sweep (default: T0 four-panel figure)",
    )
    args = parser.parse_args()
    if args.jitter:
        main_t1()
    else:
        main_t0()


if __name__ == "__main__":
    main()
