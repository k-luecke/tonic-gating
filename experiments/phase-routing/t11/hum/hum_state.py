"""HumState dynamics — reimpl from frozen T8 PREREG §2.5 (sha d975d96c…).

Order per token (PREREG §2.5): leak → idling → coupling → tonic×drive.
Gate gains = |s_e| / sum|s| (amplitude shares only); sub-threshold → index -1.

Ambiguities documented in module docstring / ACCEPTANCE_RESULT.md:
- Exact idle algebra is not written as equations in §2.5; we use additive
  idle after leak: s ← s + idle_rate * a0 * exp(i φ_e), idle_rate=λ, which
  yields resting |s|=a0 and preserves effective linear map (1-λ)(I+C) on the
  homogeneous part when coupling is s ← (I+C)s after idle.
- Coupling matrix: RandomState(seed).randn, zero diagonal, scaled to
  frozen ρ(C); matches reported effective_linear_ρ ≈ 0.98913734.
- T_hold in full T8 harness = top-1 regime persistence after control silence
  @64 (L1/L1′ zeroed, L2/L5 may continue). This module's acceptance probe uses
  excess-amplitude 1/e hold after full drive silence (dynamics-only), which
  is what HUM_PARAMS ties to 1/λ≈39.2 / hold≈41.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Frozen T8 / T3 constants (HUM_PARAMS.json / PREREG §2.5)
LAMBDA_LEAK = 0.02548046875
A0 = 0.1
COUPLING_SEED = 20260921
COUPLING_SPECTRAL_RADIUS = 0.0149999971
G_TONIC_DEFAULT = 1.0
G_TONIC_LOW = 0.5
G_TONIC_HIGH = 2.0
AMP_GATE_THRESHOLD = 0.05
N_UNITS_DEFAULT = 8


def make_coupling(
    n: int = N_UNITS_DEFAULT,
    seed: int = COUPLING_SEED,
    target_radius: float = COUPLING_SPECTRAL_RADIUS,
) -> np.ndarray:
    """Build frozen real coupling matrix C with ρ(C) ≈ target_radius."""
    rs = np.random.RandomState(int(seed))
    M = rs.randn(n, n).astype(np.float64)
    np.fill_diagonal(M, 0.0)
    eigs = np.linalg.eigvals(M)
    rho = float(np.max(np.abs(eigs)))
    if rho == 0.0:
        raise RuntimeError("coupling draw has zero spectral radius")
    return M * (target_radius / rho)


def resting_phases(n: int = N_UNITS_DEFAULT) -> np.ndarray:
    """φ_e = 2π e / n (spacing only; phase is not a veto)."""
    return 2.0 * np.pi * np.arange(n, dtype=np.float64) / float(n)


def resting_targets(n: int = N_UNITS_DEFAULT, a0: float = A0) -> np.ndarray:
    phi = resting_phases(n)
    return a0 * np.exp(1j * phi)


def gate_from_amplitudes(
    amplitudes: np.ndarray,
    threshold: float = AMP_GATE_THRESHOLD,
    g_tonic: float = G_TONIC_DEFAULT,
) -> Tuple[np.ndarray, int, int]:
    """Amplitude-share gate.

    Returns (gains, top_index, k_active).
    Sub-threshold (max amp < threshold) → index -1, gains 0.
    k from g_tonic: ≤low → top-1; ≥high → top-3; else top-2.
    """
    amp = np.asarray(amplitudes, dtype=np.float64).reshape(-1)
    n = amp.shape[0]
    gains = np.zeros(n, dtype=np.float64)
    if float(np.max(amp)) < float(threshold):
        return gains, -1, 0
    total = float(np.sum(amp))
    if total <= 0.0:
        return gains, -1, 0
    shares = amp / total
    if g_tonic <= G_TONIC_LOW:
        k = 1
    elif g_tonic >= G_TONIC_HIGH:
        k = 3
    else:
        k = 2
    k = min(k, n)
    order = np.argsort(-amp)
    top = int(order[0])
    chosen = order[:k]
    raw = shares[chosen]
    raw_sum = float(np.sum(raw))
    if raw_sum > 0:
        gains[chosen] = raw / raw_sum
    return gains, top, k


@dataclass
class HumState:
    """Complex reservoir state s ∈ C^{n} with frozen leak/idle/coupling."""

    n: int = N_UNITS_DEFAULT
    lambda_leak: float = LAMBDA_LEAK
    a0: float = A0
    idle_rate: Optional[float] = None  # defaults to lambda_leak
    coupling_seed: int = COUPLING_SEED
    coupling_radius: float = COUPLING_SPECTRAL_RADIUS
    amp_gate_threshold: float = AMP_GATE_THRESHOLD
    g_tonic: float = G_TONIC_DEFAULT

    def __post_init__(self) -> None:
        if self.idle_rate is None:
            self.idle_rate = float(self.lambda_leak)
        self.C = make_coupling(self.n, self.coupling_seed, self.coupling_radius)
        self.rest = resting_targets(self.n, self.a0)
        self.s = np.zeros(self.n, dtype=np.complex128)
        self._validate_linear_map()

    def _validate_linear_map(self) -> None:
        A = (1.0 - self.lambda_leak) * (np.eye(self.n) + self.C)
        self.effective_linear_rho = float(np.max(np.abs(np.linalg.eigvals(A))))
        self.coupling_rho = float(np.max(np.abs(np.linalg.eigvals(self.C))))

    def reset(self, s0: Optional[np.ndarray] = None) -> None:
        if s0 is None:
            self.s = np.zeros(self.n, dtype=np.complex128)
        else:
            self.s = np.asarray(s0, dtype=np.complex128).reshape(self.n).copy()

    def set_lambda(self, lam: float) -> None:
        """Eval-time λ override (H2 probe); idle_rate tracks λ per PREREG."""
        self.lambda_leak = float(lam)
        self.idle_rate = float(lam)
        self._validate_linear_map()

    def set_g_tonic(self, g: float) -> None:
        self.g_tonic = float(g)

    def step(self, drive: Optional[np.ndarray] = None) -> np.ndarray:
        """One token update. drive is complex length-n (pre-tonic); None → 0.

        1) leak:     s ← (1-λ) s
        2) idling:   s ← s + idle_rate * a0 * e^{i φ_e}
        3) coupling: s ← (I+C) s
        4) drive:    s ← s + g_tonic * drive
        """
        lam = float(self.lambda_leak)
        idle = float(self.idle_rate)
        # 1. leak
        s = (1.0 - lam) * self.s
        # 2. idling (additive resting drive; idle_rate = λ)
        s = s + idle * self.rest
        # 3. coupling
        s = s + self.C @ s  # == (I+C) @ s
        # 4. tonic × drive
        if drive is None:
            d = 0.0
        else:
            d = np.asarray(drive, dtype=np.complex128).reshape(self.n)
        s = s + self.g_tonic * d
        self.s = s
        return self.s

    def amplitudes(self) -> np.ndarray:
        return np.abs(self.s)

    def excess_amplitudes(self) -> np.ndarray:
        """Amplitude above idle floor a0 (elementwise max with 0)."""
        return np.maximum(self.amplitudes() - float(self.a0), 0.0)

    def gate(self) -> Tuple[np.ndarray, int, int]:
        return gate_from_amplitudes(
            self.amplitudes(), self.amp_gate_threshold, self.g_tonic
        )

    def run(
        self,
        drives: np.ndarray,
        reset: bool = True,
    ) -> dict:
        """Run a sequence of drives [T, n] (complex). Returns trajectories."""
        drives = np.asarray(drives, dtype=np.complex128)
        if drives.ndim != 2 or drives.shape[1] != self.n:
            raise ValueError(f"drives must be [T,{self.n}], got {drives.shape}")
        T = drives.shape[0]
        if reset:
            self.reset()
        S = np.zeros((T, self.n), dtype=np.complex128)
        amp = np.zeros((T, self.n), dtype=np.float64)
        top = np.zeros(T, dtype=np.int64)
        for t in range(T):
            self.step(drives[t])
            S[t] = self.s
            amp[t] = self.amplitudes()
            _, top[t], _ = self.gate()
        return {"S": S, "amp": amp, "top": top}


def measure_thold_excess_1e(
    hum: HumState,
    drive_scale: float = 1.0,
    drive_unit: int = 0,
    t_drive: int = 64,
    t_silence: int = 256,
) -> dict:
    """Silence-hold probe (dynamics-only acceptance).

    Drive one unit for t_drive tokens, then zero drive. T_hold = tokens after
    silence onset until that unit's excess amplitude (|s|-a0)+ falls to ≤ 1/e
    of its value at the first silence step's state *before* further decay —
    measured on the post-step amplitude at silence index 0 as baseline, then
    counting subsequent silence steps until excess ≤ baseline/e.

    Also reports tokens until raw amp < amp_gate_threshold (may be inf when
    idle floor a0 > threshold) and top-1 regime loss (may never flip under
    pure silence with identical idle targets).
    """
    n = hum.n
    drives = np.zeros((t_drive + t_silence, n), dtype=np.complex128)
    # Real positive drive on one unit (W_* = I; a0 scales resting, not drive)
    drives[:t_drive, drive_unit] = float(drive_scale)
    out = hum.run(drives, reset=True)
    amp = out["amp"]
    top = out["top"]
    # Regime mode over drive window
    drive_tops = top[:t_drive]
    # mode
    vals, counts = np.unique(drive_tops[drive_tops >= 0], return_counts=True)
    if len(vals) == 0:
        regime = drive_unit
    else:
        regime = int(vals[np.argmax(counts)])

    # Excess at silence onset = amplitude after last drive step
    amp_at_silence = amp[t_drive - 1, drive_unit]
    excess0 = max(amp_at_silence - hum.a0, 0.0)
    target = excess0 / np.e

    thold_1e = None
    thold_thresh = None
    thold_regime = None
    for k in range(t_silence):
        t = t_drive + k
        a = amp[t, drive_unit]
        ex = max(a - hum.a0, 0.0)
        if thold_1e is None and ex <= target:
            thold_1e = k + 1  # tokens of silence until condition met
        if thold_thresh is None and a < hum.amp_gate_threshold:
            thold_thresh = k + 1
        if thold_regime is None and int(top[t]) != regime:
            thold_regime = k + 1

    return {
        "T_hold": float(thold_1e) if thold_1e is not None else float("nan"),
        "T_hold_1e_excess": thold_1e,
        "T_hold_amp_threshold": thold_thresh,
        "T_hold_regime_top": thold_regime,
        "excess0": excess0,
        "amp_at_silence": float(amp_at_silence),
        "regime_top": regime,
        "lambda": float(hum.lambda_leak),
        "inv_lambda": float(1.0 / hum.lambda_leak) if hum.lambda_leak else float("nan"),
        "effective_linear_rho": float(hum.effective_linear_rho),
        "coupling_rho": float(hum.coupling_rho),
        "amp_trace_silence": amp[t_drive : t_drive + min(t_silence, 80), drive_unit].tolist(),
    }
