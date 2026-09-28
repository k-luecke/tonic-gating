"""
content.py — Phase-multiplexed positive-volley streams for the sheet.

Eight streams at θ_d = 2π d / 8, pulsed carriers max(0, cos(·))² only
(never signed). Shared with T2/T3 tests and later T5 harness.
"""

from __future__ import annotations

import numpy as np

from lockin import F0, DT

N_STREAMS = 8


def stream_phase(d: int) -> float:
    """Carrier phase offset for stream d ∈ {0..7}."""
    return 2.0 * np.pi * (int(d) % N_STREAMS) / N_STREAMS


def pulsed_volley(t: np.ndarray | float, phase: float, f0: float = F0) -> np.ndarray | float:
    """Positive-only pulsed carrier at phase `phase` (radians)."""
    return np.maximum(0.0, np.cos(2.0 * np.pi * f0 * t + phase)) ** 2


def make_stream_waveform(
    n_samples: int,
    d: int,
    *,
    f0: float = F0,
    dt: float = DT,
    shared_phase: np.ndarray | float = 0.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    Length-n_samples pulsed volley for stream d.

    shared_phase: scalar or length-n array added to the carrier (shared drift).
    """
    t = np.arange(n_samples, dtype=np.float64) * dt
    phase = stream_phase(d) + shared_phase
    return amplitude * pulsed_volley(t, phase, f0=f0)


def multiplex_streams(
    n_samples: int,
    active: dict[int, float] | None = None,
    *,
    f0: float = F0,
    dt: float = DT,
    shared_phase: np.ndarray | float = 0.0,
) -> np.ndarray:
    """
    Sum of pulsed streams. `active` maps stream index → amplitude
    (default: all eight at amplitude 1).
    """
    if active is None:
        active = {d: 1.0 for d in range(N_STREAMS)}
    out = np.zeros(n_samples, dtype=np.float64)
    for d, amp in active.items():
        if amp == 0.0:
            continue
        out += make_stream_waveform(
            n_samples, d, f0=f0, dt=dt, shared_phase=shared_phase, amplitude=amp
        )
    return out


def broadcast_to_units(waveform_t: float, n_units: int) -> np.ndarray:
    """Scalar sample → per-unit input vector."""
    return np.full(n_units, float(waveform_t), dtype=np.float64)
