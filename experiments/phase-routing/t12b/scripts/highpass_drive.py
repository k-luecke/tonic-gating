"""Roofing high-pass for hum drives — matches T8 FilterStack stage-1 only.

Frozen reference: /workspace/phase-routing/t8/t8_routing/filters.py
  roofing: x - slow_EMA, α = 2/(HP_PERIOD+1), HP_PERIOD=64

Pure numpy (no torch). Does NOT retune λ/a0/C or Affinity.
T12b lock: high-pass every hum drive before the integrator.
"""
from __future__ import annotations

import numpy as np

HP_PERIOD = 64  # frozen T8 FilterStack roofing period


def _alpha_from_period(period: float) -> float:
    return 2.0 / (float(period) + 1.0)


def roofing_highpass_1d(series: np.ndarray, hp_period: float = HP_PERIOD) -> np.ndarray:
    """Causal roofing high-pass on a 1-D real series."""
    series = np.asarray(series, dtype=np.float64).reshape(-1)
    a_hp = _alpha_from_period(hp_period)
    slow = 0.0
    out = np.empty_like(series)
    for i, v in enumerate(series):
        slow = a_hp * float(v) + (1.0 - a_hp) * slow
        out[i] = float(v) - slow
    return out


def highpass_drives(drives: np.ndarray, hp_period: float = HP_PERIOD) -> np.ndarray:
    """Apply roofing high-pass to every drive channel.

    drives: complex or real array [T, n] — high-pass real and imag independently
    (or the real part if real). Returns same dtype family as input (complex128
    if complex input).
    """
    drives = np.asarray(drives)
    if drives.ndim != 2:
        raise ValueError(f"drives must be [T,n], got {drives.shape}")
    T, n = drives.shape
    if np.iscomplexobj(drives):
        out = np.zeros((T, n), dtype=np.complex128)
        for e in range(n):
            re = roofing_highpass_1d(np.real(drives[:, e]), hp_period)
            im = roofing_highpass_1d(np.imag(drives[:, e]), hp_period)
            out[:, e] = re + 1j * im
        return out
    out = np.zeros((T, n), dtype=np.float64)
    for e in range(n):
        out[:, e] = roofing_highpass_1d(drives[:, e], hp_period)
    return out
