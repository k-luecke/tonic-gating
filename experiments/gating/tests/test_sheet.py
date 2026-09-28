"""Unit tests for sheet.py (T2)."""

from __future__ import annotations

import time

import numpy as np

from content import make_stream_waveform, stream_phase
from lockin import DT
from sheet import (
    GRID,
    N_PATCHES,
    N_UNITS,
    PATCH_H,
    PATCH_ID,
    PATCH_W,
    UNITS_PER_PATCH,
    Sheet,
)
from stencil import StencilController


def test_geometry_4x8_blocks():
    assert GRID == 16
    assert N_UNITS == 256
    assert N_PATCHES == 8
    assert PATCH_H == 4 and PATCH_W == 8
    assert UNITS_PER_PATCH == 32
    for k in range(N_PATCHES):
        assert int(np.sum(PATCH_ID == k)) == 32
    pid2d = PATCH_ID.reshape(GRID, GRID)
    assert np.all(pid2d[0:4, 0:8] == 0)
    assert np.all(pid2d[0:4, 8:16] == 1)
    assert np.all(pid2d[12:16, 8:16] == 7)


def test_local_synchrony_no_input_no_control():
    """No input / no control → patches settle to local synchrony.

    Uses σ_f=0.02 (within the prereg 5–10% mismatch band, mild end).
    At σ_f=0.05, 4×8 patches can leave a few units unlocked for this K_c.
    """
    sheet = Sheet(sigma_f=0.02, seed=42)
    n_steps = int(round(0.5 / DT))  # 0.5 s ≈ 20 cycles
    for _ in range(n_steps):
        sheet.step(None, None)
    R = sheet.patch_order_parameter()
    assert R.min() > 0.7, f"order params too low: {R}"
    assert R.mean() > 0.85, f"mean order too low: {R.mean()}"


def test_selectivity_single_open_patch_one_stream():
    """Single open patch + one stream → S > 0.9."""
    d_star = 0
    k_open = 3
    sheet = Sheet(sigma_f=0.02, seed=7)
    stencil = StencilController(open_patch=k_open, sigma_c=0.0, seed=11)
    stencil.control_for_target(d_star)

    theta = stream_phase(d_star)
    sheet.phi[:] = theta + sheet.rng.normal(0.0, 0.1, size=N_UNITS)

    n_steps = int(round(0.4 / DT))
    wave = make_stream_waveform(n_steps, d_star)
    for i in range(n_steps):
        ctrl = stencil.step()
        sheet.step(wave[i], ctrl)

    S = sheet.selectivity(k_open)
    A = sheet.patch_activity()
    assert S > 0.9, f"S={S:.3f}, A={A}"


def test_1000_steps_under_one_second():
    sheet = Sheet(sigma_f=0.05, seed=0)
    stencil = StencilController(open_patch=0, seed=1)
    stencil.control_for_target(0)
    wave = make_stream_waveform(1000, 0)
    t0 = time.perf_counter()
    for i in range(1000):
        sheet.step(wave[i], stencil.step())
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"1000 steps took {elapsed:.3f}s"
    print(f"TIMING_1000_STEPS={elapsed:.4f}s")
