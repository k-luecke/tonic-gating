"""Unit tests for stencil.py (T3)."""

from __future__ import annotations

import numpy as np

from content import stream_phase
from lockin import DT, PILOT_AMP
from sheet import N_PATCHES, N_UNITS, Sheet
from stencil import StencilController


def test_exactly_one_patch_open():
    st = StencilController(open_patch=2)
    st.control_for_target(5)
    assert st.I.sum() == N_PATCHES - 1
    assert st.I[2] == 0.0
    assert np.all(st.I[np.arange(N_PATCHES) != 2] == 1.0)


def test_pilot_phases_open_vs_closed():
    st = StencilController(open_patch=1, sigma_c=0.0)
    d_star = 3
    st.control_for_target(d_star)
    open_psi = st.psi[1]
    for k in range(N_PATCHES):
        rel = (st.psi[k] - open_psi + np.pi) % (2 * np.pi) - np.pi
        if k == 1:
            assert abs(rel) < 1e-9
        else:
            assert abs(abs(rel) - np.pi) < 1e-9
    ctrl = st.current_control()
    assert ctrl.pilot_signal is not None
    assert np.max(np.abs(ctrl.pilot_signal)) <= PILOT_AMP + 1e-12


def test_open_patch_locks_within_20deg_under_shared_drift():
    """
    With pilot present and shared drift σ_c=0.05, open-patch units stay
    within ±20° of the target stream phase.
    """
    d_star = 2
    k_open = 0
    sigma_c = 0.05
    sheet = Sheet(sigma_f=0.02, seed=99)
    stencil = StencilController(
        open_patch=k_open, sigma_c=sigma_c, seed=123
    )
    stencil.control_for_target(d_star)

    theta0 = stream_phase(d_star)
    sheet.phi[:] = theta0 + sheet.rng.normal(0.0, 0.1, size=N_UNITS)
    # Seed natural freqs near the shared carrier so PLL stays in lock
    sheet.f[:] = stencil.carrier_hz * (
        1.0 + sheet.rng.normal(0.0, 0.02, size=N_UNITS)
    )
    sheet.omega = 2.0 * np.pi * sheet.f

    n_steps = int(round(0.5 / DT))
    settle = int(round(0.2 / DT))
    max_abs_err = 0.0
    for i in range(n_steps):
        ctrl = stencil.step()
        # Pulsed content phase-locked to open pilot (shares drift)
        u = float(np.maximum(0.0, np.cos(stencil.psi[k_open])) ** 2)
        sheet.step(u, ctrl)
        if i >= settle:
            err = sheet.unit_phase_error_to(stencil.psi[k_open], patch=k_open)
            max_abs_err = max(max_abs_err, float(np.max(np.abs(err))))

    lim = np.deg2rad(20.0)
    assert max_abs_err <= lim, (
        f"max |Δφ| = {np.rad2deg(max_abs_err):.1f}° > 20° "
        f"(σ_c={sigma_c}, δ={stencil.shared_delta:.4f})"
    )
