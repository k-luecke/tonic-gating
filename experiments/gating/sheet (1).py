"""
sheet.py — 16×16 phase-oscillator substrate (T2).

Patch layout choice (recorded here and in README):
  K=8 contiguous patches of 32 units each as **4×8 blocks**
  (4 rows × 8 cols). Tiling on the 16×16 grid:
    patch row p_r ∈ {0,1,2,3}, patch col p_c ∈ {0,1}
    patch_id = p_r * 2 + p_c
    units: rows [4*p_r .. 4*p_r+3], cols [8*p_c .. 8*p_c+7]

Dynamics (per unit i):
  ω_i = 2π f_i,  f_i ~ F0 * (1 + N(0, σ_f))
  g_i = max(0, cos φ_i)
  Kuramoto (4-neighbor **within patch only**, per-edge):
    κ_i = Σ_{j∈N(i), same patch} sin(φ_j − φ_i)
    K_c = 0.1 × 2π × 40 rad/s
    Cross-patch edges omitted so anti-phase closed pilots cannot drag.
  Optional pilot PLL (when control provides pilot_signal), matching lockin.py:
    first-order loop, K_p=800, τ=3 cycles
  Phase update:
    dφ/dt = ω_i + K_c κ_i + K_p e_f   (e_f unused if no pilot)
  Leaky integrator (τ_m = 2 cycles), gated by open-patch mask:
    ds/dt = −s/τ_m + (1−I_{patch(i)}) * g_i * u_i

API: step(input_vector, control) — updates state in place.

gain_mode: "oscillatory" (default, g=max(0, cos φ)) or "fixed" (g=1, no
rhythm) for the learned-router baseline path (T4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lockin import DT, F0, PLL_K, PLL_TAU

# ---------------------------------------------------------------------------
# Geometry / constants
# ---------------------------------------------------------------------------
GRID = 16
N_UNITS = GRID * GRID  # 256
PATCH_H = 4
PATCH_W = 8
N_PATCH_ROWS = GRID // PATCH_H  # 4
N_PATCH_COLS = GRID // PATCH_W  # 2
N_PATCHES = N_PATCH_ROWS * N_PATCH_COLS  # 8
UNITS_PER_PATCH = PATCH_H * PATCH_W  # 32

K_C = 0.1 * 2.0 * np.pi * F0  # rad/s per edge
TAU_M_CYCLES = 2.0
TAU_M = TAU_M_CYCLES / F0  # seconds
DEFAULT_SIGMA_F = 0.05


def build_patch_id_map() -> np.ndarray:
    """(16, 16) int map: unit (r, c) → patch index 0..7."""
    pid = np.empty((GRID, GRID), dtype=np.int32)
    for pr in range(N_PATCH_ROWS):
        for pc in range(N_PATCH_COLS):
            k = pr * N_PATCH_COLS + pc
            r0, c0 = pr * PATCH_H, pc * PATCH_W
            pid[r0 : r0 + PATCH_H, c0 : c0 + PATCH_W] = k
    return pid


PATCH_ID_2D = build_patch_id_map()
PATCH_ID = PATCH_ID_2D.ravel().astype(np.int32)

PATCH_UNITS: list[np.ndarray] = [
    np.flatnonzero(PATCH_ID == k).astype(np.int32) for k in range(N_PATCHES)
]


@dataclass
class Control:
    """
    Per-step control from the stencil.

    inhibition: (K,) in {0, 1} — 1 = closed/inhibited, 0 = open
    pilot_signal: (K,) instantaneous pilot amp*cos(ψ_k), or None to skip PLL
    """

    inhibition: np.ndarray
    pilot_signal: np.ndarray | None = None


class Sheet:
    """16×16 Kuramoto sheet with leaky integrators and optional pilot PLL."""

    def __init__(
        self,
        *,
        sigma_f: float = DEFAULT_SIGMA_F,
        k_c: float = K_C,
        k_p: float = PLL_K,
        tau_pll: float = PLL_TAU,
        tau_m: float = TAU_M,
        dt: float = DT,
        seed: int | None = 0,
        f0: float = F0,
        gain_mode: str = "oscillatory",
    ) -> None:
        if gain_mode not in ("oscillatory", "fixed"):
            raise ValueError(
                f"gain_mode must be 'oscillatory' or 'fixed', got {gain_mode!r}"
            )
        self.gain_mode = gain_mode
        self.sigma_f = float(sigma_f)
        self.k_c = float(k_c)
        self.k_p = float(k_p)
        self.tau_pll = float(tau_pll)
        self.tau_m = float(tau_m)
        self.dt = float(dt)
        self.f0 = float(f0)
        self.rng = np.random.default_rng(seed)

        jitter = self.rng.normal(0.0, self.sigma_f, size=N_UNITS)
        self.f = self.f0 * (1.0 + jitter)
        self.omega = 2.0 * np.pi * self.f

        self.phi = self.rng.uniform(-np.pi, np.pi, size=N_UNITS)
        self.s = np.zeros(N_UNITS, dtype=np.float64)
        self.e_f = np.zeros(N_UNITS, dtype=np.float64)
        self.t = 0.0
        self.n_steps = 0

        self._a_pll = self.dt / self.tau_pll
        self._inv_tau_m = 1.0 / self.tau_m

    # ----- queries -----
    @property
    def gain(self) -> np.ndarray:
        if self.gain_mode == "fixed":
            return np.ones(N_UNITS, dtype=np.float64)
        return np.maximum(0.0, np.cos(self.phi))

    def patch_order_parameter(self) -> np.ndarray:
        """Kuramoto order |⟨e^{iφ}⟩| per patch, shape (K,)."""
        z = np.exp(1j * self.phi)
        R = np.empty(N_PATCHES, dtype=np.float64)
        for k, idx in enumerate(PATCH_UNITS):
            R[k] = np.abs(np.mean(z[idx]))
        return R

    def patch_activity(self) -> np.ndarray:
        """A_k = mean leaky-integrator state in patch k."""
        A = np.empty(N_PATCHES, dtype=np.float64)
        for k, idx in enumerate(PATCH_UNITS):
            A[k] = float(np.mean(self.s[idx]))
        return np.maximum(A, 0.0)

    def selectivity(self, k_star: int) -> float:
        """S = A_{k*} / Σ_k A_k (0 if total ~0)."""
        A = self.patch_activity()
        tot = float(A.sum())
        if tot < 1e-12:
            return 0.0
        return float(A[int(k_star)] / tot)

    def patch_mean_phase(self, k: int) -> float:
        z = np.mean(np.exp(1j * self.phi[PATCH_UNITS[int(k)]]))
        return float(np.angle(z))

    def unit_phase_error_to(
        self, target_phase: float, patch: int | None = None
    ) -> np.ndarray:
        """Wrapped phase error φ − target in (−π, π]."""
        if patch is None:
            err = self.phi - target_phase
        else:
            err = self.phi[PATCH_UNITS[int(patch)]] - target_phase
        return (err + np.pi) % (2.0 * np.pi) - np.pi

    # ----- dynamics -----
    def _kuramoto(self) -> np.ndarray:
        """Within-patch 4-neighbor coupling κ_i = Σ sin(φ_j − φ_i).

        Edges that cross patch boundaries are omitted so closed-patch
        anti-phase pilots cannot drag open-patch units (and so each
        patch can settle to its own local synchrony).
        """
        p = self.phi.reshape(GRID, GRID)
        pid = PATCH_ID_2D
        kappa = np.zeros((GRID, GRID), dtype=np.float64)
        # vertical edges
        same = pid[1:, :] == pid[:-1, :]
        dphi = np.sin(p[1:, :] - p[:-1, :])
        kappa[:-1, :] += dphi * same
        kappa[1:, :] += -dphi * same
        # horizontal edges
        same = pid[:, 1:] == pid[:, :-1]
        dphi = np.sin(p[:, 1:] - p[:, :-1])
        kappa[:, :-1] += dphi * same
        kappa[:, 1:] += -dphi * same
        return kappa.ravel()

    def step(
        self,
        input_vector: np.ndarray | float | None,
        control: Control | dict[str, Any] | None,
    ) -> None:
        """
        Advance one sample of duration `dt`.

        Parameters
        ----------
        input_vector : None | float | (N,)
            External content drive u_i. None → zeros.
        control : None | Control | dict
            None → all patches open, no PLL pilot.
            dict keys: 'inhibition' (K,), optional 'pilot_signal' (K,).
        """
        if input_vector is None:
            u = np.zeros(N_UNITS, dtype=np.float64)
        elif np.isscalar(input_vector):
            u = np.full(N_UNITS, float(input_vector), dtype=np.float64)
        else:
            u = np.asarray(input_vector, dtype=np.float64).reshape(-1)
            if u.size == 1:
                u = np.full(N_UNITS, float(u[0]), dtype=np.float64)
            elif u.size != N_UNITS:
                raise ValueError(f"input_vector length {u.size} != {N_UNITS}")

        if control is None:
            inhib = np.zeros(N_PATCHES, dtype=np.float64)
            pilot_k = None
        elif isinstance(control, Control):
            inhib = np.asarray(control.inhibition, dtype=np.float64).reshape(
                N_PATCHES
            )
            pilot_k = control.pilot_signal
        else:
            inhib = np.asarray(control["inhibition"], dtype=np.float64).reshape(
                N_PATCHES
            )
            pilot_k = control.get("pilot_signal", None)

        open_mask = 1.0 - inhib[PATCH_ID]
        kappa = self._kuramoto()

        if pilot_k is not None:
            p = np.asarray(pilot_k, dtype=np.float64).reshape(N_PATCHES)
            pilot_i = p[PATCH_ID]
            err = -pilot_i * np.sin(self.phi)
            self.e_f += self._a_pll * (err - self.e_f)
            pll_term = self.k_p * self.e_f
        else:
            pll_term = 0.0

        self.phi = self.phi + (self.omega + self.k_c * kappa + pll_term) * self.dt
        # Wrap for numerical hygiene; sin/cos are 2π-periodic
        self.phi = (self.phi + np.pi) % (2.0 * np.pi) - np.pi

        if self.gain_mode == "fixed":
            g = 1.0
        else:
            g = np.maximum(0.0, np.cos(self.phi))
        drive = open_mask * g * u
        self.s = self.s + self.dt * (-self.s * self._inv_tau_m + drive)

        self.t += self.dt
        self.n_steps += 1

    def run(self, n_steps: int, input_fn=None, control_fn=None) -> None:
        for _ in range(int(n_steps)):
            u = None if input_fn is None else input_fn(self)
            c = None if control_fn is None else control_fn(self)
            self.step(u, c)
