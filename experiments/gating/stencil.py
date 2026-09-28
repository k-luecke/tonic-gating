"""
stencil.py — Top-down stencil controller (T3).

Provides an inhibition map I_k ∈ {0,1} and per-patch pilot phase ψ_k.
In the base task exactly one patch is open (I=0); closed patches have I=1.

Pilot: 40 Hz, amplitude 0.5 (PILOT_AMP). For target stream d*:
  open patch  ψ = θ_{d*}
  closed      ψ = θ_{d*} + π

Units PLL-lock to the pilot inside Sheet (K_p=800, τ=3 cycles — lockin.py).

Shared drift σ_c: once-drawn fractional frequency offset δ ~ N(0, σ_c)
applied identically to all pilots (f = F0 (1+δ)). Harness should add the
same δ (via `shared_freq_offset` / `shared_phase` rate) to content streams.

API:
  control_for_target(d_star) → configures open patch + phase offsets
  step() → Control for Sheet.step
"""

from __future__ import annotations

import numpy as np

from content import N_STREAMS, stream_phase
from lockin import DT, F0, PILOT_AMP
from sheet import Control, N_PATCHES


class StencilController:
    """
    Slow stencil: inhibition bitmask + coherent pilots per patch.

    Parameters
    ----------
    open_patch : int
        Which patch is open in the base task (default 0).
    pilot_amp : float
        Pilot cosine amplitude (default 0.5).
    sigma_c : float
        Std of a **shared** fractional frequency offset drawn once:
        δ ~ N(0, σ_c), carrier rate = 2π F0 (1+δ) for every patch pilot.
    """

    def __init__(
        self,
        *,
        open_patch: int = 0,
        n_patches: int = N_PATCHES,
        f0: float = F0,
        pilot_amp: float = PILOT_AMP,
        dt: float = DT,
        sigma_c: float = 0.0,
        seed: int | None = 1,
        shared_delta: float | None = None,
    ) -> None:
        if not 0 <= open_patch < n_patches:
            raise ValueError(f"open_patch {open_patch} out of range")
        self.n_patches = int(n_patches)
        self.open_patch = int(open_patch)
        self.f0 = float(f0)
        self.pilot_amp = float(pilot_amp)
        self.dt = float(dt)
        self.sigma_c = float(sigma_c)
        self.rng = np.random.default_rng(seed)

        if shared_delta is None:
            self.shared_delta = (
                float(self.rng.normal(0.0, self.sigma_c)) if self.sigma_c > 0 else 0.0
            )
        else:
            self.shared_delta = float(shared_delta)

        self.I = np.ones(self.n_patches, dtype=np.float64)
        self.I[self.open_patch] = 0.0

        self._d_star: int = 0
        self._phase_offset = np.zeros(self.n_patches, dtype=np.float64)
        self.psi = np.zeros(self.n_patches, dtype=np.float64)
        self.t = 0.0
        self._shared_phase = 0.0  # ∫ 2π F0 δ dt

        self.control_for_target(0)

    @property
    def carrier_hz(self) -> float:
        return self.f0 * (1.0 + self.shared_delta)

    def control_for_target(
        self, d_star: int, open_patch: int | None = None
    ) -> "StencilController":
        """
        Configure for target stream d* ∈ {0..7}.

        Open patch pilot phase offset = θ_{d*};
        closed patches = θ_{d*} + π.
        Returns self for chaining.
        """
        d_star = int(d_star) % N_STREAMS
        self._d_star = d_star
        if open_patch is not None:
            if not 0 <= open_patch < self.n_patches:
                raise ValueError(open_patch)
            self.open_patch = int(open_patch)
        self.I = np.ones(self.n_patches, dtype=np.float64)
        self.I[self.open_patch] = 0.0

        theta = stream_phase(d_star)
        self._phase_offset = np.full(
            self.n_patches, theta + np.pi, dtype=np.float64
        )
        self._phase_offset[self.open_patch] = theta

        carrier = 2.0 * np.pi * self.carrier_hz * self.t
        self.psi = carrier + self._phase_offset
        self._shared_phase = 2.0 * np.pi * self.f0 * self.shared_delta * self.t
        return self

    @property
    def d_star(self) -> int:
        return self._d_star

    @property
    def shared_phase(self) -> float:
        """Cumulative shared phase from frequency offset (radians)."""
        return self._shared_phase

    def step(self) -> Control:
        """Advance pilots by one sample; return Control for Sheet.step."""
        dpsi = 2.0 * np.pi * self.carrier_hz * self.dt
        self._shared_phase += 2.0 * np.pi * self.f0 * self.shared_delta * self.dt
        self.psi = self.psi + dpsi
        self.t += self.dt
        return self.current_control()

    def current_control(self) -> Control:
        pilot = self.pilot_amp * np.cos(self.psi)
        return Control(inhibition=self.I.copy(), pilot_signal=pilot)

    def reset_time(self) -> None:
        self.t = 0.0
        self._shared_phase = 0.0
        self.psi = self._phase_offset.copy()
