#!/usr/bin/env python3
"""T12b routing-only arms: hum_measure, stateless_argmax, one_step_memory.

Primary contrast (counted): hum_measure vs one_step_memory (as written).
Labeled secondary (not primary contrast): majority_k3 — stateless majority
over a short window of ScoreAffinity argmax. Without this arm a positive
primary would force T12b.2; with it, a positive is interpretable as
smoothing vs held state.

G9: HumMeasureArmStub is dry-run only. Counted / freeze / primary path must use
HumMeasureArm (frozen HumState + ScoreAffinity + HP-every-drive). arm_factory
refuses stub when mode != "dry-run".

C₃ / hum / Affinity __all__ frozen — read-only import; no retune.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

T12B = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]  # /workspace/scp
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "t11" / "imports"))
sys.path.insert(0, str(ROOT / "experiments" / "phase-routing" / "t11"))

from highpass_drive import HP_PERIOD, roofing_highpass_1d  # noqa: E402
from hum.hum_state import (  # noqa: E402
    A0,
    AMP_GATE_THRESHOLD,
    COUPLING_SEED,
    COUPLING_SPECTRAL_RADIUS,
    LAMBDA_LEAK,
    HumState,
)
from hum_measure.affinity_inputs import ScoreAffinityProvider  # noqa: E402
from hum_measure.regime_probe import SPEC_ORDER  # noqa: E402

PRIMARY_ARMS = ("hum_measure", "stateless_argmax", "one_step_memory")
SECONDARY_ARMS = ("majority_k3",)  # labeled secondary — not primary contrast
ARMS = PRIMARY_ARMS + SECONDARY_ARMS
MAJORITY_K = 3
DEFAULT_SPEC: Optional[str] = None
IDX_TO_SPEC = {i: s for i, s in enumerate(SPEC_ORDER)}


def _argmax_spec(vec: Sequence[float]) -> Optional[str]:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size))
    if float(np.abs(arr[:3]).sum()) <= 0.0:
        return None
    return SPEC_ORDER[int(np.argmax(np.abs(arr[:3])))]


class StatelessArgmaxArm:
    name = "stateless_argmax"

    def reset(self) -> None:
        return None

    def decide(self, affinity: Sequence[float], **_: Any) -> Optional[str]:
        return _argmax_spec(affinity)


class OneStepMemoryArm:
    name = "one_step_memory"

    def __init__(self) -> None:
        self._last: Optional[str] = None

    def reset(self) -> None:
        self._last = None

    def decide(self, affinity: Sequence[float], **_: Any) -> Optional[str]:
        inst = _argmax_spec(affinity)
        if inst is not None:
            self._last = inst
            return inst
        return self._last if self._last is not None else DEFAULT_SPEC


class MajorityK3Arm:
    """Labeled secondary: stateless majority over last k ScoreAffinity argmax.

    Not held integrator state — a sliding-window vote over instantaneous
    argmax labels. Primary contrast remains hum vs one_step_memory.
    Report-only / secondary; does not change H null (hum vs one_step).
    """

    name = "majority_k3"
    counted = False  # secondary — not part of primary contrast
    labeled_secondary = True
    k = MAJORITY_K

    def __init__(self, k: int = MAJORITY_K) -> None:
        self.k = int(k)
        self._window: List[Optional[str]] = []

    def reset(self) -> None:
        self._window = []

    def decide(self, affinity: Sequence[float], **_: Any) -> Optional[str]:
        inst = _argmax_spec(affinity)
        # Only push non-null instantaneous labels into the window (zeros do not vote)
        if inst is not None:
            self._window.append(inst)
            if len(self._window) > self.k:
                self._window = self._window[-self.k :]
        if not self._window:
            return DEFAULT_SPEC
        # majority over current window; ties → most recent
        counts: Dict[str, int] = {}
        for lab in self._window:
            if lab is None:
                continue
            counts[lab] = counts.get(lab, 0) + 1
        if not counts:
            return DEFAULT_SPEC
        best = max(counts.values())
        winners = [lab for lab, c in counts.items() if c == best]
        if len(winners) == 1:
            return winners[0]
        # tie-break: most recent among winners
        for lab in reversed(self._window):
            if lab in winners:
                return lab
        return winners[0]


class HumMeasureArmStub:
    """Dry-run only stand-in (hold-last on zero). NOT a counted arm (G9)."""

    name = "hum_measure"
    counted = False
    dry_run_only = True

    def __init__(self) -> None:
        self._sap = ScoreAffinityProvider()
        self._held: Optional[str] = None
        self.affinity_label = getattr(self._sap, "label", "score_vector")

    def reset(self) -> None:
        self._held = None

    def decide(self, affinity: Sequence[float], **_: Any) -> Optional[str]:
        arr = np.asarray(affinity, dtype=np.float64).reshape(-1)
        if float(np.abs(arr[:3]).sum()) <= 0.0:
            return self._held
        scores = {s: float(arr[i]) if i < len(arr) else 0.0 for i, s in enumerate(SPEC_ORDER)}
        aff = self._sap.affinities(observation={"scores": scores}, hot_specialist=None, step=0)
        pred = _argmax_spec(np.abs(np.asarray(aff)).reshape(-1)[:3])
        if pred is not None:
            self._held = pred
        return pred


class HumMeasureArm:
    """Counted hum_measure: frozen HumState + ScoreAffinity + HP every drive.

    Imports HumState from experiments/phase-routing/t11/hum (T8 reimpl).
    ScoreAffinityProvider from t11/imports/hum_measure (frozen package).
    Optional: HumMeasureOrchestrator import-checked for G9 path presence.
    Linux: pure numpy dynamics — no Windows-only bits required for step path.
    """

    name = "hum_measure"
    counted = True
    dry_run_only = False

    def __init__(self, *, hp_period: float = HP_PERIOD) -> None:
        self._sap = ScoreAffinityProvider()
        self.affinity_label = getattr(self._sap, "label", "score_vector")
        self.hp_period = float(hp_period)
        self._hum: Optional[HumState] = None
        # causal HP state per drive channel (real/imag)
        self._hp_slow_re = np.zeros(3, dtype=np.float64)
        self._hp_slow_im = np.zeros(3, dtype=np.float64)
        self._a_hp = 2.0 / (self.hp_period + 1.0)
        self._init_hum()
        # G9: confirm orchestrator importable (scaffold reuse; not required per-step)
        self._orchestrator_cls = None
        try:
            from routing.hum_measure_arm import HumMeasureOrchestrator  # type: ignore

            self._orchestrator_cls = HumMeasureOrchestrator
        except Exception as exc:  # noqa: BLE001
            self._orchestrator_import_error = str(exc)
        else:
            self._orchestrator_import_error = None

    def _init_hum(self) -> None:
        self._hum = HumState(
            n=3,
            lambda_leak=LAMBDA_LEAK,
            a0=A0,
            coupling_seed=COUPLING_SEED,
            coupling_radius=COUPLING_SPECTRAL_RADIUS,
            amp_gate_threshold=AMP_GATE_THRESHOLD,
            g_tonic=1.0,
        )
        self._hum.reset()
        self._hp_slow_re[:] = 0.0
        self._hp_slow_im[:] = 0.0

    def reset(self) -> None:
        self._init_hum()

    def _highpass_drive(self, drive: np.ndarray) -> np.ndarray:
        """Roofing HP on every drive channel (real & imag) — causal online."""
        d = np.asarray(drive, dtype=np.complex128).reshape(3)
        out = np.zeros(3, dtype=np.complex128)
        a = self._a_hp
        for e in range(3):
            re = float(d[e].real)
            im = float(d[e].imag)
            self._hp_slow_re[e] = a * re + (1.0 - a) * self._hp_slow_re[e]
            self._hp_slow_im[e] = a * im + (1.0 - a) * self._hp_slow_im[e]
            out[e] = (re - self._hp_slow_re[e]) + 1j * (im - self._hp_slow_im[e])
        return out

    def decide(self, affinity: Sequence[float], **_: Any) -> Optional[str]:
        assert self._hum is not None
        arr = np.asarray(affinity, dtype=np.float64).reshape(-1)
        scores = {s: float(arr[i]) if i < len(arr) else 0.0 for i, s in enumerate(SPEC_ORDER)}
        raw = self._sap.affinities(
            observation={"scores": scores}, hot_specialist=None, step=0
        )
        # HP every drive (including near-zero gap frames — reject one-frame lure)
        drive = self._highpass_drive(raw)
        self._hum.step(drive)
        gains, top, k = self._hum.gate()
        if top < 0:
            return None
        return IDX_TO_SPEC[int(top)]


def arm_factory(mode: str = "dry-run") -> Dict[str, Any]:
    """Build arms. mode='dry-run' may use stub; mode='primary'|'freeze' refuses stub."""
    mode = str(mode)
    if mode in ("primary", "freeze", "counted"):
        hum: Any = HumMeasureArm()
        if getattr(hum, "dry_run_only", False):
            raise RuntimeError("G9: counted path refused HumMeasureArmStub")
        if not getattr(hum, "counted", False):
            raise RuntimeError("G9: hum_measure arm not marked counted")
    elif mode == "dry-run":
        # dry-run default still exposes stub for legacy smoke; prefer real when possible
        try:
            hum = HumMeasureArm()
        except Exception:
            hum = HumMeasureArmStub()
    else:
        raise ValueError(f"unknown arm_factory mode: {mode!r}")

    arms = {
        "hum_measure": hum,
        "stateless_argmax": StatelessArgmaxArm(),
        "one_step_memory": OneStepMemoryArm(),
        "majority_k3": MajorityK3Arm(k=MAJORITY_K),  # labeled secondary
    }
    missing = [a for a in PRIMARY_ARMS if a not in arms]
    if missing:
        raise RuntimeError(f"missing required primary arms: {missing}")
    if "majority_k3" not in arms:
        raise RuntimeError("missing labeled secondary arm majority_k3")
    if mode in ("primary", "freeze", "counted"):
        hm = arms["hum_measure"]
        if isinstance(hm, HumMeasureArmStub) or getattr(hm, "dry_run_only", False):
            raise RuntimeError(
                "G9: freeze/primary must not use HumMeasureArmStub as counted hum_measure"
            )
    return arms


def refuse_stub_as_counted(arm: Any) -> None:
    if isinstance(arm, HumMeasureArmStub) or getattr(arm, "dry_run_only", False):
        raise RuntimeError("G9: HumMeasureArmStub is dry-run only; refused as counted arm")


def smoke_decide(mode: str = "dry-run") -> Dict[str, Any]:
    arms = arm_factory(mode=mode)
    for a in arms.values():
        a.reset()
    live = [1.0, 0.0, 0.0]
    zero = [0.0, 0.0, 0.0]
    out: Dict[str, Any] = {
        "mode": mode,
        "arms": list(arms),
        "primary_arms": list(PRIMARY_ARMS),
        "secondary_arms": list(SECONDARY_ARMS),
        "majority_k3_labeled_secondary": True,
        "hum_class": type(arms["hum_measure"]).__name__,
        "hum_counted": bool(getattr(arms["hum_measure"], "counted", False)),
        "live": {},
        "gap": {},
    }
    if hasattr(arms["hum_measure"], "_orchestrator_cls"):
        out["HumMeasureOrchestrator_importable"] = arms["hum_measure"]._orchestrator_cls is not None
        out["HumMeasureOrchestrator_error"] = getattr(
            arms["hum_measure"], "_orchestrator_import_error", None
        )
    for name, arm in arms.items():
        out["live"][name] = arm.decide(live)
    for name, arm in arms.items():
        out["gap"][name] = arm.decide(zero)
    return out


if __name__ == "__main__":
    print(smoke_decide(mode="dry-run"))
    print(smoke_decide(mode="primary"))
