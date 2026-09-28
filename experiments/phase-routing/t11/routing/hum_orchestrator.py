"""Local Hum orchestrator for T11 smoke — HumState + keyword projection drive.

DISCLOSURE
----------
- Uses experiments/phase-routing/t11/hum/hum_state.py (T8 reimpl; acceptance PASS).
- n_units = 3 (one per specialist). Coupling rebuilt for n=3 (same seed/ρ formula).
- Level-2 evidence = fixed keyword projection (HANDOFF §3.3 fallback); no specialist
  affinity API and no GNURadio measure/ckpt wired on this box.
- Tiering draft: warm after k=2 unselected segments, cold after m=4 (not locked in PREREG).
- This is a local dynamics smoke, NOT GNURadio's production hum/measure harness.

GNURadio handoff (required before multi-seed primary):
1. Regime probe (level-1) on segment context — frozen definition + hash.
2. Per-step affinity inputs from hot specialist (preferred over keyword projection).
3. Confirm n_units=3 coupling vs T8's n=8 import policy; hash C.
4. Serialize/load API for amp+phase+tier (hum-state carry) matching their ckpt schema.
5. Optional: Hum+SCP (H5) reference source when typed goal present.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import HumState from sibling package
_HUM_DIR = Path(__file__).resolve().parents[1] / "hum"
if str(_HUM_DIR.parent.parent.parent) not in sys.path:
    # /workspace/scp on path so `experiments...` is not required; import by path
    sys.path.insert(0, str(_HUM_DIR.parent))

from hum.hum_state import (  # type: ignore  # noqa: E402
    A0,
    AMP_GATE_THRESHOLD,
    COUPLING_SEED,
    COUPLING_SPECTRAL_RADIUS,
    LAMBDA_LEAK,
    HumState,
)

from .lexicon import (
    SPECIALISTS,
    first_n_tokens,
    keyword_counts,
    refuse_match,
    tokenize,
)

SPEC_TO_IDX = {"Browser": 0, "Structured": 1, "General": 2}
IDX_TO_SPEC = {v: k for k, v in SPEC_TO_IDX.items()}

CONTEXT_CARRY_N = 32
K_WARM = 2
M_COLD = 4
DRIVE_SCALE = 1.0


@dataclass
class HumOrchestrator:
    name: str = "hum_local"
    status: str = "local_dynamics_smoke"  # disclosed; not GNURadio measure
    disclosure: str = (
        "HumState n=3 + keyword-projection drive on first-32 tokens; "
        "tiering k=2/m=4 draft; no GNURadio ckpt/measure API"
    )
    first_n: int = 32
    context_carry_n: int = CONTEXT_CARRY_N
    k_warm: int = K_WARM
    m_cold: int = M_COLD
    _hum: Optional[HumState] = field(default=None, repr=False)
    _unselected: Dict[str, int] = field(default_factory=dict)
    _tier: Dict[str, str] = field(default_factory=dict)  # hot/warm/cold label
    _prior_tokens: List[str] = field(default_factory=list)
    _carry_mode_active: Optional[str] = None

    def __post_init__(self) -> None:
        self._init_state()

    def _init_state(self) -> None:
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
        self._unselected = {s: 0 for s in SPECIALISTS}
        self._tier = {s: "warm" for s in SPECIALISTS}
        self._prior_tokens = []
        self._carry_mode_active = None

    def reset(self) -> None:
        self._init_state()

    def serialize_hum_state(self) -> Dict[str, Any]:
        assert self._hum is not None
        # amp + phase per specialist + tiering
        s = self._hum.s
        amp = np.abs(s)
        phase = np.angle(s)
        payload = {
            "kind": "hum_state",
            "amp": amp.astype(float).tolist(),
            "phase": phase.astype(float).tolist(),
            "s_real": s.real.astype(float).tolist(),
            "s_imag": s.imag.astype(float).tolist(),
            "tier": dict(self._tier),
            "unselected": dict(self._unselected),
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        payload["bytes"] = len(raw)
        # Also report minimal numeric size: 3 complex128 + 3 tier ints ≈ 3*16+3*4
        payload["bytes_numeric"] = 3 * 16 + 3 * 4
        return payload

    def load_hum_state(self, blob: Optional[Dict[str, Any]]) -> None:
        if not blob or self._hum is None:
            return
        if "s_real" in blob and "s_imag" in blob:
            s = np.array(blob["s_real"], dtype=np.float64) + 1j * np.array(
                blob["s_imag"], dtype=np.float64
            )
            self._hum.reset(s0=s)
        elif "amp" in blob and "phase" in blob:
            amp = np.array(blob["amp"], dtype=np.float64)
            phase = np.array(blob["phase"], dtype=np.float64)
            self._hum.reset(s0=amp * np.exp(1j * phase))
        if "tier" in blob:
            self._tier = dict(blob["tier"])
        if "unselected" in blob:
            self._unselected = {k: int(v) for k, v in blob["unselected"].items()}

    def on_window_break(self, carry_mode: str, prior_text: str = "") -> Dict[str, Any]:
        assert self._hum is not None
        self._carry_mode_active = carry_mode
        if carry_mode == "hum-state":
            ser = self.serialize_hum_state()
            # identity reload (state already live); bytes reported for M6b
            return {
                "carry_mode": carry_mode,
                "bytes_carried": ser["bytes_numeric"],
                "bytes_json": ser["bytes"],
                "tokens_carried": 0,
                "state": {k: ser[k] for k in ("amp", "phase", "tier", "unselected")},
            }
        if carry_mode == "context-carry":
            # cold reservoir; will briefly drive from carried tokens on next decide
            saved_prior = tokenize(prior_text)[-self.context_carry_n :]
            self._hum.reset()
            self._unselected = {s: 0 for s in SPECIALISTS}
            self._tier = {s: "warm" for s in SPECIALISTS}
            self._prior_tokens = saved_prior
            raw = " ".join(saved_prior).encode("utf-8")
            return {
                "carry_mode": carry_mode,
                "bytes_carried": len(raw),
                "tokens_carried": len(saved_prior),
            }
        # cold / classifier-buffer → reset (hum ignores classifier buffer)
        self._hum.reset()
        self._unselected = {s: 0 for s in SPECIALISTS}
        self._tier = {s: "warm" for s in SPECIALISTS}
        self._prior_tokens = []
        return {
            "carry_mode": carry_mode,
            "bytes_carried": 0,
            "tokens_carried": 0,
            "note": "hum_local treats classifier-buffer/cold as cold reset",
        }

    def _drive_from_token(self, tok: str) -> np.ndarray:
        """Keyword projection → complex drive aligned to resting phases."""
        assert self._hum is not None
        counts = keyword_counts([tok])
        d = np.zeros(3, dtype=np.complex128)
        for spec, idx in SPEC_TO_IDX.items():
            if counts[spec] > 0:
                d[idx] = DRIVE_SCALE * counts[spec] * self._hum.rest[idx] / A0
        return d

    def _run_tokens(self, tokens: List[str]) -> Tuple[str, Dict[str, Any]]:
        assert self._hum is not None
        tops: List[int] = []
        for tok in tokens:
            self._hum.step(self._drive_from_token(tok))
            _, top, _ = self._hum.gate()
            tops.append(int(top))
        # Final gate readout
        gains, top, k = self._hum.gate()
        if top < 0:
            # sub-threshold → General floor for smoke (disclosed)
            pick = "General"
            sub = True
        else:
            pick = IDX_TO_SPEC[int(top)]
            sub = False
        return pick, {
            "gains": gains.tolist(),
            "top_idx": int(top),
            "k_active": int(k),
            "subthreshold": sub,
            "amp": self._hum.amplitudes().tolist(),
            "tops_trace": tops[-8:],
        }

    def _update_tier(self, pick: str) -> None:
        for s in SPECIALISTS:
            if s == pick:
                self._unselected[s] = 0
                self._tier[s] = "hot"
            else:
                self._unselected[s] = self._unselected.get(s, 0) + 1
                u = self._unselected[s]
                if u >= self.m_cold:
                    self._tier[s] = "cold"
                elif u >= self.k_warm:
                    self._tier[s] = "warm"
                else:
                    self._tier[s] = "warm"

    def decide(self, text: str, *, post_break: bool = False, observation=None) -> Dict[str, Any]:
        assert self._hum is not None
        base = first_n_tokens(text, self.first_n)

        # Refuse before route
        cue = refuse_match(base)
        if cue is not None:
            return {
                "specialist": None,
                "abstain": True,
                "abstain_reason": cue,
                "window_tokens": base,
                "used_carry": False,
                "tier": dict(self._tier),
            }

        used_carry = False
        tokens = list(base)
        if post_break and self._carry_mode_active == "context-carry" and self._prior_tokens:
            # briefly drive with carried tokens first, then segment tokens
            for tok in self._prior_tokens:
                self._hum.step(self._drive_from_token(tok))
            used_carry = True
        elif post_break and self._carry_mode_active == "hum-state":
            used_carry = True  # state already live

        pick, trace = self._run_tokens(tokens)
        self._update_tier(pick)
        self._prior_tokens = tokenize(text)
        return {
            "specialist": pick,
            "abstain": False,
            "abstain_reason": None,
            "window_tokens": tokens,
            "used_carry": used_carry,
            "tier": dict(self._tier),
            "unselected": dict(self._unselected),
            "hum": trace,
        }
