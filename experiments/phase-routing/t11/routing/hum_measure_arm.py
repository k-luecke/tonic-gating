"""Real Hum arm wired to GNURadio hum_measure package (T11).

Uses:
  - Level-1 RegimeProbe (probe_segment) on segment open
  - Level-2 AffinityProvider:
      * ScoreAffinityProvider (preferred) when observation carries scores/receipts
        — tracks full-segment evidence (gate-4 path)
      * KeywordAffinityProvider labeled interim baseline otherwise
        — CRITICAL: first_n=32, per_token=True → per-step inside opening 32 only;
          late tokens past 32 are invisible (GNURadio / Kyle flag)
  - Optional H5 typed_goal_to_drive when typed_goal_family present
  - hum_state_schema serialize (schema_version + n)

Does NOT load T8 MoE .pt. Coupling rebuilt n=3 (sha 40e3c0c9…).
Does NOT retune C₃ / hum. ScoreAffinity imported directly from
hum_measure.affinity_inputs (not in package __all__; package frozen).
PREREG stays unfrozen until Kyle greenlight.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HUM_DIR = Path(__file__).resolve().parents[1] / "hum"
if str(_HUM_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_HUM_DIR.parent))

_IMPORTS = Path(__file__).resolve().parents[4] / "t11" / "imports"
if str(_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_IMPORTS))

from hum.hum_state import (  # type: ignore  # noqa: E402
    A0,
    AMP_GATE_THRESHOLD,
    COUPLING_SEED,
    COUPLING_SPECTRAL_RADIUS,
    LAMBDA_LEAK,
    HumState,
)

from hum_measure import (  # type: ignore  # noqa: E402
    KeywordAffinityProvider,
    probe_segment,
    serialize_hum_state,
    load_hum_state_blob,
    typed_goal_to_drive,
)
# Prefer ScoreAffinity on live scores/receipts (not in package __all__; direct import).
from hum_measure.affinity_inputs import ScoreAffinityProvider  # type: ignore  # noqa: E402
from hum_measure.regime_probe import FIRST_N  # type: ignore  # noqa: E402

from .lexicon import SPECIALISTS, first_n_tokens, refuse_match, tokenize

SPEC_TO_IDX = {"Browser": 0, "Structured": 1, "General": 2}
IDX_TO_SPEC = {v: k for k, v in SPEC_TO_IDX.items()}

CONTEXT_CARRY_N = 32
K_WARM = 2
M_COLD = 4

PROBE_DEF_SHA = "c3d70ed6778d6e6076add14144d0195e1f91888a8a50fb6f0b1c1e6657345fed"
COUPLING_N3_SHA = "40e3c0c95849ad67af6738e7842c1a4d24783dafa742ed247d03391db37a1493"
SCHEMA_JSON_SHA = "4b55f40b9c87225926a7160974ad99f27ac5403cb7d99392c43172e87208f814"


@dataclass
class HumMeasureOrchestrator:
    """GNURadio-wired hum arm (AffinityProvider + L1 probe + schema)."""

    name: str = "hum"
    status: str = "gnuradio_hum_measure"
    disclosure: str = (
        "HumState n=3 + AffinityProvider level-2 (KeywordAffinityProvider "
        "labeled fallback) + RegimeProbe level-1 on segment open; "
        "H5 typed_goal when family present; schema_version+n serialize; "
        "no T8 MoE .pt"
    )
    first_n: int = FIRST_N
    context_carry_n: int = CONTEXT_CARRY_N
    k_warm: int = K_WARM
    m_cold: int = M_COLD
    affinity_label: str = "keyword_fallback"
    _hum: Optional[HumState] = field(default=None, repr=False)
    _affinity: Any = field(default=None, repr=False)
    _unselected: Dict[str, int] = field(default_factory=dict)
    _tier: Dict[str, str] = field(default_factory=dict)
    _prior_tokens: List[str] = field(default_factory=list)
    _carry_mode_active: Optional[str] = None
    _hot: Optional[str] = None
    _step: int = 0
    _h5_used: int = 0
    _l1_used: int = 0
    _l2_fallback: int = 0

    def __post_init__(self) -> None:
        # Keyword = labeled interim baseline (opening-32 ceiling). Score = preferred L2.
        self._keyword_affinity = KeywordAffinityProvider(
            first_n=self.first_n, per_token=True
        )
        self._score_affinity = ScoreAffinityProvider()
        self._affinity = self._keyword_affinity
        self.affinity_label = "keyword_fallback"
        self.affinity_disclosure = (
            "KeywordAffinityProvider(first_n=32, per_token=True) is per-step inside "
            "opening 32 only — late tokens past 32 invisible. Prefer ScoreAffinityProvider "
            "on live scores/receipts for full-segment L2 evidence. C₃/hum frozen."
        )
        self._init_state()

    def _observation_has_scores(self, observation: Any) -> bool:
        if not isinstance(observation, dict):
            return False
        if "scores" in observation or "affinities" in observation:
            return True
        return any(k in observation for k in SPECIALISTS)

    def _select_affinity(self, observation: Any) -> None:
        """Wire ScoreAffinity when scores/receipts present; else Keyword interim."""
        if self._observation_has_scores(observation):
            self._affinity = self._score_affinity
            self.affinity_label = getattr(self._score_affinity, "label", "score_vector")
        else:
            self._affinity = self._keyword_affinity
            self.affinity_label = getattr(self._keyword_affinity, "label", "keyword_fallback")

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
        self._hot = None
        self._step = 0
        self._h5_used = 0
        self._l1_used = 0
        self._l2_fallback = 0

    def reset(self) -> None:
        self._init_state()

    def serialize_hum_state(self) -> Dict[str, Any]:
        assert self._hum is not None
        return serialize_hum_state(self._hum.s, self._tier, self._unselected)

    def load_hum_state(self, blob: Optional[Dict[str, Any]]) -> None:
        if not blob or self._hum is None:
            return
        loaded = load_hum_state_blob(blob)
        self._hum.reset(s0=loaded["s"])
        if loaded["tier"]:
            self._tier = dict(loaded["tier"])
        if loaded["unselected"]:
            self._unselected = {k: int(v) for k, v in loaded["unselected"].items()}

    def on_window_break(self, carry_mode: str, prior_text: str = "") -> Dict[str, Any]:
        assert self._hum is not None
        self._carry_mode_active = carry_mode
        if carry_mode == "hum-state":
            ser = self.serialize_hum_state()
            return {
                "carry_mode": carry_mode,
                "bytes_carried": ser["bytes_numeric"],
                "bytes_json": ser["bytes"],
                "tokens_carried": 0,
                "schema_version": ser.get("schema_version"),
                "n": ser.get("n"),
                "state": {
                    k: ser[k]
                    for k in ("amp", "phase", "tier", "unselected", "schema_version", "n")
                    if k in ser
                },
            }
        if carry_mode == "context-carry":
            saved_prior = tokenize(prior_text)[-self.context_carry_n :]
            self._hum.reset()
            self._unselected = {s: 0 for s in SPECIALISTS}
            self._tier = {s: "warm" for s in SPECIALISTS}
            self._prior_tokens = saved_prior
            self._hot = None
            raw = " ".join(saved_prior).encode("utf-8")
            return {
                "carry_mode": carry_mode,
                "bytes_carried": len(raw),
                "tokens_carried": len(saved_prior),
            }
        self._hum.reset()
        self._unselected = {s: 0 for s in SPECIALISTS}
        self._tier = {s: "warm" for s in SPECIALISTS}
        self._prior_tokens = []
        self._hot = None
        return {
            "carry_mode": carry_mode,
            "bytes_carried": 0,
            "tokens_carried": 0,
            "note": "hum treats classifier-buffer/cold as cold reset",
        }

    def _run_tokens_scored(
        self, tokens: List[str], score_obs: Any
    ) -> Tuple[str, Dict[str, Any]]:
        """L2 via ScoreAffinityProvider — same score vector each step across FULL segment.

        Unlike Keyword (opening-32 ceiling), this lets hum integrate evidence for
        every token position while affinities come from live scores/receipts.
        """
        assert self._hum is not None
        tops: List[int] = []
        for _tok in tokens:
            drive = self._affinity.affinities(
                observation=score_obs,
                hot_specialist=self._hot,
                step=self._step,
            )
            self._hum.step(drive)
            self._step += 1
            _, top, _ = self._hum.gate()
            tops.append(int(top))
        gains, top, k = self._hum.gate()
        if top < 0:
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
            "affinity_provider": self.affinity_label,
            "n_tokens_scored": len(tokens),
        }

    def _run_tokens(self, tokens: List[str]) -> Tuple[str, Dict[str, Any]]:

        assert self._hum is not None
        tops: List[int] = []
        for i, tok in enumerate(tokens):
            drive = self._affinity.affinities(
                observation=tok,
                hot_specialist=self._hot,
                step=self._step,
            )
            self._l2_fallback += 1
            self._hum.step(drive)
            self._step += 1
            _, top, _ = self._hum.gate()
            tops.append(int(top))
        gains, top, k = self._hum.gate()
        if top < 0:
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
            "affinity_provider": self.affinity_label,
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
                else:
                    self._tier[s] = "warm"
        self._hot = pick

    def decide(
        self,
        text: str,
        *,
        post_break: bool = False,
        observation: Any = None,
    ) -> Dict[str, Any]:
        assert self._hum is not None
        base = first_n_tokens(text, self.first_n)

        cue = refuse_match(base)
        if cue is not None:
            return {
                "specialist": None,
                "abstain": True,
                "abstain_reason": cue,
                "window_tokens": base,
                "used_carry": False,
                "tier": dict(self._tier),
                "affinity_provider": self.affinity_label,
            }

        used_carry = False
        obs = observation if observation is not None else {"text": text}
        if isinstance(obs, dict) and "text" not in obs:
            obs = {**obs, "text": text}

        self._select_affinity(obs)
        use_scores = self.affinity_label == "score_vector"

        # H5 level-0 when typed_goal_family present (never from oracle)
        h5_drive = typed_goal_to_drive(obs)
        if h5_drive is not None:
            self._hum.step(h5_drive)
            self._h5_used += 1
            h5_applied = True
        else:
            h5_applied = False

        # Level-1 regime probe on segment open
        l1 = probe_segment(text, first_n=self.first_n)
        self._hum.step(l1)
        self._l1_used += 1

        # Score path: full-segment token loop so L2 tracks evidence past opening 32.
        # Keyword path: opening-32 only (labeled interim ceiling).
        tokens = list(tokenize(text)) if use_scores else list(base)
        if post_break and self._carry_mode_active == "context-carry" and self._prior_tokens:
            for tok in self._prior_tokens:
                drive_obs = obs if use_scores else tok
                drive = self._affinity.affinities(
                    observation=drive_obs, hot_specialist=self._hot, step=self._step
                )
                self._hum.step(drive)
                self._step += 1
                self._l2_fallback += 1
            used_carry = True
        elif post_break and self._carry_mode_active == "hum-state":
            used_carry = True

        if use_scores:
            pick, trace = self._run_tokens_scored(tokens, obs)
        else:
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
            "affinity_provider": self.affinity_label,
            "affinity_disclosure": self.affinity_disclosure,
            "h5_applied": h5_applied,
            "l1_probe": True,
            "measure_refs": {
                "probe_definition_sha256": PROBE_DEF_SHA,
                "coupling_n3_sha256": COUPLING_N3_SHA,
                "schema_json_sha256": SCHEMA_JSON_SHA,
            },
        }
