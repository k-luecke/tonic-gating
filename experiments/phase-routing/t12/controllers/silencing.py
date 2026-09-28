"""H4 operational silencing — estimate may compute; zero/hold at controller input.

PREREG §3.4: regime estimate may still be computed for disclosure/receipts,
but is zeroed or held at the controller input — not fed into the budget decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Optional, Sequence

import numpy as np

SilenceMode = Literal["none", "zero", "hold"]


@dataclass
class RegimeEstimate:
    """Disclosure-side estimate (always computable)."""

    soft: np.ndarray  # length n_regimes (easy/hard → 2) or feature vec
    hard_label: str  # "easy" | "hard" | "unknown"
    confidence: float
    features: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "soft": self.soft.tolist(),
            "hard_label": self.hard_label,
            "confidence": float(self.confidence),
            "features": self.features.tolist(),
        }


@dataclass
class ControllerInput:
    """What the budget controller actually sees after silencing."""

    regime_channel: np.ndarray  # zeroed or held or live
    silenced: bool
    mode: SilenceMode
    difficulty_tag: Optional[str]  # may be None under silence
    disclosure_estimate: Optional[RegimeEstimate] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "regime_channel": self.regime_channel.tolist(),
            "silenced": self.silenced,
            "mode": self.mode,
            "difficulty_tag": self.difficulty_tag,
            "disclosure_estimate": None
            if self.disclosure_estimate is None
            else self.disclosure_estimate.as_dict(),
        }


class SilencingGate:
    """Gate between estimate (disclosure) and controller input."""

    def __init__(self, mode: SilenceMode = "zero", n_channel: int = 2):
        self.mode: SilenceMode = mode
        self.n_channel = int(n_channel)
        self._held: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._held = None

    def estimate(
        self,
        *,
        difficulty: Optional[str] = None,
        features: Optional[Sequence[float]] = None,
        soft: Optional[Sequence[float]] = None,
    ) -> RegimeEstimate:
        """Always compute estimate for disclosure (even on H4 silence cells)."""
        if soft is not None:
            s = np.asarray(soft, dtype=np.float64).reshape(-1)
        elif difficulty == "easy":
            s = np.array([1.0, 0.0], dtype=np.float64)
        elif difficulty == "hard":
            s = np.array([0.0, 1.0], dtype=np.float64)
        elif features is not None:
            f = np.asarray(features, dtype=np.float64).reshape(-1)
            # map feature mean → soft easy/hard
            m = float(np.clip(np.mean(f), 0.0, 1.0))
            s = np.array([1.0 - m, m], dtype=np.float64)
        else:
            s = np.array([0.5, 0.5], dtype=np.float64)
        if s.size < self.n_channel:
            s = np.pad(s, (0, self.n_channel - s.size))
        s = s[: self.n_channel]
        s = s / (s.sum() + 1e-12)
        if difficulty in ("easy", "hard"):
            label = difficulty
            conf = float(np.max(s))
        else:
            label = "easy" if s[0] >= s[1] else "hard"
            conf = float(np.max(s))
        feats = s.copy()
        if features is not None:
            feats = np.asarray(features, dtype=np.float64).reshape(-1)
        return RegimeEstimate(soft=s, hard_label=label, confidence=conf, features=feats)

    def gate(
        self,
        estimate: RegimeEstimate,
        *,
        force_mode: Optional[SilenceMode] = None,
    ) -> ControllerInput:
        """Produce controller input; may zero/hold the regime channel."""
        mode: SilenceMode = force_mode or self.mode
        live = estimate.soft.copy()
        if mode == "none":
            return ControllerInput(
                regime_channel=live,
                silenced=False,
                mode=mode,
                difficulty_tag=estimate.hard_label,
                disclosure_estimate=estimate,
            )
        if mode == "zero":
            ch = np.zeros_like(live)
            return ControllerInput(
                regime_channel=ch,
                silenced=True,
                mode=mode,
                difficulty_tag=None,  # not fed to budget
                disclosure_estimate=estimate,
            )
        # hold: freeze first non-silenced channel seen; else zeros
        if self._held is None:
            self._held = live.copy()
        return ControllerInput(
            regime_channel=self._held.copy(),
            silenced=True,
            mode=mode,
            difficulty_tag=None,
            disclosure_estimate=estimate,
        )

    def prepare(
        self,
        *,
        difficulty: Optional[str] = None,
        features: Optional[Sequence[float]] = None,
        soft: Optional[Sequence[float]] = None,
        force_mode: Optional[SilenceMode] = None,
    ) -> ControllerInput:
        est = self.estimate(difficulty=difficulty, features=features, soft=soft)
        return self.gate(est, force_mode=force_mode)


def apply_silence_to_decide_kwargs(
    ctrl_in: ControllerInput,
    base: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build decide() kwargs: under silence, strip live difficulty / live soft."""
    out: Dict[str, Any] = dict(base or {})
    out["controller_input"] = ctrl_in
    out["features"] = ctrl_in.regime_channel.tolist()
    if ctrl_in.silenced:
        # operational cut: do NOT pass live difficulty into budget
        out.pop("difficulty", None)
        out["difficulty"] = None
        out["silenced"] = True
        out["silence_mode"] = ctrl_in.mode
        # disclosure still available separately
        out["disclosure_estimate"] = (
            None
            if ctrl_in.disclosure_estimate is None
            else ctrl_in.disclosure_estimate.as_dict()
        )
    else:
        out["difficulty"] = ctrl_in.difficulty_tag
        out["silenced"] = False
        out["silence_mode"] = "none"
        out["disclosure_estimate"] = (
            None
            if ctrl_in.disclosure_estimate is None
            else ctrl_in.disclosure_estimate.as_dict()
        )
    return out
