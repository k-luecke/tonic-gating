"""Classifier stub — DISCLOSED untrained baseline (not a calibrated MLP).

Stub policy
-----------
- Features: HAND_RULE keyword counts on first 32 tokens (same lexicon).
- Decision: argmax counts; ties broken alphabetically (Browser < General < Structured)
  — intentionally *different* from hand-rule order so the stub is not identical.
- Memory buffer (classifier-buffer carry): EMA of count vectors across segments;
  added to the current count vector before argmax.
- NOT trained / calibrated. Replace with hashed MLP before freeze.

GNURadio / future: swap this module for a frozen classifier weights blob +
disjoint calibration set; keep the same Orchestrator interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .lexicon import (
    SPECIALISTS,
    count_vector,
    first_n_tokens,
    keyword_counts,
    refuse_match,
    tokenize,
)

CONTEXT_CARRY_N = 32
EMA_ALPHA = 0.5


def _stub_pick(counts: Dict[str, float]) -> str:
    if all(v == 0 for v in counts.values()):
        return "General"
    best = max(counts.values())
    # Alphabetical tie-break (differs from hand-rule Browser>Structured>General)
    for s in sorted(SPECIALISTS):
        if counts[s] == best:
            return s
    return "General"


@dataclass
class ClassifierStubOrchestrator:
    name: str = "classifier_stub"
    status: str = "stub"  # DISCLOSED
    disclosure: str = (
        "untrained keyword-count argmax on first-32 tokens; "
        "alphabetical tie-break; optional EMA buffer for classifier-buffer carry; "
        "NOT a calibrated context classifier"
    )
    first_n: int = 32
    context_carry_n: int = CONTEXT_CARRY_N
    ema_alpha: float = EMA_ALPHA
    _buffer: Optional[Dict[str, float]] = None
    _prior_tokens: List[str] = field(default_factory=list)
    _carry_mode_active: Optional[str] = None

    def reset(self) -> None:
        self._buffer = None
        self._prior_tokens = []
        self._carry_mode_active = None

    def serialize_buffer(self) -> Dict[str, Any]:
        if self._buffer is None:
            return {"kind": "classifier_buffer", "buffer": None, "bytes": 0}
        # 3 float64s
        return {
            "kind": "classifier_buffer",
            "buffer": dict(self._buffer),
            "bytes": 3 * 8,
        }

    def load_buffer(self, blob: Optional[Dict[str, Any]]) -> None:
        if not blob or not blob.get("buffer"):
            self._buffer = None
            return
        self._buffer = {k: float(v) for k, v in blob["buffer"].items()}

    def on_window_break(self, carry_mode: str, prior_text: str = "") -> Dict[str, Any]:
        self._carry_mode_active = carry_mode
        if carry_mode == "classifier-buffer":
            ser = self.serialize_buffer()
            # buffer already in memory; "reload" is a no-op identity
            return {
                "carry_mode": carry_mode,
                "bytes_carried": ser["bytes"],
                "tokens_carried": 0,
                "buffer": ser["buffer"],
            }
        if carry_mode == "context-carry":
            self._buffer = None  # drop memory; use token paste instead
            toks = tokenize(prior_text)[-self.context_carry_n :]
            self._prior_tokens = toks
            raw = " ".join(toks).encode("utf-8")
            return {
                "carry_mode": carry_mode,
                "bytes_carried": len(raw),
                "tokens_carried": len(toks),
            }
        # cold / hum-state → no classifier state
        self._buffer = None
        self._prior_tokens = []
        return {
            "carry_mode": carry_mode,
            "bytes_carried": 0,
            "tokens_carried": 0,
            "note": "classifier_stub treats hum-state/cold as cold",
        }

    def decide(self, text: str, *, post_break: bool = False, observation=None) -> Dict[str, Any]:
        base = first_n_tokens(text, self.first_n)
        used_carry = False
        if post_break and self._prior_tokens and self._carry_mode_active == "context-carry":
            window = (self._prior_tokens + base)[-self.first_n :]
            used_carry = True
        else:
            window = base

        cue = refuse_match(window)
        if cue is not None:
            return {
                "specialist": None,
                "abstain": True,
                "abstain_reason": cue,
                "counts": keyword_counts(window),
                "window_tokens": window,
                "used_carry": used_carry,
                "buffer": self._buffer,
            }

        counts = {k: float(v) for k, v in keyword_counts(window).items()}
        if (
            post_break
            and self._carry_mode_active == "classifier-buffer"
            and self._buffer is not None
        ):
            for k in counts:
                counts[k] = counts[k] + float(self._buffer.get(k, 0.0))
            used_carry = True

        pick = _stub_pick(counts)

        # Update EMA buffer for future classifier-buffer carry
        cur = {k: float(v) for k, v in keyword_counts(window).items()}
        if self._buffer is None:
            self._buffer = cur
        else:
            a = self.ema_alpha
            self._buffer = {
                k: a * cur[k] + (1.0 - a) * float(self._buffer.get(k, 0.0)) for k in cur
            }

        self._prior_tokens = tokenize(text)
        return {
            "specialist": pick,
            "abstain": False,
            "abstain_reason": None,
            "counts": counts,
            "window_tokens": window,
            "used_carry": used_carry,
            "buffer": dict(self._buffer) if self._buffer else None,
        }
