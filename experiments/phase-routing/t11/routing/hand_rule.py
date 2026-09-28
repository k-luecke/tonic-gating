"""Hand-rule orchestrator — Kyle-signed HAND_RULE.md (full implementation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .lexicon import (
    first_n_tokens,
    hand_rule_pick,
    keyword_counts,
    refuse_match,
    tokenize,
)

CONTEXT_CARRY_N = 32  # draft; locked at smoke


@dataclass
class HandRuleOrchestrator:
    """Stateless keyword rule; context-carry only injects prior tokens into window."""

    name: str = "hand_rule"
    status: str = "full"  # not a stub
    first_n: int = 32
    context_carry_n: int = CONTEXT_CARRY_N
    _prior_tokens: List[str] = field(default_factory=list)

    def reset(self) -> None:
        self._prior_tokens = []

    def serialize(self) -> Dict[str, Any]:
        # Hand rule has no orchestrator state worth carrying except for context-carry
        # which is handled by the harness via prior tokens.
        return {"kind": "hand_rule", "bytes": 0}

    def load(self, blob: Optional[Dict[str, Any]]) -> None:
        return

    def on_window_break(self, carry_mode: str, prior_text: str = "") -> Dict[str, Any]:
        """Apply harness carry mode. Returns M6b-ish byte accounting."""
        if carry_mode == "context-carry":
            toks = tokenize(prior_text)[-self.context_carry_n :]
            self._prior_tokens = toks
            raw = " ".join(toks).encode("utf-8")
            return {"carry_mode": carry_mode, "bytes_carried": len(raw), "tokens_carried": len(toks)}
        # hum-state / classifier-buffer / cold → hand has nothing to restore
        self._prior_tokens = []
        return {
            "carry_mode": carry_mode,
            "bytes_carried": 0,
            "tokens_carried": 0,
            "note": "hand_rule ignores non-context carry (acts as cold)",
        }

    def decide(self, text: str, *, post_break: bool = False, observation=None) -> Dict[str, Any]:
        base = first_n_tokens(text, self.first_n)
        if post_break and self._prior_tokens:
            # Decision window = last N carried + first tokens of new segment (cap 32 for rule)
            window = (self._prior_tokens + base)[-self.first_n :]
            used_carry = True
        else:
            window = base
            used_carry = False

        cue = refuse_match(window)
        if cue is not None:
            return {
                "specialist": None,
                "abstain": True,
                "abstain_reason": cue,
                "counts": keyword_counts(window),
                "window_tokens": window,
                "used_carry": used_carry,
            }
        counts = keyword_counts(window)
        pick = hand_rule_pick(counts)
        # After decide on a normal segment, remember tokens for possible next break
        self._prior_tokens = tokenize(text)
        return {
            "specialist": pick,
            "abstain": False,
            "abstain_reason": None,
            "counts": counts,
            "window_tokens": window,
            "used_carry": used_carry,
        }
