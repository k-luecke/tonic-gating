"""Gate-input observation sanitizer (T11 side-channel disclosure).

T10 lesson: a known oracle side-channel blocks freeze. Disclosure is for what
you could not have known — planted scoring labels in arm inputs are not
"disclosable"; they must be stripped or the run cannot answer its question.

Harness may keep oracle_specialist / family / notes for *scoring only*.
Orchestrator decide() must receive a sanitized observation.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Mapping, Optional

# Fields arms may see (gate / H5 / affinity inputs).
ALLOWED_GATE_KEYS: FrozenSet[str] = frozenset(
    {
        "text",
        "tokens",
        "window_tokens",
        "typed_goal",
        "typed_goal_family",  # H5 only when harness supplies without reading oracle
        "task_goal",
        "goal",
        "TASK_GOAL",
        "task_family",
        "goal_family",
        "scp_projection",
        "affinities",  # ScoreAffinityProvider preferred path
        "scores",
        # specialist score keys (ScoreAffinityProvider)
        "Browser",
        "Structured",
        "General",
    }
)

# Explicit denylist — scoring / generator metadata that must never reach arms.
DENIED_SCORING_KEYS: FrozenSet[str] = frozenset(
    {
        "oracle_specialist",
        "family",  # content/scoring family label — NOT typed_goal_family
        "notes",
        "browser_episode",
        "distraction_span",
        "seed",
        "sequence_type",
        "window_break",
    }
)


def sanitize_observation(
    observation: Any,
    *,
    text_fallback: Optional[str] = None,
) -> Any:
    """Return arm-safe observation.

    - None / str / list / tuple: returned as-is (str/list are content tokens).
    - dict: keep ALLOWED_GATE_KEYS only; drop DENIED_SCORING_KEYS.
    - Ensures 'text' is present when text_fallback is given.
    """
    if observation is None:
        if text_fallback is not None:
            return {"text": text_fallback}
        return None
    if isinstance(observation, (str, list, tuple)):
        return observation
    if not isinstance(observation, Mapping):
        return observation

    out: Dict[str, Any] = {}
    for k, v in observation.items():
        if k in DENIED_SCORING_KEYS:
            continue
        if k in ALLOWED_GATE_KEYS:
            out[k] = v
        # unknown keys dropped (fail-closed)
    if "text" not in out and text_fallback is not None:
        out["text"] = text_fallback
    return out


def assert_no_scoring_labels(observation: Any, *, where: str = "observation") -> None:
    """Raise if a scoring label key is still present (post-sanitize check)."""
    if not isinstance(observation, Mapping):
        return
    bad = sorted(set(observation.keys()) & DENIED_SCORING_KEYS)
    if bad:
        raise AssertionError(f"{where} still contains scoring keys: {bad}")
