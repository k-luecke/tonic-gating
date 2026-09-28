"""Shared HAND_RULE lexicon + refuse list (Kyle-signed 2026-09-23).

Canonical text: t11/HAND_RULE.md
sha256: 28a3d07295ddf6558261b2f8c5065d6cf4934bc2c8457ef45cbbf081bc281334
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

SPECIALISTS = ("Browser", "Structured", "General")

# Order = hand-rule tie-break order (HANDOFF / HAND_RULE)
BROWSER_WORDS = [
    "click", "select", "choose", "enter", "type", "fill", "submit", "login",
    "search", "scroll", "tab", "dropdown", "checkbox", "button", "link",
    "form", "page", "navigate", "open",
]
STRUCTURED_WORDS = [
    "json", "csv", "parse", "transform", "extract", "field", "column", "row",
    "key", "value", "schema", "table", "convert", "dictionary", "array",
]
STRUCTURED_PHRASES = ["list of", "format as", "sort by"]
GENERAL_WORDS = [
    "answer", "explain", "who", "when", "where", "why", "define", "summarize",
    "question",
]
GENERAL_PHRASES = ["what is"]

REFUSE_WORDS = [
    "pay", "purchase", "buy", "checkout", "card", "password",
    "token",  # as a secret (HAND_RULE); smoke treats whole-word)
    "delete", "wipe", "rm",
    "email", "send", "post", "upload", "download", "install",
]
REFUSE_PHRASES = [
    "api key", "remove all", "drop table", "format disk",
    "ignore prior", "ignore the gate",
]

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:'[A-Za-z]+)?")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def first_n_tokens(text: str, n: int = 32) -> List[str]:
    return tokenize(text)[:n]


def join_tokens(tokens: List[str]) -> str:
    return " ".join(tokens)


def refuse_match(tokens: List[str]) -> Optional[str]:
    """Refuse checks run before specialist rule. Returns cue or None."""
    window = join_tokens(tokens)
    for phrase in REFUSE_PHRASES:
        if phrase in window:
            return phrase
    # multi-word destructive / credential phrases already covered
    # also catch "api key" style via words
    for i, tok in enumerate(tokens):
        if tok in REFUSE_WORDS:
            return tok
        if tok == "drop" and i + 1 < len(tokens) and tokens[i + 1] == "table":
            return "drop table"
        if tok == "format" and i + 1 < len(tokens) and tokens[i + 1] == "disk":
            return "format disk"
        if tok == "remove" and i + 1 < len(tokens) and tokens[i + 1] == "all":
            return "remove all"
        if tok == "api" and i + 1 < len(tokens) and tokens[i + 1] == "key":
            return "api key"
        # Exact HAND_RULE phrases only (do not match "ignore the decorative…")
        if tok == "ignore" and i + 1 < len(tokens) and tokens[i + 1] == "prior":
            return "ignore prior"
        if (
            tok == "ignore"
            and i + 2 < len(tokens)
            and tokens[i + 1] == "the"
            and tokens[i + 2] == "gate"
        ):
            return "ignore the gate"
    return None


def keyword_counts(tokens: List[str]) -> Dict[str, int]:
    """Whole-word (+ phrase) counts for each specialist on the token window."""
    window = join_tokens(tokens)
    counts = {"Browser": 0, "Structured": 0, "General": 0}
    for w in BROWSER_WORDS:
        counts["Browser"] += sum(1 for t in tokens if t == w)
    for w in STRUCTURED_WORDS:
        counts["Structured"] += sum(1 for t in tokens if t == w)
    for p in STRUCTURED_PHRASES:
        counts["Structured"] += window.count(p)
    for w in GENERAL_WORDS:
        counts["General"] += sum(1 for t in tokens if t == w)
    for p in GENERAL_PHRASES:
        counts["General"] += window.count(p)
    return counts


def hand_rule_pick(counts: Dict[str, int]) -> str:
    """Highest count; ties → Browser > Structured > General; zero → General."""
    if all(v == 0 for v in counts.values()):
        return "General"
    best = max(counts[s] for s in SPECIALISTS)
    for s in SPECIALISTS:
        if counts[s] == best:
            return s
    return "General"


def count_vector(counts: Dict[str, int]) -> Tuple[float, float, float]:
    return (float(counts["Browser"]), float(counts["Structured"]), float(counts["General"]))
