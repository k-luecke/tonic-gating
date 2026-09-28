"""Feature extraction for T11 context classifier (first-32 tokens).

Lexicon dims are fixed from HAND_RULE (not fit on data). Optional unigram bag
vocab is fit on the *train* split only and frozen into the weight blob.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .lexicon import (
    BROWSER_WORDS,
    GENERAL_PHRASES,
    GENERAL_WORDS,
    STRUCTURED_PHRASES,
    STRUCTURED_WORDS,
    first_n_tokens,
    join_tokens,
)

# Stable lexicon feature order (HAND_RULE-derived; not data-fit).
LEXICON_FEATURES: List[Tuple[str, str]] = (
    [("kw:Browser:" + w, "word") for w in BROWSER_WORDS]
    + [("kw:Structured:" + w, "word") for w in STRUCTURED_WORDS]
    + [("ph:Structured:" + p, "phrase") for p in STRUCTURED_PHRASES]
    + [("kw:General:" + w, "word") for w in GENERAL_WORDS]
    + [("ph:General:" + p, "phrase") for p in GENERAL_PHRASES]
)

LEXICON_NAMES: List[str] = [n for n, _ in LEXICON_FEATURES]


def lexicon_vector(tokens: Sequence[str]) -> List[float]:
    window = join_tokens(list(tokens))
    toks = list(tokens)
    out: List[float] = []
    for name, kind in LEXICON_FEATURES:
        key = name.split(":", 2)[2]
        if kind == "word":
            out.append(float(sum(1 for t in toks if t == key)))
        else:
            out.append(float(window.count(key)))
    return out


def unigram_vector(tokens: Sequence[str], vocab: Sequence[str]) -> List[float]:
    counts = {t: 0 for t in vocab}
    for t in tokens:
        if t in counts:
            counts[t] += 1
    return [float(counts[t]) for t in vocab]


def featurize(
    text: str,
    *,
    first_n: int = 32,
    unigram_vocab: Sequence[str] | None = None,
    tokens_override: Sequence[str] | None = None,
) -> Tuple[List[float], List[str]]:
    tokens = list(tokens_override) if tokens_override is not None else first_n_tokens(text, first_n)
    feats = lexicon_vector(tokens)
    if unigram_vocab:
        feats = feats + unigram_vector(tokens, unigram_vocab)
    return feats, tokens


def feature_dim(unigram_vocab: Sequence[str] | None = None) -> int:
    return len(LEXICON_NAMES) + (len(unigram_vocab) if unigram_vocab else 0)


def feature_names(unigram_vocab: Sequence[str] | None = None) -> List[str]:
    names = list(LEXICON_NAMES)
    if unigram_vocab:
        names.extend("uni:" + t for t in unigram_vocab)
    return names
