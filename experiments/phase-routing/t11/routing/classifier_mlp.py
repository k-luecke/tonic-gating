"""Calibrated context classifier — loads hashed freeze-ready MLP weights.

Architecture: 2-layer MLP on first-32 token features (HAND_RULE lexicon +
train-only unigram bag). Trained on sealed seeds disjoint from smoke seed
20260923. See artifacts/classifier_mlp_v1.json and
t11/docs/T11_CLASSIFIER_CALIBRATION_RECEIPT.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .features import featurize
from .lexicon import first_n_tokens, keyword_counts, refuse_match, tokenize
from .mlp_core import LABELS, MLP2

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "artifacts" / "classifier_mlp_v1.json"
CONTEXT_CARRY_N = 32
EMA_ALPHA = 0.5


def load_weights(path: Path | str = DEFAULT_WEIGHTS) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class ClassifierMLPOrchestrator:
    """Stateless calibrated MLP; optional EMA feature buffer for classifier-buffer carry."""

    name: str = "classifier_mlp"
    status: str = "calibrated"
    weights_path: str = str(DEFAULT_WEIGHTS)
    first_n: int = 32
    context_carry_n: int = CONTEXT_CARRY_N
    ema_alpha: float = EMA_ALPHA
    _blob: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _mlp: Optional[MLP2] = field(default=None, repr=False)
    _unigram_vocab: List[str] = field(default_factory=list, repr=False)
    _buffer: Optional[np.ndarray] = field(default=None, repr=False)
    _prior_tokens: List[str] = field(default_factory=list)
    _carry_mode_active: Optional[str] = None
    disclosure: str = ""

    def __post_init__(self) -> None:
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._mlp is not None:
            return
        blob = load_weights(self.weights_path)
        self._blob = blob
        self._mlp = MLP2.from_dict(blob["weights"])
        self._unigram_vocab = list(blob["features"]["unigram_vocab"])
        self.first_n = int(blob["architecture"].get("first_n_tokens", 32))
        self.disclosure = blob.get(
            "disclosure",
            "calibrated 2-layer MLP; hashed weights; disjoint train/eval",
        )
        # Sanity: dims match
        assert self._mlp.in_dim == (
            len(blob["features"]["lexicon_names"]) + len(self._unigram_vocab)
        )

    @property
    def weights_meta(self) -> Dict[str, Any]:
        self._ensure_loaded()
        assert self._blob is not None
        return {
            "format": self._blob.get("format"),
            "weights_path": self.weights_path,
            "architecture": self._blob.get("architecture"),
            "split": self._blob.get("split"),
            "metrics": self._blob.get("metrics"),
        }

    def reset(self) -> None:
        self._buffer = None
        self._prior_tokens = []
        self._carry_mode_active = None

    def serialize_buffer(self) -> Dict[str, Any]:
        if self._buffer is None:
            return {"kind": "classifier_buffer", "buffer": None, "bytes": 0}
        # float64 feature EMA
        return {
            "kind": "classifier_buffer",
            "buffer": self._buffer.tolist(),
            "bytes": int(self._buffer.size * 8),
        }

    def load_buffer(self, blob: Optional[Dict[str, Any]]) -> None:
        if not blob or blob.get("buffer") is None:
            self._buffer = None
            return
        self._buffer = np.asarray(blob["buffer"], dtype=np.float64)

    def on_window_break(self, carry_mode: str, prior_text: str = "") -> Dict[str, Any]:
        self._carry_mode_active = carry_mode
        if carry_mode == "classifier-buffer":
            ser = self.serialize_buffer()
            return {
                "carry_mode": carry_mode,
                "bytes_carried": ser["bytes"],
                "tokens_carried": 0,
                "buffer_dim": int(self._buffer.size) if self._buffer is not None else 0,
            }
        if carry_mode == "context-carry":
            self._buffer = None
            toks = tokenize(prior_text)[-self.context_carry_n :]
            self._prior_tokens = toks
            raw = " ".join(toks).encode("utf-8")
            return {
                "carry_mode": carry_mode,
                "bytes_carried": len(raw),
                "tokens_carried": len(toks),
            }
        self._buffer = None
        self._prior_tokens = []
        return {
            "carry_mode": carry_mode,
            "bytes_carried": 0,
            "tokens_carried": 0,
            "note": "classifier_mlp treats hum-state/cold as cold",
        }

    def _window_tokens(self, text: str, post_break: bool) -> tuple[List[str], bool]:
        base = first_n_tokens(text, self.first_n)
        used_carry = False
        if post_break and self._prior_tokens and self._carry_mode_active == "context-carry":
            window = (self._prior_tokens + base)[-self.first_n :]
            used_carry = True
        else:
            window = base
        return window, used_carry

    def decide(self, text: str, *, post_break: bool = False, observation=None) -> Dict[str, Any]:
        self._ensure_loaded()
        assert self._mlp is not None

        window, used_carry = self._window_tokens(text, post_break)
        cue = refuse_match(window)
        if cue is not None:
            return {
                "specialist": None,
                "abstain": True,
                "abstain_reason": cue,
                "counts": keyword_counts(window),
                "window_tokens": window,
                "used_carry": used_carry,
                "buffer": self._buffer.tolist() if self._buffer is not None else None,
                "probs": None,
            }

        feats, _ = featurize(
            text,
            first_n=self.first_n,
            unigram_vocab=self._unigram_vocab,
            tokens_override=window,
        )
        x = np.asarray(feats, dtype=np.float64)

        if (
            post_break
            and self._carry_mode_active == "classifier-buffer"
            and self._buffer is not None
            and self._buffer.shape == x.shape
        ):
            x = x + self._buffer
            used_carry = True

        probs = self._mlp.predict_proba(x)[0]
        pick_i = int(np.argmax(probs))
        pick = LABELS[pick_i]

        # EMA buffer update on raw (pre-carry-add) features
        raw = np.asarray(feats, dtype=np.float64)
        if self._buffer is None:
            self._buffer = raw.copy()
        else:
            a = self.ema_alpha
            self._buffer = a * raw + (1.0 - a) * self._buffer

        self._prior_tokens = tokenize(text)
        return {
            "specialist": pick,
            "abstain": False,
            "abstain_reason": None,
            "counts": keyword_counts(window),
            "window_tokens": window,
            "used_carry": used_carry,
            "buffer": self._buffer.tolist(),
            "probs": {lab: float(probs[i]) for i, lab in enumerate(LABELS)},
        }
