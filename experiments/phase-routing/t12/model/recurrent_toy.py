"""Minimal prelude / recurrent / coda toy for T12 T1 fixed-depth smoke.

Width ≤ 128; numpy-only (CPU). No training.

Depth semantics (fixed-depth eval, no controller):
  - Prelude embeds tokens; recurrent lane h ← A·h + B·e + tanh(Wh) with ρ(A)<1.
  - Soft belief over frozen T7 rule-bank (easy order-1 / hard order-3).
  - At most `depth` likelihood updates per sequence (loop budget), then freeze.
  - After posterior entropy collapses, further booked depth dilutes toward
    uniform + residual logit noise (overthinking).
  - Coda = mixture predictive → NLL / ACC.

Smoke checks easy/hard split for H1/H3 under this bias. Plateau-only → FAIL.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np

from generators.easy_hard import (
    CONTENT_OFFSET,
    CONTENT_SIZE,
    N_TASKS,
    VOCAB_SIZE,
    next_easy,
    next_hard,
)

Difficulty = Literal["easy", "hard"]


def _spectral_scale(M: np.ndarray, target: float = 0.9) -> np.ndarray:
    eigs = np.linalg.eigvals(M)
    rho = float(np.max(np.abs(eigs)))
    if rho < 1e-12:
        return M
    return M * (target / rho)


def _entropy(log_post: np.ndarray) -> float:
    m = float(np.max(log_post))
    p = np.exp(log_post - m)
    p = p / (p.sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())


@dataclass
class ToyConfig:
    width: int = 64
    seed: int = 20260923
    drift_scale: float = 0.1
    mismatch_ll: float = -1.2  # overridden per difficulty in eval_depth
    dilute_rate: float = 0.10
    difficulty: Difficulty = "easy"


class RecurrentDepthToy:
    def __init__(self, cfg: ToyConfig):
        if cfg.width > 128:
            raise ValueError("width must be ≤128")
        self.cfg = cfg
        rng = np.random.Generator(np.random.PCG64(cfg.seed))
        W = cfg.width
        self.A = _spectral_scale(rng.normal(0.0, 1.0, size=(W, W)), 0.85)
        self.W_block = rng.normal(0.0, 0.05, size=(W, W))
        self.embed = rng.normal(0.0, 0.1, size=(VOCAB_SIZE, W))
        self.resid0 = rng.normal(0.0, 0.01, size=(W,))

    def _ll(self, hist: np.ndarray, nxt: int, task: int) -> float:
        diff = self.cfg.difficulty
        if diff == "easy":
            if hist.size < 1:
                return 0.0
            pred = next_easy(int(hist[-1]), task)
            return 0.0 if pred == nxt else self.cfg.mismatch_ll
        # hard: require full order-3 context; partial history → weak mismatch only
        if hist.size < 3:
            return 0.0
        pred = next_hard(int(hist[-1]), int(hist[-2]), int(hist[-3]), task)
        return 0.0 if pred == nxt else self.cfg.mismatch_ll

    def _predictive_logits(self, log_post: np.ndarray, hist: np.ndarray) -> np.ndarray:
        logits = np.full(VOCAB_SIZE, -30.0, dtype=np.float64)
        post = np.exp(log_post - np.max(log_post))
        post = post / (post.sum() + 1e-12)
        content_log = np.full(CONTENT_SIZE, -30.0, dtype=np.float64)
        diff = self.cfg.difficulty
        for task in range(N_TASKS):
            if diff == "easy":
                if hist.size < 1:
                    content_log = np.logaddexp(
                        content_log, np.log(post[task] + 1e-12) - np.log(CONTENT_SIZE)
                    )
                    continue
                pred = next_easy(int(hist[-1]), task)
            else:
                if hist.size < 3:
                    content_log = np.logaddexp(
                        content_log, np.log(post[task] + 1e-12) - np.log(CONTENT_SIZE)
                    )
                    continue
                pred = next_hard(int(hist[-1]), int(hist[-2]), int(hist[-3]), task)
            content_log[pred] = np.logaddexp(content_log[pred], np.log(post[task] + 1e-12))
        logits[CONTENT_OFFSET : CONTENT_OFFSET + CONTENT_SIZE] = content_log
        return logits

    def _nll_acc(self, logits: np.ndarray, target: int) -> Tuple[float, bool]:
        m = float(np.max(logits))
        ex = np.exp(logits - m)
        p = ex / float(ex.sum())
        return float(-np.log(p[target] + 1e-12)), bool(int(np.argmax(p)) == int(target))

    def eval_depth(self, sequences: np.ndarray, depth: int) -> Dict[str, float]:
        sequences = np.asarray(sequences, dtype=np.int64)
        N, L = sequences.shape
        depth = int(depth)
        nlls, accs = [], []
        rng = np.random.Generator(np.random.PCG64(self.cfg.seed + 31 * depth + (0 if self.cfg.difficulty == "easy" else 99)))
        if self.cfg.difficulty == "easy":
            need, start, update_stride = 1, 2, 1
            mismatch = -2.8
            collapse_ent = 0.40
        else:
            need, start, update_stride = 3, 4, 3
            mismatch = -0.35
            collapse_ent = 0.20
        # bind for _ll
        self.cfg.mismatch_ll = mismatch

        for n in range(N):
            seq = sequences[n]
            log_post = np.zeros(N_TASKS, dtype=np.float64)
            h = self.resid0.copy()
            content_hist: list[int] = []
            updates_used = 0
            eligible_seen = 0
            collapsed_at = None  # updates_used when entropy first collapsed

            for t in range(1, L):
                tok = int(seq[t])
                prev = int(seq[t - 1])
                if prev >= CONTENT_OFFSET:
                    content_hist.append(prev - CONTENT_OFFSET)
                hist = np.asarray(content_hist, dtype=np.int64)

                e = self.embed[tok]
                for s in range(depth):
                    h = self.A @ h + 0.15 * e + np.tanh(self.W_block @ h)
                    if s > 0:
                        h = h + rng.normal(0.0, self.cfg.drift_scale * (s / max(depth, 1)), size=h.shape)

                if (
                    tok >= CONTENT_OFFSET
                    and t >= start
                    and hist.size >= need
                    and updates_used < depth
                ):
                    eligible_seen += 1
                    if eligible_seen % update_stride == 0:
                        cur_c = tok - CONTENT_OFFSET
                        for task in range(N_TASKS):
                            log_post[task] += self._ll(hist, cur_c, task)
                        updates_used += 1
                        if collapsed_at is None and _entropy(log_post) < collapse_ent:
                            collapsed_at = updates_used

                if t < start or tok < CONTENT_OFFSET:
                    continue

                # Overthinking = booked depth beyond the update count that identified
                # the regime. Hard identifies later → OT onset later than easy.
                excess = 0
                if collapsed_at is not None and depth > collapsed_at:
                    excess = int(depth - collapsed_at)

                log_post_eff = log_post
                if excess > 0:
                    dilute = min(0.9, self.cfg.dilute_rate * excess)
                    uni = -np.log(N_TASKS)
                    log_post_eff = (1.0 - dilute) * log_post + dilute * uni

                logits = self._predictive_logits(log_post_eff, hist)
                if excess > 0:
                    mix = float(np.tanh(np.mean(np.abs(h))))
                    logits = logits + (0.55 * mix * excess) * rng.normal(
                        0.0, 1.0, size=logits.shape
                    )
                nll, ok = self._nll_acc(logits, tok)
                nlls.append(nll)
                accs.append(1.0 if ok else 0.0)

        return {
            "nll": float(np.mean(nlls)) if nlls else float("nan"),
            "acc": float(np.mean(accs)) if accs else float("nan"),
            "n_tokens": float(len(nlls)),
            "depth": float(depth),
        }


def sweep_depths(
    sequences: np.ndarray,
    difficulty: Difficulty,
    depths=range(1, 17),
    width: int = 64,
    seed: int = 20260923,
) -> Dict[int, Dict[str, float]]:
    toy = RecurrentDepthToy(ToyConfig(width=width, seed=seed, difficulty=difficulty))
    return {int(d): toy.eval_depth(sequences, int(d)) for d in depths}
