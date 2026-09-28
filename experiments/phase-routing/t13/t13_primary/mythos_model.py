#!/usr/bin/env python3
"""T13 mythos_tiny MoE-of-loops mixture (numpy, analytic grads). Affinity untouched."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from mythos_config import (
    BATCH,
    K,
    L_MAX,
    LEAK,
    S_E,
    S_FULL,
    VOCAB,
    W,
    hashed_phases,
    slice_bounds,
)


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(logits, axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


@dataclass
class MythosParams:
    embed: np.ndarray  # [V, W]
    W_h: np.ndarray  # [K, W, W]
    W_x: np.ndarray  # [K, W, W]
    b: np.ndarray  # [K, W]
    alloc_w: np.ndarray  # [K, W]
    readout_re: np.ndarray  # [W, V]
    readout_im: np.ndarray  # [W, V]
    phases: np.ndarray  # [K] fixed


def init_params(seed: int, phases: Optional[np.ndarray] = None) -> MythosParams:
    rng = np.random.Generator(np.random.PCG64(seed))
    scale = 0.25
    phases = hashed_phases() if phases is None else np.asarray(phases, dtype=np.float64)
    embed = rng.normal(0.0, scale, size=(VOCAB, W))
    # value-hint: dim0 carries token id scale for summary tokens 0..15
    embed[:, 0] = (np.arange(VOCAB) % 16) / 16.0
    W_h = np.stack([np.eye(W) * 0.5 + rng.normal(0.0, 0.02, size=(W, W)) for _ in range(K)])
    W_x = np.stack([np.eye(W) * 0.5 + rng.normal(0.0, 0.02, size=(W, W)) for _ in range(K)])
    return MythosParams(
        embed=embed,
        W_h=W_h,
        W_x=W_x,
        b=np.zeros((K, W), dtype=np.float64),
        alloc_w=rng.normal(0.0, 0.05, size=(K, W)),
        readout_re=rng.normal(0.0, 0.05, size=(W, VOCAB)),
        readout_im=rng.normal(0.0, 0.05, size=(W, VOCAB)),
        phases=phases.copy(),
    )


def param_list(p: MythosParams) -> List[np.ndarray]:
    return [p.embed, p.W_h, p.W_x, p.b, p.alloc_w, p.readout_re, p.readout_im]


def clone_params(p: MythosParams) -> MythosParams:
    return MythosParams(
        embed=p.embed.copy(),
        W_h=p.W_h.copy(),
        W_x=p.W_x.copy(),
        b=p.b.copy(),
        alloc_w=p.alloc_w.copy(),
        readout_re=p.readout_re.copy(),
        readout_im=p.readout_im.copy(),
        phases=p.phases.copy(),
    )


def save_params(path, p: MythosParams) -> None:
    np.savez_compressed(
        path,
        embed=p.embed,
        W_h=p.W_h,
        W_x=p.W_x,
        b=p.b,
        alloc_w=p.alloc_w,
        readout_re=p.readout_re,
        readout_im=p.readout_im,
        phases=p.phases,
    )


def load_params(path) -> MythosParams:
    z = np.load(path)
    return MythosParams(
        embed=z["embed"],
        W_h=z["W_h"],
        W_x=z["W_x"],
        b=z["b"],
        alloc_w=z["alloc_w"],
        readout_re=z["readout_re"],
        readout_im=z["readout_im"],
        phases=z["phases"],
    )


def make_batch(rng: np.random.Generator, batch: int = BATCH) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-slice sum task — small per-slice summaries so target is linearly aggregable.

    Each slice summary s_e = sum(tokens) % 16 written into slice end; target = sum_e s_e
    (range 0..60). Vocab remains 256 (locked); unused ids are simply never targets.
    """
    x = rng.integers(0, VOCAB, size=(batch, S_FULL), dtype=np.int64)
    summaries = []
    for e in range(K):
        lo, hi = slice_bounds(e)
        s = np.sum(x[:, lo:hi], axis=1) % 16
        summaries.append(s)
        x[:, hi - 1] = s  # observable summary token in-slice
    target = np.zeros(batch, dtype=np.int64)
    for s in summaries:
        target = target + s  # 0..60, no wrap
    return x, target.astype(np.int64)


def _run_expert_loop(
    p: MythosParams, pooled: np.ndarray, e: int, depth: int = L_MAX
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Returns h_final, hs (incl h0), pres (pre-tanh)."""
    h = pooled.copy()
    hs = [h.copy()]
    pres = []
    Wh = p.W_h[e]
    Wx = p.W_x[e]
    be = p.b[e]
    for _ in range(depth):
        pre = h @ Wh.T + pooled @ Wx.T + be
        h = (1.0 - LEAK) * h + LEAK * np.tanh(pre)
        hs.append(h.copy())
        pres.append(pre)
    return h, hs, pres


def expert_features_raw(
    p: MythosParams,
    x: np.ndarray,
    shared_ctx: bool = False,
    depth: int = L_MAX,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Ungated expert features + allocator gates. cache for backward."""
    B = x.shape[0]
    raw = np.zeros((B, K, W), dtype=np.float64)
    pools = np.zeros((B, K, W), dtype=np.float64)
    caches = []
    for e in range(K):
        if shared_ctx:
            emb = p.embed[x]
            pooled = emb.mean(axis=1)
        else:
            lo, hi = slice_bounds(e)
            emb = p.embed[x[:, lo:hi]]
            pooled = emb.mean(axis=1)
        pools[:, e, :] = pooled
        h, hs, pres = _run_expert_loop(p, pooled, e, depth=depth)
        raw[:, e, :] = h
        caches.append({"hs": hs, "pres": pres, "emb_tokens": x if shared_ctx else x[:, slice_bounds(e)[0] : slice_bounds(e)[1]]})
    global_pool = p.embed[x].mean(axis=1)
    scores = global_pool @ p.alloc_w.T
    gates = _softmax(scores, axis=1)
    feats = raw * gates[:, :, None]
    cache = {
        "raw": raw,
        "pools": pools,
        "gates": gates,
        "scores": scores,
        "global_pool": global_pool,
        "expert_caches": caches,
        "x": x,
        "shared_ctx": shared_ctx,
        "depth": depth,
    }
    return feats, raw, gates, cache


def medium_aggregate(feats: np.ndarray, phases: np.ndarray) -> np.ndarray:
    out = np.zeros((feats.shape[0], feats.shape[2]), dtype=np.complex128)
    for e in range(feats.shape[1]):
        out += feats[:, e, :].astype(np.float64) * np.exp(1j * float(phases[e]))
    return out


def recover_c_hat(B: np.ndarray, phases: np.ndarray, e: int) -> np.ndarray:
    rot = B * np.exp(-1j * float(phases[e]))
    return _relu(np.real(rot))


def readout_from_medium(p: MythosParams, B: np.ndarray) -> np.ndarray:
    return np.real(B) @ p.readout_re + np.imag(B) @ p.readout_im


def nll_acc(logits: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float, float, np.ndarray]:
    probs = _softmax(logits, axis=1)
    idx = np.arange(target.shape[0])
    nll = -np.log(probs[idx, target] + 1e-12)
    acc = float(np.mean(np.argmax(probs, axis=1) == target))
    return nll, float(np.mean(nll)), acc, probs


def forward(
    p: MythosParams,
    x: np.ndarray,
    target: np.ndarray,
    shared_ctx: bool = False,
    silence: Optional[int] = None,
) -> Dict:
    feats, raw, gates, cache = expert_features_raw(p, x, shared_ctx=shared_ctx)
    if silence is not None:
        feats = feats.copy()
        feats[:, silence, :] = 0.0
    Bmed = medium_aggregate(feats, p.phases)
    logits = readout_from_medium(p, Bmed)
    nll_vec, nll, acc, probs = nll_acc(logits, target)
    return {
        "nll": nll,
        "nll_vec": nll_vec,
        "acc": acc,
        "logits": logits,
        "probs": probs,
        "feats": feats,
        "raw": raw,
        "gates": gates,
        "B": Bmed,
        "cache": cache,
        "target": target,
    }


def sgd_step(
    p: MythosParams,
    x: np.ndarray,
    target: np.ndarray,
    lr: float,
    shared_ctx: bool = False,
) -> Tuple[float, float, float, bool]:
    """One analytic SGD step. Returns nll, acc, grad_norm, finite_ok."""
    out = forward(p, x, target, shared_ctx=shared_ctx)
    nll, acc, probs, logits = out["nll"], out["acc"], out["probs"], out["logits"]
    feats, cache = out["feats"], out["cache"]
    Bsz = x.shape[0]

    # dL/dlogits
    dlogits = probs.copy()
    dlogits[np.arange(Bsz), target] -= 1.0
    dlogits /= Bsz

    # readout grads
    Bre = np.real(out["B"])
    Bim = np.imag(out["B"])
    d_readout_re = Bre.T @ dlogits
    d_readout_im = Bim.T @ dlogits
    dBre = dlogits @ p.readout_re.T
    dBim = dlogits @ p.readout_im.T
    # complex dB
    dB = dBre + 1j * dBim

    # dL/dfeats via phases
    dfeats = np.zeros_like(feats)
    for e in range(K):
        phase = np.exp(1j * float(p.phases[e]))
        # dL/d(f_e) = Re(conj(phase) * dB) = Re(dB * exp(-iφ))
        dfeats[:, e, :] = np.real(dB * np.exp(-1j * float(p.phases[e])))

    raw = cache["raw"]
    gates = cache["gates"]
    # feats = raw * gates[:, :, None]
    d_raw = dfeats * gates[:, :, None]
    d_gates = np.sum(dfeats * raw, axis=2)  # [B, K]

    # softmax backward
    # gates = softmax(scores); d_scores = gates * (d_gates - sum(d_gates*gates))
    sg = np.sum(d_gates * gates, axis=1, keepdims=True)
    d_scores = gates * (d_gates - sg)
    global_pool = cache["global_pool"]
    d_alloc_w = d_scores.T @ global_pool  # [K, W]
    d_global_pool = d_scores @ p.alloc_w  # [B, W]

    # embed grads from global pool (mean over S_FULL)
    d_embed = np.zeros_like(p.embed)
    for t in range(S_FULL):
        np.add.at(d_embed, x[:, t], d_global_pool / S_FULL)

    d_W_h = np.zeros_like(p.W_h)
    d_W_x = np.zeros_like(p.W_x)
    d_b = np.zeros_like(p.b)

    for e in range(K):
        ec = cache["expert_caches"][e]
        hs, pres = ec["hs"], ec["pres"]
        pooled = cache["pools"][:, e, :]
        dh = d_raw[:, e, :].copy()
        Wh = p.W_h[e]
        Wx = p.W_x[e]
        d_pooled = np.zeros_like(pooled)
        for step in reversed(range(len(pres))):
            pre = pres[step]
            h_prev = hs[step]
            # h = (1-LEAK)*h_prev + LEAK*tanh(pre)
            dtanh = dh * LEAK * (1.0 - np.tanh(pre) ** 2)
            d_pooled += dtanh @ Wx
            d_W_x[e] += dtanh.T @ pooled
            d_W_h[e] += dtanh.T @ h_prev
            d_b[e] += dtanh.sum(axis=0)
            dh = dh * (1.0 - LEAK) + dtanh @ Wh
        d_pooled += dh  # residual from h0=pooled

        # pooled = mean(embed[tokens])
        if cache["shared_ctx"]:
            for t in range(S_FULL):
                np.add.at(d_embed, x[:, t], d_pooled / S_FULL)
        else:
            lo, hi = slice_bounds(e)
            for t in range(lo, hi):
                np.add.at(d_embed, x[:, t], d_pooled / S_E)

    # grad norm + finite check
    grads = [d_embed, d_W_h, d_W_x, d_b, d_alloc_w, d_readout_re, d_readout_im]
    g2 = 0.0
    finite = np.isfinite(nll)
    for g in grads:
        if not np.all(np.isfinite(g)):
            finite = False
        g2 += float(np.sum(g * g))
    grad_norm = float(np.sqrt(g2))

    if finite:
        p.embed -= lr * d_embed
        p.W_h -= lr * d_W_h
        p.W_x -= lr * d_W_x
        p.b -= lr * d_b
        p.alloc_w -= lr * d_alloc_w
        p.readout_re -= lr * d_readout_re
        p.readout_im -= lr * d_readout_im
        for arr in param_list(p):
            if not np.all(np.isfinite(arr)):
                finite = False
                break

    return nll, acc, grad_norm, bool(finite)
