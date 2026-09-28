"""
router.py — Learned top-k router baseline (T4).

Same 8 patches as the sheet; router path uses Sheet(gain_mode="fixed") so
gain is 1 (no rhythm). The stencil is replaced by a small MLP:

  features: 32-dim vector =
      24-bin phase histogram of the summed symbol window
      + 8-dim one-hot goal (d*)
  output:   softmax over 8 patches

The one-hot matches the stencil's access to the trial goal (categories map
1:1 to patches); the histogram is the content feature named in the handoff
and lets T5 study content-dependent routing under disturbance.

Train on 2,000 clean symbols, freeze weights, report params / MACs, save
with sha256 content hash.

Patch bit-readouts: per-patch linear filter on the symbol window, trained
once on clean data with the correct patch open (oracle), then frozen.
With gain_mode="fixed" the sheet integrator is not phase-selective, so the
readout uses the window available at the open patch's input (harness intent:
recover target-stream bit after routing).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from content import N_STREAMS, multiplex_streams, stream_phase
from lockin import DT, F0, SAMPLES_PER_SYMBOL
from sheet import Control, N_PATCHES, N_UNITS, PATCH_UNITS, Sheet

# ---------------------------------------------------------------------------
# Feature geometry
# ---------------------------------------------------------------------------
N_HIST_BINS = 24
N_GOAL = N_PATCHES  # 8
FEATURE_DIM = N_HIST_BINS + N_GOAL  # 32
HIDDEN = 64
TOP_K_DEFAULT = 1

DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parent / "results" / "router_weights.npz"
)
DEFAULT_STATS_PATH = (
    Path(__file__).resolve().parent / "results" / "T4_router_stats.txt"
)
DEFAULT_READOUT_PATH = (
    Path(__file__).resolve().parent / "results" / "router_readouts.npz"
)


def phase_histogram(
    window: np.ndarray,
    n_bins: int = N_HIST_BINS,
    *,
    f0: float = F0,
    dt: float = DT,
) -> np.ndarray:
    """
    Amplitude-weighted histogram of carrier phase over a symbol window.

    Phase at sample i is (2π f0 t_i) mod 2π; weight is max(window_i, 0).
    Returns a length-n_bins vector normalized to sum 1 (or zeros if empty).
    """
    x = np.asarray(window, dtype=np.float64).reshape(-1)
    t = np.arange(x.size, dtype=np.float64) * dt
    phase = (2.0 * np.pi * f0 * t) % (2.0 * np.pi)
    weights = np.maximum(x, 0.0)
    bins = np.clip((phase / (2.0 * np.pi) * n_bins).astype(np.int64), 0, n_bins - 1)
    hist = np.bincount(bins, weights=weights, minlength=n_bins).astype(np.float64)
    s = float(hist.sum())
    if s > 0.0:
        hist /= s
    return hist


def one_hot(d: int, n: int = N_GOAL) -> np.ndarray:
    v = np.zeros(n, dtype=np.float64)
    v[int(d) % n] = 1.0
    return v


def router_features(
    window: np.ndarray,
    d_star: int,
    *,
    n_hist_bins: int = N_HIST_BINS,
) -> np.ndarray:
    """32-dim features: phase hist || one-hot(d*)."""
    h = phase_histogram(window, n_bins=n_hist_bins)
    return np.concatenate([h, one_hot(d_star)])


# ---------------------------------------------------------------------------
# Small MLP (numpy)
# ---------------------------------------------------------------------------
def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


@dataclass
class MLPWeights:
    W1: np.ndarray  # (H, F)
    b1: np.ndarray  # (H,)
    W2: np.ndarray  # (K, H)
    b2: np.ndarray  # (K,)

    def param_count(self) -> int:
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    def macs_per_decision(self) -> int:
        """
        Multiply-accumulates for one forward pass (dense matvecs).
        Count each multiply-add as one MAC: F*H + H*K.
        Biases are adds (not counted as MACs).
        """
        f, h = self.W1.shape[1], self.W1.shape[0]
        k = self.W2.shape[0]
        return int(f * h + h * k)


class Router:
    """Top-k MLP router over 8 patches."""

    def __init__(
        self,
        *,
        feature_dim: int = FEATURE_DIM,
        hidden: int = HIDDEN,
        n_patches: int = N_PATCHES,
        seed: int | None = 0,
        weights: MLPWeights | None = None,
    ) -> None:
        self.feature_dim = int(feature_dim)
        self.hidden = int(hidden)
        self.n_patches = int(n_patches)
        self.rng = np.random.default_rng(seed)
        if weights is None:
            # He init for ReLU
            s1 = np.sqrt(2.0 / self.feature_dim)
            s2 = np.sqrt(2.0 / self.hidden)
            self.weights = MLPWeights(
                W1=self.rng.normal(0.0, s1, size=(self.hidden, self.feature_dim)),
                b1=np.zeros(self.hidden, dtype=np.float64),
                W2=self.rng.normal(0.0, s2, size=(self.n_patches, self.hidden)),
                b2=np.zeros(self.n_patches, dtype=np.float64),
            )
        else:
            self.weights = weights
        self.frozen = False

    # ----- forward -----
    def logits(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        if x.shape[-1] != self.feature_dim:
            raise ValueError(f"expected feature dim {self.feature_dim}, got {x.shape[-1]}")
        h = _relu(x @ self.weights.W1.T + self.weights.b1)
        z = h @ self.weights.W2.T + self.weights.b2
        return z[0] if single else z

    def probs(self, features: np.ndarray) -> np.ndarray:
        return _softmax(self.logits(features), axis=-1)

    def select_topk(self, features: np.ndarray, k: int = TOP_K_DEFAULT) -> np.ndarray:
        """Return indices of top-k patches (descending score)."""
        p = self.probs(features)
        if p.ndim == 1:
            return np.argsort(p)[::-1][:k].astype(np.int64)
        # batch: (B, k)
        return np.argsort(p, axis=-1)[..., ::-1][..., :k].astype(np.int64)

    def make_control(self, open_patches: np.ndarray | list[int] | int) -> Control:
        """Inhibition mask for selected open patches; no pilot (router path)."""
        if np.isscalar(open_patches):
            opens = [int(open_patches)]
        else:
            opens = [int(x) for x in np.asarray(open_patches).reshape(-1)]
        inhib = np.ones(self.n_patches, dtype=np.float64)
        for k in opens:
            if not 0 <= k < self.n_patches:
                raise ValueError(k)
            inhib[k] = 0.0
        return Control(inhibition=inhib, pilot_signal=None)

    def control_for_target(
        self, d_star: int, window: np.ndarray | None = None, *, top_k: int = 1
    ) -> Control:
        """
        Build Control for trial goal d*.

        If window is given, features = hist(window)||one_hot(d*) and the MLP
        selects top-k (should recover d* after clean training). If window is
        None, open patch d* directly (oracle / stencil-parity shortcut).
        """
        d_star = int(d_star) % self.n_patches
        if window is None:
            return self.make_control(d_star)
        feat = router_features(window, d_star)
        top = self.select_topk(feat, k=top_k)
        return self.make_control(top)

    # ----- train -----
    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        n_epochs: int = 80,
        batch_size: int = 64,
        lr: float = 0.05,
        l2: float = 1e-4,
        seed: int | None = 0,
        verbose: bool = False,
    ) -> list[float]:
        """SGD + CE on soft labels (integer class indices). Returns epoch losses."""
        if self.frozen:
            raise RuntimeError("router is frozen; refuse to train")
        X = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        n = X.shape[0]
        rng = np.random.default_rng(seed)
        losses: list[float] = []
        W1, b1 = self.weights.W1, self.weights.b1
        W2, b2 = self.weights.W2, self.weights.b2

        for epoch in range(int(n_epochs)):
            perm = rng.permutation(n)
            total = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb = X[idx]
                yb = y[idx]
                # forward
                h_pre = xb @ W1.T + b1
                h = _relu(h_pre)
                logits = h @ W2.T + b2
                p = _softmax(logits, axis=-1)
                # CE
                p_y = p[np.arange(len(yb)), yb]
                loss = float(-np.mean(np.log(p_y + 1e-12)))
                total += loss
                n_batches += 1
                # dL/dlogits
                dlogits = p
                dlogits[np.arange(len(yb)), yb] -= 1.0
                dlogits /= len(yb)
                dW2 = dlogits.T @ h + l2 * W2
                db2 = dlogits.sum(axis=0)
                dh = dlogits @ W2
                dh_pre = dh * (h_pre > 0.0)
                dW1 = dh_pre.T @ xb + l2 * W1
                db1 = dh_pre.sum(axis=0)
                W1 = W1 - lr * dW1
                b1 = b1 - lr * db1
                W2 = W2 - lr * dW2
                b2 = b2 - lr * db2
            self.weights = MLPWeights(W1=W1, b1=b1, W2=W2, b2=b2)
            ep_loss = total / max(n_batches, 1)
            losses.append(ep_loss)
            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                acc = self.accuracy(X, y)
                print(f"epoch {epoch:3d}  loss={ep_loss:.4f}  acc={acc:.4f}")
        return losses

    def accuracy(self, features: np.ndarray, labels: np.ndarray) -> float:
        pred = self.select_topk(features, k=1).reshape(-1)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        return float(np.mean(pred == y))

    def freeze(self) -> None:
        self.frozen = True

    def param_count(self) -> int:
        return self.weights.param_count()

    def macs_per_decision(self) -> int:
        return self.weights.macs_per_decision()

    # ----- persistence -----
    def save(self, path: str | Path) -> str:
        """
        Save weights to npz. Returns sha256 hex digest of the file bytes.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W1=self.weights.W1,
            b1=self.weights.b1,
            W2=self.weights.W2,
            b2=self.weights.b2,
            feature_dim=np.array([self.feature_dim]),
            hidden=np.array([self.hidden]),
            n_patches=np.array([self.n_patches]),
            n_hist_bins=np.array([N_HIST_BINS]),
            frozen=np.array([1 if self.frozen else 0]),
        )
        return sha256_file(path)

    @classmethod
    def load(cls, path: str | Path) -> "Router":
        path = Path(path)
        data = np.load(path)
        weights = MLPWeights(
            W1=np.array(data["W1"], dtype=np.float64, copy=True),
            b1=np.array(data["b1"], dtype=np.float64, copy=True),
            W2=np.array(data["W2"], dtype=np.float64, copy=True),
            b2=np.array(data["b2"], dtype=np.float64, copy=True),
        )
        r = cls(
            feature_dim=int(data["feature_dim"][0]),
            hidden=int(data["hidden"][0]),
            n_patches=int(data["n_patches"][0]),
            weights=weights,
        )
        r.frozen = bool(int(data["frozen"][0])) if "frozen" in data else True
        return r


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Patch bit-readouts (linear on symbol window)
# ---------------------------------------------------------------------------
@dataclass
class PatchReadouts:
    """Per-patch ridge linear filters on the symbol window → bit score."""

    weights: np.ndarray  # (K, SAMPLES+1) including bias in last column
    n_samples: int = SAMPLES_PER_SYMBOL

    def predict_bit(self, window: np.ndarray, patch: int) -> int:
        w = np.asarray(window, dtype=np.float64).reshape(-1)
        if w.size != self.n_samples:
            raise ValueError(f"window length {w.size} != {self.n_samples}")
        xb = np.append(w, 1.0)
        score = float(xb @ self.weights[int(patch)])
        return 1 if score > 0.5 else 0

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weights=self.weights, n_samples=np.array([self.n_samples]))
        return sha256_file(path)

    @classmethod
    def load(cls, path: str | Path) -> "PatchReadouts":
        data = np.load(path)
        return cls(
            weights=np.array(data["weights"], dtype=np.float64, copy=True),
            n_samples=int(data["n_samples"][0]),
        )


def fit_patch_readouts(
    n_symbols: int = 2000,
    *,
    seed: int = 1,
    lam: float = 1e-2,
) -> PatchReadouts:
    """
    Train per-patch linear readouts on clean multiplexed symbols.

    For each symbol we record the window and all 8 bits; readout k is fit to
    predict bit of stream k (correct-patch-open / category↔patch training).
    """
    rng = np.random.default_rng(seed)
    S = SAMPLES_PER_SYMBOL
    X = np.empty((n_symbols, S), dtype=np.float64)
    Y = np.empty((n_symbols, N_STREAMS), dtype=np.float64)
    for i in range(n_symbols):
        bits = rng.integers(0, 2, size=N_STREAMS).astype(np.float64)
        active = {d: float(bits[d]) for d in range(N_STREAMS)}
        X[i] = multiplex_streams(S, active)
        Y[i] = bits
    Xb = np.concatenate([X, np.ones((n_symbols, 1))], axis=1)
    W = np.empty((N_PATCHES, S + 1), dtype=np.float64)
    eye = np.eye(S + 1)
    eye[-1, -1] = 0.0  # do not regularize bias
    AtA = Xb.T @ Xb + lam * eye
    for k in range(N_PATCHES):
        W[k] = np.linalg.solve(AtA, Xb.T @ Y[:, k])
    return PatchReadouts(weights=W, n_samples=S)


# ---------------------------------------------------------------------------
# Clean-symbol dataset for the router
# ---------------------------------------------------------------------------
def make_clean_router_dataset(
    n_symbols: int = 2000,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    2000 clean symbols: random d*, random bits on all 8 streams.

    Features = hist(window)||one_hot(d*); label = d* (patch index).
    Also returns meta list with bits and windows for ACC eval.
    """
    rng = np.random.default_rng(seed)
    S = SAMPLES_PER_SYMBOL
    feats = np.empty((n_symbols, FEATURE_DIM), dtype=np.float64)
    labels = np.empty(n_symbols, dtype=np.int64)
    meta: list[dict[str, Any]] = []
    for i in range(n_symbols):
        d_star = int(rng.integers(0, N_STREAMS))
        bits = rng.integers(0, 2, size=N_STREAMS).astype(np.float64)
        active = {d: float(bits[d]) for d in range(N_STREAMS)}
        window = multiplex_streams(S, active)
        feats[i] = router_features(window, d_star)
        labels[i] = d_star
        meta.append({"d_star": d_star, "bits": bits, "window": window})
    return feats, labels, meta


def evaluate_router_path_acc(
    router: Router,
    readouts: PatchReadouts,
    n_symbols: int = 500,
    *,
    seed: int = 123,
    sigma_f: float = 0.0,
    use_mlp: bool = True,
) -> dict[str, float]:
    """
    Clean-condition ACC for the router path.

    For each symbol: build features, MLP selects top-1 patch, run sheet with
    gain_mode='fixed' and that patch open (for side effects / future measures),
    then recover the target bit via the frozen patch readout on the window.
    Primary ACC uses the readout for the *selected* patch vs bit[d*]; with a
    well-trained goal-conditioned router, selected == d*.
    """
    rng = np.random.default_rng(seed)
    sheet = Sheet(sigma_f=sigma_f, seed=seed, gain_mode="fixed")
    correct = 0
    routing_correct = 0
    for _ in range(n_symbols):
        d_star = int(rng.integers(0, N_STREAMS))
        bits = rng.integers(0, 2, size=N_STREAMS).astype(np.float64)
        active = {d: float(bits[d]) for d in range(N_STREAMS)}
        window = multiplex_streams(SAMPLES_PER_SYMBOL, active)
        feat = router_features(window, d_star)
        if use_mlp:
            k_hat = int(router.select_topk(feat, k=1)[0])
        else:
            k_hat = d_star
        routing_correct += int(k_hat == d_star)
        ctrl = router.make_control(k_hat)
        sheet.s[:] = 0.0
        for i in range(SAMPLES_PER_SYMBOL):
            sheet.step(window[i], ctrl)
        # Bit from frozen readout associated with selected patch; label is bit[d*]
        # When k_hat == d_star, readout[k_hat] predicts bit[d*].
        # Readout for the selected patch predicts that stream's bit; label is
        # bit[d*]. Correct routing (k_hat == d*) + good readout → high ACC.
        bit_hat = readouts.predict_bit(window, k_hat)
        correct += int(bit_hat == int(bits[d_star]))
    return {
        "acc": float(correct / n_symbols),
        "routing_acc": float(routing_correct / n_symbols),
        "n_symbols": float(n_symbols),
    }


def train_and_freeze(
    *,
    n_train: int = 2000,
    n_eval: int = 500,
    seed: int = 20260921,
    weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
    readout_path: str | Path = DEFAULT_READOUT_PATH,
    stats_path: str | Path = DEFAULT_STATS_PATH,
    verbose: bool = True,
) -> dict[str, Any]:
    """Full T4 pipeline: train router + readouts, freeze, eval, write stats."""
    weights_path = Path(weights_path)
    readout_path = Path(readout_path)
    stats_path = Path(stats_path)

    feats, labels, _meta = make_clean_router_dataset(n_train, seed=seed)
    router = Router(seed=seed)
    if verbose:
        print(f"Training router on {n_train} clean symbols…")
    router.train(
        feats, labels, n_epochs=60, batch_size=64, lr=0.08, seed=seed, verbose=verbose
    )
    train_acc = router.accuracy(feats, labels)
    router.freeze()

    if verbose:
        print("Fitting patch readouts…")
    readouts = fit_patch_readouts(n_symbols=n_train, seed=seed + 1)

    weight_hash = router.save(weights_path)
    readout_hash = readouts.save(readout_path)

    metrics = evaluate_router_path_acc(
        router, readouts, n_symbols=n_eval, seed=seed + 7, use_mlp=True
    )
    params = router.param_count()
    macs = router.macs_per_decision()

    lines = [
        "T4 — Learned top-k router stats",
        "================================",
        f"feature_dim:        {FEATURE_DIM}  (24-bin phase hist + 8-dim goal one-hot)",
        f"hidden:             {HIDDEN}",
        f"n_patches:          {N_PATCHES}",
        f"param_count:        {params}",
        f"MACs_per_decision:  {macs}",
        f"train_symbols:      {n_train}",
        f"train_routing_acc:  {train_acc:.6f}",
        f"eval_symbols:       {n_eval}",
        f"clean_ACC:          {metrics['acc']:.6f}",
        f"clean_routing_acc:  {metrics['routing_acc']:.6f}",
        f"weights_path:       {weights_path}",
        f"weights_sha256:     {weight_hash}",
        f"readouts_path:      {readout_path}",
        f"readouts_sha256:    {readout_hash}",
        f"gain_mode:          fixed (g=1)",
        f"seed:               {seed}",
    ]
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text("\n".join(lines) + "\n")
    if verbose:
        print("\n".join(lines))

    return {
        "param_count": params,
        "macs_per_decision": macs,
        "train_routing_acc": train_acc,
        "clean_acc": metrics["acc"],
        "clean_routing_acc": metrics["routing_acc"],
        "weights_sha256": weight_hash,
        "readouts_sha256": readout_hash,
        "weights_path": str(weights_path),
        "readouts_path": str(readout_path),
        "stats_path": str(stats_path),
    }


# ---------------------------------------------------------------------------
# T5 helper
# ---------------------------------------------------------------------------
def load_frozen_router(
    weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
) -> Router:
    """Load frozen router for T5 harness."""
    r = Router.load(weights_path)
    r.freeze()
    return r


def load_frozen_readouts(
    readout_path: str | Path = DEFAULT_READOUT_PATH,
) -> PatchReadouts:
    return PatchReadouts.load(readout_path)


if __name__ == "__main__":
    train_and_freeze(verbose=True)
