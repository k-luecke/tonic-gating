"""Pure-numpy 2-layer MLP + helpers (no torch/sklearn dependency)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

LABELS = ("Browser", "Structured", "General")
LABEL_TO_I = {lab: i for i, lab in enumerate(LABELS)}


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


class MLP2:
    """x -> ReLU(x W1 + b1) -> (h W2 + b2); float64."""

    def __init__(
        self,
        W1: np.ndarray,
        b1: np.ndarray,
        W2: np.ndarray,
        b2: np.ndarray,
        temperature: float = 1.0,
    ):
        self.W1 = np.asarray(W1, dtype=np.float64)
        self.b1 = np.asarray(b1, dtype=np.float64)
        self.W2 = np.asarray(W2, dtype=np.float64)
        self.b2 = np.asarray(b2, dtype=np.float64)
        self.temperature = float(temperature)

    @property
    def in_dim(self) -> int:
        return int(self.W1.shape[0])

    @property
    def hidden(self) -> int:
        return int(self.W1.shape[1])

    @property
    def out_dim(self) -> int:
        return int(self.W2.shape[1])

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        h_pre = X @ self.W1 + self.b1
        h = np.maximum(h_pre, 0.0)
        logits = h @ self.W2 + self.b2
        return logits, h, h_pre

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits, _, _ = self.forward(X)
        return softmax(logits / max(self.temperature, 1e-6))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=-1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "temperature": self.temperature,
            "in_dim": self.in_dim,
            "hidden": self.hidden,
            "out_dim": self.out_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MLP2":
        return cls(
            W1=np.asarray(d["W1"], dtype=np.float64),
            b1=np.asarray(d["b1"], dtype=np.float64),
            W2=np.asarray(d["W2"], dtype=np.float64),
            b2=np.asarray(d["b2"], dtype=np.float64),
            temperature=float(d.get("temperature", 1.0)),
        )


def init_mlp(in_dim: int, hidden: int = 24, out_dim: int = 3, rng: np.random.Generator | None = None) -> MLP2:
    rng = rng or np.random.default_rng(0)
    # He init for ReLU
    W1 = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0.0, np.sqrt(2.0 / hidden), size=(hidden, out_dim))
    b2 = np.zeros(out_dim)
    return MLP2(W1, b1, W2, b2, temperature=1.0)


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    n = y.shape[0]
    p = probs[np.arange(n), y]
    return float(-np.mean(np.log(np.clip(p, 1e-12, 1.0))))


def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(pred == y)) if len(y) else float("nan")


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    hidden: int = 24,
    epochs: int = 400,
    lr: float = 0.05,
    l2: float = 1e-3,
    batch_size: int = 32,
    seed: int = 20260923,
    patience: int = 40,
) -> Tuple[MLP2, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    mlp = init_mlp(X.shape[1], hidden=hidden, out_dim=3, rng=rng)
    n = X.shape[0]
    history: Dict[str, List[float]] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_state = None
    best_val = float("inf")
    bad = 0

    # Adam state
    params = [mlp.W1, mlp.b1, mlp.W2, mlp.b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    tstep = 0

    for epoch in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = idx[start : start + batch_size]
            xb = X[batch]
            yb = y[batch]
            logits, h, h_pre = mlp.forward(xb)
            probs = softmax(logits)
            # dL/dlogits
            dlogits = probs
            dlogits[np.arange(len(yb)), yb] -= 1.0
            dlogits /= len(yb)

            dW2 = h.T @ dlogits + l2 * mlp.W2
            db2 = np.sum(dlogits, axis=0)
            dh = dlogits @ mlp.W2.T
            dh_pre = dh * (h_pre > 0)
            dW1 = xb.T @ dh_pre + l2 * mlp.W1
            db1 = np.sum(dh_pre, axis=0)

            grads = [dW1, db1, dW2, db2]
            tstep += 1
            for i, g in enumerate(grads):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                mhat = m[i] / (1 - beta1**tstep)
                vhat = v[i] / (1 - beta2**tstep)
                params[i] -= lr * mhat / (np.sqrt(vhat) + eps)

            mlp.W1, mlp.b1, mlp.W2, mlp.b2 = params

        tr_probs = mlp.predict_proba(X)
        tr_loss = cross_entropy(tr_probs, y)
        tr_acc = accuracy(np.argmax(tr_probs, axis=1), y)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)

        if X_val is not None and y_val is not None and len(y_val):
            va_probs = mlp.predict_proba(X_val)
            va_loss = cross_entropy(va_probs, y_val)
            va_acc = accuracy(np.argmax(va_probs, axis=1), y_val)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)
            if va_loss < best_val - 1e-6:
                best_val = va_loss
                bad = 0
                best_state = (
                    mlp.W1.copy(),
                    mlp.b1.copy(),
                    mlp.W2.copy(),
                    mlp.b2.copy(),
                )
            else:
                bad += 1
                if bad >= patience:
                    break
        else:
            best_state = (mlp.W1.copy(), mlp.b1.copy(), mlp.W2.copy(), mlp.b2.copy())

    if best_state is not None:
        mlp.W1, mlp.b1, mlp.W2, mlp.b2 = best_state

    meta = {
        "epochs_ran": len(history["train_loss"]),
        "hidden": hidden,
        "lr": lr,
        "l2": l2,
        "batch_size": batch_size,
        "seed": seed,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_train_acc": history["train_acc"][-1] if history["train_acc"] else None,
        "best_val_loss": best_val if X_val is not None else None,
        "final_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
        "history_tail": {
            "train_acc_last10": history["train_acc"][-10:],
            "val_acc_last10": history["val_acc"][-10:],
        },
    }
    return mlp, meta


def fit_temperature(mlp: MLP2, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Scalar temperature via 1-D grid on val NLL (calibration)."""
    if X_val is None or len(y_val) == 0:
        return 1.0
    logits, _, _ = mlp.forward(X_val)
    best_t, best_nll = 1.0, float("inf")
    for t in np.concatenate([np.linspace(0.5, 3.0, 26), np.array([1.0])]):
        probs = softmax(logits / t)
        nll = cross_entropy(probs, y_val)
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t
