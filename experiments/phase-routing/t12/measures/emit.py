"""Emit all §4.1 measures on a smoke batch."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from model.recurrent_toy import RecurrentDepthToy, ToyConfig, sweep_depths


@dataclass
class MeasureBundle:
    ACC: float
    C_bar: float
    LPC: float
    OT_T: Dict[str, Any]
    BP: Dict[str, Any]
    SD: Dict[str, Any]
    T_b: Dict[str, Any]
    nll: float = float("nan")
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def compute_lpc(acc: float, c_bar: float, nll: Optional[float] = None) -> float:
    """Accuracy-per-compute; also stash nll/C if provided via caller."""
    c = max(float(c_bar), 1e-9)
    return float(acc) / c


def budget_profile(
    depths_by_regime: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for reg, depths in depths_by_regime.items():
        arr = np.asarray(list(depths), dtype=np.float64)
        if arr.size == 0:
            out[reg] = {"mean": None, "std": None, "hist": {}, "n": 0}
            continue
        hist: Dict[str, int] = {}
        for d in arr.astype(int).tolist():
            hist[str(d)] = hist.get(str(d), 0) + 1
        out[reg] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "hist": hist,
            "n": int(arr.size),
        }
    # separation: easy mean < hard mean
    if "easy" in out and "hard" in out and out["easy"]["mean"] is not None:
        out["separation"] = {
            "easy_mean": out["easy"]["mean"],
            "hard_mean": out["hard"]["mean"],
            "gap": float(out["hard"]["mean"] - out["easy"]["mean"]),
            "separates": bool(out["hard"]["mean"] - out["easy"]["mean"] >= 2.0),
        }
    return out


def overthinking_curve(
    sequences_by_regime: Mapping[str, np.ndarray],
    depths: Sequence[int] = range(1, 13),
    width: int = 64,
    seed: int = 20260923,
) -> Dict[str, Any]:
    """OT(T): NLL/ACC vs forced depth T by regime (measurement sweep)."""
    out: Dict[str, Any] = {"depths": list(depths), "by_regime": {}}
    for reg, seqs in sequences_by_regime.items():
        curve = sweep_depths(seqs, reg, depths=depths, width=width, seed=seed)  # type: ignore[arg-type]
        out["by_regime"][reg] = {str(k): v for k, v in curve.items()}
        # OT signature: elevate past T*
        nlls = [(d, curve[d]["nll"]) for d in sorted(curve)]
        best_d = min(nlls, key=lambda x: x[1])[0]
        after = [n for d, n in nlls if d > best_d]
        out["by_regime"][reg]["_ot"] = {
            "T_star": best_d,
            "degrades_past": bool(after and float(np.mean(after)) > curve[best_d]["nll"] + 0.005),
        }
    return out


def silenced_delta(
    coord_acc: float,
    stateless_acc: float,
    coord_c: float,
    stateless_c: float,
) -> Dict[str, Any]:
    """SD: coordinator vs stateless under silence (H4)."""
    return {
        "delta_acc": float(coord_acc - stateless_acc),
        "delta_C": float(coord_c - stateless_c),
        "coord_acc": float(coord_acc),
        "stateless_acc": float(stateless_acc),
        "coord_C": float(coord_c),
        "stateless_C": float(stateless_c),
        "coord_beats_stateless_acc": bool(coord_acc > stateless_acc + 1e-9),
    }


def eval_at_depth(
    sequences: np.ndarray,
    difficulty: str,
    depth: int,
    width: int = 64,
    seed: int = 20260923,
) -> Dict[str, float]:
    toy = RecurrentDepthToy(
        ToyConfig(width=width, seed=seed, difficulty=difficulty)  # type: ignore[arg-type]
    )
    return toy.eval_depth(sequences, depth)


def summarize_batch(
    *,
    acc: float,
    nll: float,
    depths: Sequence[int],
    depths_by_regime: Mapping[str, Sequence[int]],
    ot: Dict[str, Any],
    sd: Dict[str, Any],
    t_b_by_example: Optional[Sequence[int]] = None,
) -> MeasureBundle:
    arr = np.asarray(list(depths), dtype=np.float64)
    c_bar = float(arr.mean()) if arr.size else float("nan")
    bp = budget_profile(depths_by_regime)
    t_b = {
        "mean": float(np.mean(t_b_by_example)) if t_b_by_example else c_bar,
        "by_regime_means": {
            r: bp[r]["mean"] for r in bp if r in ("easy", "hard") and bp[r]["mean"] is not None
        },
        "values_head": list(t_b_by_example)[:16] if t_b_by_example else list(arr.astype(int)[:16]),
    }
    return MeasureBundle(
        ACC=float(acc),
        C_bar=c_bar,
        LPC=compute_lpc(acc, c_bar, nll),
        OT_T=ot,
        BP=bp,
        SD=sd,
        T_b=t_b,
        nll=float(nll),
    )
