#!/usr/bin/env python3
"""T12 T1 smoke — fixed-depth sweep 1–16 easy vs hard + ACT/Convergence stubs.

No train, no freeze, no PREREG.sha256. Affinity/C₃/hum untouched.

Usage (from repo root or this dir):
  cd /workspace/scp/experiments/phase-routing/t12
  python3 scripts/run_t1_smoke.py

Receipt: results/t1/T1_SWEEP_RECEIPT.json (+ .md)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import numpy as np

T12 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T12))

from controllers.act_stub import ACTStub  # noqa: E402
from controllers.convergence_stub import ConvergenceThresholdStub  # noqa: E402
from generators.easy_hard import build_easy_hard  # noqa: E402
from model.recurrent_toy import sweep_depths  # noqa: E402

PT = ZoneInfo("America/Chicago")
OUT_DIR = T12 / "results" / "t1"
DEPTHS = list(range(1, 17))


def _argmin_metric(curve: Dict[int, Dict[str, float]], key: str = "nll") -> int:
    """First depth within 5% of best NLL (early bottom of the basin)."""
    best_v = min(curve[d][key] for d in curve)
    thresh = best_v + 0.05 * max(abs(best_v), 0.05)
    for d in sorted(curve):
        if curve[d][key] <= thresh:
            return int(d)
    return int(min(curve, key=lambda d: curve[d][key]))


def _degrades_past(curve: Dict[int, Dict[str, float]], t_star: int, key: str = "nll") -> bool:
    """True if mean NLL after T* is measurably worse than at T* (overthinking)."""
    base = curve[t_star][key]
    after = [curve[d][key] for d in sorted(curve) if d > t_star]
    if not after:
        return False
    # require clear rise: last point or mean-after exceeds base by margin
    margin = 0.005  # nats (smoke-scale OT)
    return (float(np.mean(after)) > base + margin) or (after[-1] > base + margin)


def _still_improving(curve: Dict[int, Dict[str, float]], t_star_easy: int, key: str = "nll") -> bool:
    """Hard not-yet-bottomed relative to easy: hard T* > easy T*, or hard NLL still falling past easy T*."""
    # caller passes hard curve; compare slope past easy optimum
    past = [curve[d][key] for d in sorted(curve) if d >= t_star_easy]
    if len(past) < 2:
        return False
    # improving if later mean lower than early-past
    mid = max(1, len(past) // 2)
    return float(np.mean(past[mid:])) < float(np.mean(past[:mid])) - 0.01


def judge_h1_h3(
    easy: Dict[int, Dict[str, float]],
    hard: Dict[int, Dict[str, float]],
) -> Dict[str, Any]:
    t_easy = _argmin_metric(easy, "nll")
    t_hard = _argmin_metric(hard, "nll")
    gap = abs(t_hard - t_easy)
    # also compare NLL at each other's optima
    nll_gap = abs(easy[t_easy]["nll"] - hard[t_hard]["nll"])
    h1 = bool(gap >= 2 or (gap >= 1 and (nll_gap >= 0.02 or abs(easy[t_easy]["acc"] - hard[t_hard]["acc"]) >= 0.02)))
    easy_ot = _degrades_past(easy, t_easy, "nll")
    hard_not_yet = (t_hard > t_easy) or _still_improving(hard, t_easy, "nll")
    # plateau-only: both flat (max-min NLL small) → fail H3
    def span(curve):
        vals = [curve[d]["nll"] for d in curve]
        return float(max(vals) - min(vals))

    plateau = span(easy) < 0.03 and span(hard) < 0.03
    h3 = bool(easy_ot and hard_not_yet and not plateau)
    return {
        "T_star_easy": t_easy,
        "T_star_hard": t_hard,
        "T_star_gap": gap,
        "nll_at_Tstar_easy": easy[t_easy]["nll"],
        "nll_at_Tstar_hard": hard[t_hard]["nll"],
        "acc_at_Tstar_easy": easy[t_easy]["acc"],
        "acc_at_Tstar_hard": hard[t_hard]["acc"],
        "easy_nll_span": span(easy),
        "hard_nll_span": span(hard),
        "easy_degrades_past_optimum": easy_ot,
        "hard_not_yet_at_easy_optimum": hard_not_yet,
        "plateau_only": plateau,
        "H1_measurable_gap": h1,
        "H3_overthinking_signature": h3,
    }


def stub_smoke() -> Dict[str, Any]:
    act = ACTStub()
    conv = ConvergenceThresholdStub()
    out = {
        "act_easy": act.decide(difficulty="easy").__dict__,
        "act_hard": act.decide(difficulty="hard").__dict__,
        "conv_easy": conv.decide(difficulty="easy").__dict__,
        "conv_hard": conv.decide(difficulty="hard").__dict__,
    }
    # numpy → list
    for k, v in list(out.items()):
        for fk, fv in list(v.items()):
            if isinstance(fv, np.ndarray):
                v[fk] = fv.tolist()
    out["interfaces_ok"] = True
    out["note"] = "stateless per-item stubs; dry callable; not trained"
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = build_easy_hard(seed=20260923, n_per_task=12, seq_len=48)
    easy_curve = sweep_depths(data["easy"]["sequences"], "easy", DEPTHS, width=64)
    hard_curve = sweep_depths(data["hard"]["sequences"], "hard", DEPTHS, width=64)
    verdict = judge_h1_h3(easy_curve, hard_curve)
    stubs = stub_smoke()

    h1_ok = verdict["H1_measurable_gap"]
    h3_ok = verdict["H3_overthinking_signature"]
    stubs_ok = stubs["interfaces_ok"]
    t1_pass = bool(h1_ok and h3_ok and stubs_ok)

    receipt: Dict[str, Any] = {
        "ticket": "T1",
        "status": "PASS" if t1_pass else "FAIL",
        "pt": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT"),
        "depths": DEPTHS,
        "controller": None,
        "train": False,
        "freeze": False,
        "width": 64,
        "n_per_task": 12,
        "seq_len": 48,
        "easy_curve": {str(k): v for k, v in easy_curve.items()},
        "hard_curve": {str(k): v for k, v in hard_curve.items()},
        "verdict": verdict,
        "H1": "PASS" if h1_ok else "FAIL",
        "H3": "PASS" if h3_ok else "FAIL",
        "stubs_ACT_Convergence": "PASS" if stubs_ok else "FAIL",
        "stubs": stubs,
        "affinity_c3_hum": "UNTOUCHED",
        "prereg": "DRAFT — no PREREG.sha256 emit",
    }
    if not t1_pass:
        reasons = []
        if not h1_ok:
            reasons.append("H1: no measurable easy/hard gap in T* / NLL bottom")
        if not h3_ok:
            if verdict["plateau_only"]:
                reasons.append("H3: plateau-only / no crossover — H3 untestable")
            else:
                reasons.append(
                    "H3: missing easy-degrade-past-optimum and/or hard-not-yet signature"
                )
        receipt["fail_reasons"] = reasons

    json_path = OUT_DIR / "T1_SWEEP_RECEIPT.json"
    json_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # markdown summary
    lines: List[str] = [
        "# T12 T1 Sweep Receipt",
        "",
        f"- **Status:** **{receipt['status']}**",
        f"- **PT:** {receipt['pt']}",
        f"- **H1:** {receipt['H1']} (T*_easy={verdict['T_star_easy']}, T*_hard={verdict['T_star_hard']}, gap={verdict['T_star_gap']})",
        f"- **H3:** {receipt['H3']} (easy_degrades={verdict['easy_degrades_past_optimum']}, hard_not_yet={verdict['hard_not_yet_at_easy_optimum']}, plateau_only={verdict['plateau_only']})",
        f"- **Stubs ACT+Convergence:** {receipt['stubs_ACT_Convergence']}",
        f"- **NLL@T* easy/hard:** {verdict['nll_at_Tstar_easy']:.4f} / {verdict['nll_at_Tstar_hard']:.4f}",
        f"- **ACC@T* easy/hard:** {verdict['acc_at_Tstar_easy']:.4f} / {verdict['acc_at_Tstar_hard']:.4f}",
        "",
        "## Depth curves (NLL)",
        "",
        "| depth | easy NLL | easy ACC | hard NLL | hard ACC |",
        "|------:|---------:|---------:|---------:|---------:|",
    ]
    for d in DEPTHS:
        lines.append(
            f"| {d} | {easy_curve[d]['nll']:.4f} | {easy_curve[d]['acc']:.4f} | "
            f"{hard_curve[d]['nll']:.4f} | {hard_curve[d]['acc']:.4f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Fixed-depth sweep loops **1–16**, **no controller**.",
        "- Toy: prelude/recurrent/coda spirit, width 64, numpy CPU, **no train**.",
        "- ACT + Convergence threshold wired as **stateless per-item stubs** (dry callable).",
        "- Affinity / C₃ / hum **untouched**. No PREREG.sha256.",
        "",
    ]
    if receipt.get("fail_reasons"):
        lines.append("## Fail reasons")
        lines.append("")
        for r in receipt["fail_reasons"]:
            lines.append(f"- {r}")
        lines.append("")
    md_path = OUT_DIR / "T1_SWEEP_RECEIPT.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({k: receipt[k] for k in ("status", "H1", "H3", "stubs_ACT_Convergence", "verdict")}, indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0 if t1_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
