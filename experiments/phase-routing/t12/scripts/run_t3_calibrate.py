#!/usr/bin/env python3
"""T12 T3 calibration — lock ε / ACT priors / Convergence threshold; C̄=7±0.25.

Locks match T1 T* budgets (easy=4, hard=10). Soft-hard-to-8 removed;
equal-mix mean = 7 → Fixed T=7.

Usage:
  cd /workspace/scp/experiments/phase-routing/t12
  python3 scripts/run_t3_calibrate.py

Receipt: results/t3/T3_CALIBRATION_RECEIPT.json (+ .md)
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
from controllers.calibration import LOCK, within_c_bar  # noqa: E402
from controllers.convergence_stub import ConvergenceThresholdStub  # noqa: E402
from controllers.coordinator_stub import CoordinatorRegimeStub  # noqa: E402
from controllers.fixed_stub import FixedStub  # noqa: E402
from harness.parity import DRAFT_SEEDS  # noqa: E402

PT = ZoneInfo("America/Chicago")
OUT_DIR = T12 / "results" / "t3"
SMOKE_SEEDS = DRAFT_SEEDS[:2]


def depths_for_arm(arm_name: str, difficulties: List[str]) -> Dict[str, List[int]]:
    if arm_name == "fixed_t7":
        ctrl: Any = FixedStub(LOCK.fixed_t)
    elif arm_name == "act":
        ctrl = ACTStub()
    elif arm_name == "convergence":
        ctrl = ConvergenceThresholdStub()
    elif arm_name == "coordinator_regime":
        ctrl = CoordinatorRegimeStub()
    else:
        raise ValueError(arm_name)
    out: Dict[str, List[int]] = {d: [] for d in difficulties}
    for diff in difficulties:
        for _ in range(32):  # replicate per-item decisions (deterministic → same)
            dec = ctrl.decide(difficulty=diff, silenced=False)
            out[diff].append(int(dec.depth))
    return out


def mean_c(depths_by_reg: Dict[str, List[int]]) -> float:
    all_d = [d for v in depths_by_reg.values() for d in v]
    return float(np.mean(all_d)) if all_d else float("nan")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    difficulties = ["easy", "hard"]
    arms = ["fixed_t7", "act", "convergence", "coordinator_regime"]
    profiles = {}
    c_bars = {}
    for a in arms:
        prof = depths_for_arm(a, difficulties)
        profiles[a] = {
            reg: {"mean": float(np.mean(ds)), "depth": int(ds[0])} for reg, ds in prof.items()
        }
        c_bars[a] = mean_c(prof)

    # Primary calibration check: ACT + coordinator + fixed on equal-mix
    primary_c = float(np.mean([c_bars["act"], c_bars["coordinator_regime"], c_bars["fixed_t7"]]))
    # Also each arm that should hit ~7
    act_ok = within_c_bar(c_bars["act"])
    coord_ok = within_c_bar(c_bars["coordinator_regime"])
    fixed_ok = within_c_bar(c_bars["fixed_t7"])
    conv_ok = within_c_bar(c_bars["convergence"])  # soft — may be near 7 via tau_default mix

    # Separation retained
    def sep(arm: str) -> Dict[str, Any]:
        e = profiles[arm]["easy"]["mean"]
        h = profiles[arm]["hard"]["mean"]
        return {
            "easy": e,
            "hard": h,
            "gap": h - e,
            "separates": (h - e) >= 2.0,
            "easy_near_4": abs(e - 4.0) <= 1.5,
            "hard_near_10": abs(h - 10.0) <= 1.5,
        }

    seps = {a: sep(a) for a in ("act", "convergence", "coordinator_regime")}
    sep_ok = all(seps[a]["separates"] for a in seps)

    t3_pass = bool(act_ok and coord_ok and fixed_ok and sep_ok)

    receipt: Dict[str, Any] = {
        "ticket": "T3",
        "status": "PASS" if t3_pass else "FAIL",
        "pt": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT"),
        "target_C_bar": LOCK.c_bar_target,
        "tol": LOCK.c_bar_tol,
        "locks": LOCK.as_dict(),
        "C_bar_by_arm": c_bars,
        "primary_mean_C_bar": primary_c,
        "primary_within_tol": within_c_bar(primary_c),
        "profiles": profiles,
        "separation": seps,
        "checks": {
            "act_C_bar": act_ok,
            "coord_C_bar": coord_ok,
            "fixed_C_bar": fixed_ok,
            "conv_C_bar": conv_ok,
            "easy_hard_separation_retained": sep_ok,
        },
        "t1_reference": {
            "T_star_easy": LOCK.t_star_easy,
            "T_star_hard": LOCK.t_star_hard,
            "gap": LOCK.t_star_gap,
        },
        "smoke_seeds_documented": SMOKE_SEEDS,
        "affinity_c3_hum": "UNTOUCHED",
        "prereg": "DRAFT — no PREREG.sha256 emit; locks recorded in receipt only",
        "train": False,
        "freeze": False,
    }

    json_path = OUT_DIR / "T3_CALIBRATION_RECEIPT.json"
    json_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# T12 T3 Calibration Receipt",
        "",
        f"- **Status:** **{receipt['status']}**",
        f"- **PT:** {receipt['pt']}",
        f"- **Target C̄:** {LOCK.c_bar_target} ± {LOCK.c_bar_tol}",
        f"- **ACT C̄:** {c_bars['act']:.4f} ({'PASS' if act_ok else 'FAIL'})",
        f"- **Coordinator C̄:** {c_bars['coordinator_regime']:.4f} ({'PASS' if coord_ok else 'FAIL'})",
        f"- **Fixed T=7 C̄:** {c_bars['fixed_t7']:.4f} ({'PASS' if fixed_ok else 'FAIL'})",
        f"- **Convergence C̄:** {c_bars['convergence']:.4f} (informational; within_tol={conv_ok})",
        f"- **Separation retained:** {'PASS' if sep_ok else 'FAIL'}",
        "",
        "## Locked values",
        "",
        f"| Param | Value |",
        f"|-------|------:|",
        f"| ACT prior_mean | {LOCK.act_prior_mean} |",
        f"| ACT easy_center | {LOCK.act_easy_center} |",
        f"| ACT hard_center | {LOCK.act_hard_center} |",
        f"| CONV ε | {LOCK.conv_eps} |",
        f"| CONV τ_easy | {LOCK.conv_tau_easy} |",
        f"| CONV τ_hard | {LOCK.conv_tau_hard} |",
        f"| CONV τ_default | {LOCK.conv_tau_default} |",
        f"| Fixed T | {LOCK.fixed_t} |",
        "",
        "## Budget profiles (mean depth)",
        "",
        "| Arm | easy | hard | gap |",
        "|-----|-----:|-----:|----:|",
    ]
    for a in arms:
        e = profiles[a]["easy"]["mean"]
        h = profiles[a]["hard"]["mean"]
        lines.append(f"| {a} | {e:.2f} | {h:.2f} | {h-e:.2f} |")
    lines += [
        "",
        "## Notes",
        "",
        "- ACT/BP centers = T1 T* = 4/10 (soft-hard-to-8 removed); equal-mix C̄≈7.",
        "- Fixed T=7 matched-mean baseline. T1 T*_easy=4 / T*_hard=10 are the locks.",
        "- Affinity / C₃ / hum **untouched**. No PREREG.sha256.",
        "",
    ]
    md_path = OUT_DIR / "T3_CALIBRATION_RECEIPT.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": receipt["status"],
                "C_bar_by_arm": c_bars,
                "separation": seps,
                "locks_eps": LOCK.conv_eps,
                "locks_act_centers": [LOCK.act_easy_center, LOCK.act_hard_center],
            },
            indent=2,
        )
    )
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0 if t3_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
