#!/usr/bin/env python3
"""T12 T4 — emit all §4.1 measures on a smoke batch.

Measures: ACC, C̄, LPC, OT(T), BP, SD, T_b (+ regression lock status).

Usage:
  cd /workspace/scp/experiments/phase-routing/t12
  python3 scripts/run_t4_measures.py

Receipt: results/t4/T4_MEASURES_RECEIPT.json (+ .md)
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

import numpy as np

T12 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T12))

from controllers.act_stub import ACTStub  # noqa: E402
from controllers.calibration import LOCK  # noqa: E402
from controllers.coordinator_stub import CoordinatorRegimeStub  # noqa: E402
from controllers.fixed_stub import FixedStub  # noqa: E402
from controllers.silencing import SilencingGate, apply_silence_to_decide_kwargs  # noqa: E402
from controllers.stateless_regime_stub import StatelessRegimeStub  # noqa: E402
from generators.easy_hard import build_easy_hard  # noqa: E402
from harness.parity import DRAFT_SEEDS, ParityHarness  # noqa: E402
from measures.emit import (  # noqa: E402
    eval_at_depth,
    overthinking_curve,
    silenced_delta,
    summarize_batch,
)

PT = ZoneInfo("America/Chicago")
OUT_DIR = T12 / "results" / "t4"
SMOKE_SEEDS = DRAFT_SEEDS[:2]
WIDTH = 64
N_PER_TASK = 6
SEQ_LEN = 40


def regression_lock_status() -> Dict[str, Any]:
    """Run T8 shift persistence check if available; report status."""
    script = T12 / "scripts" / "check_t8_rerun_persistence.py"
    if not script.is_file():
        return {"available": False, "pass": None, "detail": "script missing"}
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(T12),
            capture_output=True,
            text=True,
            timeout=60,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        # parse last JSON-ish or PASS/FAIL lines
        passed = proc.returncode == 0
        return {
            "available": True,
            "pass": passed,
            "returncode": proc.returncode,
            "detail_tail": out[-800:],
            "note": "T8 shift ΔL + hum persistence regression lock",
        }
    except Exception as e:  # noqa: BLE001
        return {"available": True, "pass": False, "detail": repr(e)}


def run_arm_on_clean(
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, List[int]], Dict[str, List[float]], Dict[str, List[float]]]:
    """Evaluate arms on clean easy/hard; collect depths, acc, nll."""
    data = build_easy_hard(seed=seed, n_per_task=N_PER_TASK, seq_len=SEQ_LEN)
    arms = {
        "fixed_t7": FixedStub(LOCK.fixed_t),
        "act": ACTStub(),
        "coordinator_regime": CoordinatorRegimeStub(),
        "stateless_regime": StatelessRegimeStub(),
    }
    depths_by_arm_reg: Dict[str, Dict[str, List[int]]] = {a: {"easy": [], "hard": []} for a in arms}
    acc_by_arm: Dict[str, List[float]] = {a: [] for a in arms}
    nll_by_arm: Dict[str, List[float]] = {a: [] for a in arms}
    per_arm_summary: Dict[str, Any] = {}

    for diff in ("easy", "hard"):
        seqs = data[diff]["sequences"]
        for arm_name, ctrl in arms.items():
            dec = ctrl.decide(difficulty=diff, silenced=False)
            depth = int(dec.depth)
            depths_by_arm_reg[arm_name][diff].append(depth)
            metrics = eval_at_depth(seqs, diff, depth, width=WIDTH, seed=seed)
            acc_by_arm[arm_name].append(metrics["acc"])
            nll_by_arm[arm_name].append(metrics["nll"])

    for arm_name in arms:
        dlists = depths_by_arm_reg[arm_name]
        all_d = dlists["easy"] + dlists["hard"]
        per_arm_summary[arm_name] = {
            "ACC": float(np.mean(acc_by_arm[arm_name])),
            "NLL": float(np.mean(nll_by_arm[arm_name])),
            "C_bar": float(np.mean(all_d)),
            "T_b_easy": float(np.mean(dlists["easy"])),
            "T_b_hard": float(np.mean(dlists["hard"])),
            "depths_by_regime": dlists,
        }
    return per_arm_summary, depths_by_arm_reg["coordinator_regime"], acc_by_arm, nll_by_arm


def run_silence_sd(seed: int) -> Dict[str, Any]:
    """H4 SD on silence cell: coordinator vs stateless (both gated)."""
    harness = ParityHarness(n_per_task=N_PER_TASK, seq_len=SEQ_LEN)
    batches = [
        b
        for b in harness.iter_arm_batches(seed, cells=["silence"])
        if b.arm in ("coordinator_regime", "stateless_regime")
    ]
    coord = CoordinatorRegimeStub()
    stateless = StatelessRegimeStub()
    coord_accs, stat_accs = [], []
    coord_cs, stat_cs = [], []
    for b in batches:
        ctrl = coord if b.arm == "coordinator_regime" else stateless
        dec = ctrl.decide(**b.decide_kwargs)
        depth = int(dec.depth)
        m = eval_at_depth(b.sequences, b.difficulty, depth, width=WIDTH, seed=seed)
        if b.arm == "coordinator_regime":
            coord_accs.append(m["acc"])
            coord_cs.append(depth)
        else:
            stat_accs.append(m["acc"])
            stat_cs.append(depth)
    return silenced_delta(
        float(np.mean(coord_accs)) if coord_accs else 0.0,
        float(np.mean(stat_accs)) if stat_accs else 0.0,
        float(np.mean(coord_cs)) if coord_cs else 0.0,
        float(np.mean(stat_cs)) if stat_cs else 0.0,
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = SMOKE_SEEDS[0]
    per_arm, coord_depths, acc_by_arm, nll_by_arm = run_arm_on_clean(seed)

    # Primary arm for headline measures: coordinator_regime
    primary = "coordinator_regime"
    primary_acc = per_arm[primary]["ACC"]
    primary_nll = per_arm[primary]["NLL"]
    primary_depths = coord_depths["easy"] + coord_depths["hard"]

    data = build_easy_hard(seed=seed, n_per_task=N_PER_TASK, seq_len=SEQ_LEN)
    ot = overthinking_curve(
        {"easy": data["easy"]["sequences"], "hard": data["hard"]["sequences"]},
        depths=range(1, 13),
        width=WIDTH,
        seed=seed,
    )
    sd = run_silence_sd(seed)
    bundle = summarize_batch(
        acc=primary_acc,
        nll=primary_nll,
        depths=primary_depths,
        depths_by_regime=coord_depths,
        ot=ot,
        sd=sd,
        t_b_by_example=primary_depths,
    )
    reglock = regression_lock_status()

    # Separation check on BP
    sep = bundle.BP.get("separation", {})
    sep_ok = bool(sep.get("separates"))
    measures_present = all(
        x is not None
        for x in (
            bundle.ACC,
            bundle.C_bar,
            bundle.LPC,
            bundle.OT_T,
            bundle.BP,
            bundle.SD,
            bundle.T_b,
        )
    )
    t4_pass = bool(measures_present and sep_ok and np.isfinite(bundle.C_bar))

    receipt: Dict[str, Any] = {
        "ticket": "T4",
        "status": "PASS" if t4_pass else "FAIL",
        "pt": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT"),
        "smoke_seeds": SMOKE_SEEDS,
        "seed_used": seed,
        "width": WIDTH,
        "n_per_task": N_PER_TASK,
        "seq_len": SEQ_LEN,
        "primary_arm": primary,
        "measures": bundle.as_dict(),
        "per_arm": per_arm,
        "regression_locks": reglock,
        "calibration_ref": {
            "C_bar_target": LOCK.c_bar_target,
            "primary_C_bar": bundle.C_bar,
            "within_tol": abs(bundle.C_bar - LOCK.c_bar_target) <= LOCK.c_bar_tol,
        },
        "affinity_c3_hum": "UNTOUCHED",
        "prereg": "DRAFT — no PREREG.sha256 emit",
        "train": False,
        "freeze": False,
        "run_cmd": "python3 scripts/run_t4_measures.py",
    }

    json_path = OUT_DIR / "T4_MEASURES_RECEIPT.json"
    json_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# T12 T4 Measures Receipt",
        "",
        f"- **Status:** **{receipt['status']}**",
        f"- **PT:** {receipt['pt']}",
        f"- **Seed:** {seed} (smoke seeds {SMOKE_SEEDS})",
        f"- **Primary arm:** `{primary}`",
        "",
        "## §4.1 Measures",
        "",
        f"| ID | Value |",
        f"|----|------:|",
        f"| ACC | {bundle.ACC:.4f} |",
        f"| C̄ | {bundle.C_bar:.4f} |",
        f"| LPC (ACC/C̄) | {bundle.LPC:.4f} |",
        f"| NLL | {bundle.nll:.4f} |",
        f"| T_b mean | {bundle.T_b['mean']:.4f} |",
        f"| BP easy/hard | {sep.get('easy_mean')}/{sep.get('hard_mean')} (gap={sep.get('gap')}) |",
        f"| SD ΔACC | {sd.get('delta_acc')} |",
        f"| OT easy degrades | {ot['by_regime']['easy'].get('_ot', {})} |",
        f"| OT hard | {ot['by_regime']['hard'].get('_ot', {})} |",
        "",
        f"- **Regression lock:** available={reglock.get('available')} pass={reglock.get('pass')}",
        "",
        "## Re-run",
        "",
        "```bash",
        "cd /workspace/scp/experiments/phase-routing/t12",
        "python3 scripts/run_t4_measures.py",
        "```",
        "",
        "## Notes",
        "",
        "- Smoke batch only (CPU, width 64). Affinity / C₃ / hum **untouched**.",
        "- No PREREG.sha256. No freeze.",
        "",
    ]
    md_path = OUT_DIR / "T4_MEASURES_RECEIPT.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": receipt["status"],
                "ACC": bundle.ACC,
                "C_bar": bundle.C_bar,
                "LPC": bundle.LPC,
                "BP_separation": sep,
                "SD_delta_acc": sd.get("delta_acc"),
                "regression_lock_pass": reglock.get("pass"),
            },
            indent=2,
        )
    )
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0 if t4_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
