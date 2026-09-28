#!/usr/bin/env python3
"""T12 T2 smoke — disturbance grid + silencing + parity harness.

Usage:
  cd /workspace/scp/experiments/phase-routing/t12
  python3 scripts/run_t2_smoke.py

Receipt: results/t2/T2_HARNESS_RECEIPT.json (+ .md)
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
from controllers.coordinator_stub import CoordinatorRegimeStub  # noqa: E402
from controllers.calibration import LOCK
from controllers.fixed_stub import FixedStub  # noqa: E402
from controllers.silencing import SilencingGate  # noqa: E402
from controllers.stateless_regime_stub import StatelessRegimeStub  # noqa: E402
from generators.disturbance import GRID, SHIFT_DELTA, cell_list  # noqa: E402
from harness.parity import DRAFT_SEEDS, INIT_FAMILY, ParityHarness  # noqa: E402

PT = ZoneInfo("America/Chicago")
OUT_DIR = T12 / "results" / "t2"
SMOKE_SEEDS = DRAFT_SEEDS[:2]  # documented: smoke uses 2 of 5 draft seeds


def _arm_map():
    return {
        "fixed_t7": FixedStub(LOCK.fixed_t),
        "act": ACTStub(),
        "convergence": ConvergenceThresholdStub(),
        "coordinator_regime": CoordinatorRegimeStub(),
        "stateless_regime": StatelessRegimeStub(),
    }


def silencing_checks() -> Dict[str, Any]:
    gate = SilencingGate(mode="zero")
    live = gate.prepare(difficulty="hard", force_mode="none")
    zeroed = gate.prepare(difficulty="hard", force_mode="zero")
    gate_h = SilencingGate(mode="hold")
    # first call sets hold from easy, second should keep held even if hard estimated
    held1 = gate_h.prepare(difficulty="easy", force_mode="hold")
    held2 = gate_h.prepare(difficulty="hard", force_mode="hold")
    # disclosure still present under silence
    disc_ok = (
        zeroed.disclosure_estimate is not None
        and zeroed.disclosure_estimate.hard_label == "hard"
        and zeroed.difficulty_tag is None
        and float(np.sum(np.abs(zeroed.regime_channel))) == 0.0
    )
    hold_ok = (
        held1.disclosure_estimate is not None
        and held2.disclosure_estimate is not None
        and held2.disclosure_estimate.hard_label == "hard"  # estimate still computed
        and np.allclose(held1.regime_channel, held2.regime_channel)  # held at input
        and held2.difficulty_tag is None
    )
    # controller under silence must not see difficulty
    act = ACTStub()
    d_live = act.decide(difficulty="hard", silenced=False)
    d_sil = act.decide(difficulty=None, features=zeroed.regime_channel.tolist(), silenced=True)
    coord = CoordinatorRegimeStub()
    c_sil = coord.decide(difficulty=None, features=zeroed.regime_channel.tolist(), silenced=True)
    return {
        "pass": bool(disc_ok and hold_ok and d_sil.depth == int(round(LOCK.act_prior_mean)) and c_sil.depth == int(round(LOCK.act_prior_mean))),
        "disclosure_under_zero": disc_ok,
        "hold_freezes_channel": hold_ok,
        "live_hard_act_depth": d_live.depth,
        "silenced_act_depth": d_sil.depth,
        "silenced_coord_depth": c_sil.depth,
        "live_channel": live.regime_channel.tolist(),
        "zero_channel": zeroed.regime_channel.tolist(),
        "held_channel": held2.regime_channel.tolist(),
        "note": "estimate computed for disclosure; zero/hold at controller input",
    }


def arm_decide_smoke(seed: int) -> Dict[str, Any]:
    harness = ParityHarness(n_per_task=2, seq_len=24)
    batches = harness.iter_arm_batches(seed)
    arms = _arm_map()
    rows = []
    for b in batches:
        ctrl = arms[b.arm]
        dec = ctrl.decide(**b.decide_kwargs)
        depth = int(getattr(dec, "depth"))
        rows.append(
            {
                "arm": b.arm,
                "difficulty": b.difficulty,
                "cell": b.cell,
                "depth": depth,
                "silenced": b.meta["silenced"],
                "parity_key": b.meta["parity_key"],
            }
        )
    # under silence, coordinator/act/stateless should not use live hard/easy
    sil_depths = {
        r["arm"]: r["depth"]
        for r in rows
        if r["cell"] == "silence" and r["difficulty"] == "hard"
    }
    return {
        "n_rows": len(rows),
        "sample": rows[:12],
        "silence_hard_depths": sil_depths,
        "silence_uses_prior_not_hard": all(
            sil_depths.get(a) == int(round(LOCK.act_prior_mean)) for a in ("act", "coordinator_regime", "stateless_regime")
        ),
    }


def shift_lock_path_check() -> Dict[str, Any]:
    """Confirm shift cell is wired to T8 δ=0.3 regression-lock path (params)."""
    cell = GRID["shift"]
    return {
        "pass": abs(cell.delta - SHIFT_DELTA) < 1e-12 and abs(SHIFT_DELTA - 0.3) < 1e-12,
        "delta": cell.delta,
        "ref_T8_delta": 0.3,
        "note": "T8 shift regression lock path present in disturbance grid; "
        "numeric MoE ΔL lock remains scripts/check_t8_rerun_persistence.py",
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    harness = ParityHarness(n_per_task=4, seq_len=32)
    parity_results = [harness.parity_check(s) for s in SMOKE_SEEDS]
    parity_ok = all(p["pass"] for p in parity_results)
    sil = silencing_checks()
    arms = arm_decide_smoke(SMOKE_SEEDS[0])
    shift = shift_lock_path_check()

    t2_pass = bool(
        parity_ok
        and sil["pass"]
        and arms["silence_uses_prior_not_hard"]
        and shift["pass"]
        and set(cell_list()) == {"clean", "noise", "shift", "silence"}
    )

    receipt: Dict[str, Any] = {
        "ticket": "T2",
        "status": "PASS" if t2_pass else "FAIL",
        "pt": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT"),
        "smoke_seeds": SMOKE_SEEDS,
        "draft_five_seeds": DRAFT_SEEDS,
        "init_family": INIT_FAMILY,
        "cells": cell_list(),
        "grid_notes": {k: GRID[k].note for k in GRID},
        "parity": parity_results,
        "silencing": sil,
        "arm_decide": arms,
        "shift_regression_lock_path": shift,
        "affinity_c3_hum": "UNTOUCHED",
        "prereg": "DRAFT — no PREREG.sha256 emit",
        "train": False,
        "freeze": False,
    }

    json_path = OUT_DIR / "T2_HARNESS_RECEIPT.json"
    json_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines: List[str] = [
        "# T12 T2 Harness Receipt",
        "",
        f"- **Status:** **{receipt['status']}**",
        f"- **PT:** {receipt['pt']}",
        f"- **Smoke seeds:** {SMOKE_SEEDS} (of draft five {DRAFT_SEEDS})",
        f"- **Init family:** `{INIT_FAMILY}`",
        f"- **Cells:** {', '.join(cell_list())}",
        f"- **Parity:** {'PASS' if parity_ok else 'FAIL'}",
        f"- **Silencing (zero/hold + disclosure):** {'PASS' if sil['pass'] else 'FAIL'}",
        f"- **Silence → prior (not live hard):** {arms['silence_uses_prior_not_hard']}",
        f"- **Shift lock path δ=0.3:** {'PASS' if shift['pass'] else 'FAIL'}",
        "",
        "## Silencing",
        "",
        "- Estimate **may compute** for disclosure.",
        "- **Zero/hold at controller input** — not fed into budget decision.",
        f"- Live hard ACT depth={sil['live_hard_act_depth']}; silenced ACT depth={sil['silenced_act_depth']}.",
        "",
        "## Notes",
        "",
        "- Same seed → same sequences for all arms (parity harness).",
        "- Silence cell keeps sequences clean; gate zeros/holds regime channel.",
        "- Affinity / C₃ / hum **untouched**. No PREREG.sha256.",
        "",
    ]
    md_path = OUT_DIR / "T2_HARNESS_RECEIPT.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": receipt["status"],
                "parity": parity_ok,
                "silencing": sil["pass"],
                "shift_path": shift["pass"],
                "silence_prior": arms["silence_uses_prior_not_hard"],
            },
            indent=2,
        )
    )
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0 if t2_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
