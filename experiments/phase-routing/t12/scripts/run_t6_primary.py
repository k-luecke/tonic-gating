#!/usr/bin/env python3
"""T12 T6 — one primary × 5 seeds under frozen PREREG.

Reuse T4 measures harness scaled up (width 128, all 5 seeds, full arm set).
Affinity / C₃ / hum UNTOUCHED. No T12.2 unless gate defect. No closing verdict.

Usage:
  cd /workspace/scp/experiments/phase-routing/t12
  python3 scripts/run_t6_primary.py

Receipts: results/t6/ (per-seed + aggregate)
"""
from __future__ import annotations

import hashlib
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
from controllers.calibration import LOCK, within_c_bar  # noqa: E402
from controllers.convergence_stub import ConvergenceThresholdStub  # noqa: E402
from controllers.coordinator_stub import CoordinatorRegimeStub  # noqa: E402
from controllers.coordinator_traj_stub import CoordinatorRegimeTrajStub  # noqa: E402
from controllers.fixed_stub import FixedStub  # noqa: E402
from controllers.oracle_stub import OracleStub  # noqa: E402
from controllers.stateless_regime_stub import StatelessRegimeStub  # noqa: E402
from generators.easy_hard import build_easy_hard  # noqa: E402
from harness.parity import DRAFT_SEEDS, INIT_FAMILY, ParityHarness  # noqa: E402
from measures.emit import (  # noqa: E402
    eval_at_depth,
    overthinking_curve,
    silenced_delta,
    summarize_batch,
)

PT = ZoneInfo("America/Chicago")
OUT_DIR = T12 / "results" / "t6"
PREREG_PATH = T12 / "PREREG.md"
PREREG_SHA_FILE = T12 / "PREREG.sha256"
LOCKED_PREREG_SHA = "bf757ee4fc01e5c40b45f035150e7b716d12f7d8b30d745b9a7669f79bbf307c"
AFFINITY_SHA = "7dae877a3975c1ba4ca264f03f572d6749cce39a114ea3adbcffc010c7852c50"
AFFINITY_REL = "imports/hum_measure/affinity_inputs.py"

# Primary scale (PREREG §3.2 / §3.7) — smoke used width 64 + 2 seeds
PRIMARY_SEEDS: List[int] = list(DRAFT_SEEDS)  # T8/T11 family locked at harness
WIDTH = 128
N_PER_TASK = 8
SEQ_LEN = 48
PRIMARY_ARM = "coordinator_regime"

ARMS_PRIMARY = (
    "fixed_t7",
    "act",
    "convergence",
    "coordinator_regime",
    "coordinator_regime_traj",
    "oracle",
    "stateless_regime",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def pin_checks() -> Dict[str, Any]:
    prereg_sha = sha256_file(PREREG_PATH)
    pin_txt = PREREG_SHA_FILE.read_text(encoding="utf-8").strip().split()[0]
    aff_path = T12 / AFFINITY_REL
    aff_sha = sha256_file(aff_path) if aff_path.is_file() else None
    ok = (
        prereg_sha == LOCKED_PREREG_SHA
        and pin_txt == LOCKED_PREREG_SHA
        and aff_sha == AFFINITY_SHA
    )
    return {
        "pass": bool(ok),
        "prereg_sha256": prereg_sha,
        "prereg_sha256_locked": LOCKED_PREREG_SHA,
        "prereg_sha_file": pin_txt,
        "affinity_path": str(aff_path),
        "affinity_sha256": aff_sha,
        "affinity_sha256_locked": AFFINITY_SHA,
        "affinity_untouched": aff_sha == AFFINITY_SHA,
        "gate_defect": not ok,
        "note": "Gate defect → T12.2 only if freeze pin broken; Affinity/C3/hum must stay untouched",
    }


def regression_lock_status() -> Dict[str, Any]:
    script = T12 / "scripts" / "check_t8_rerun_persistence.py"
    if not script.is_file():
        return {"available": False, "pass": None, "detail": "script missing"}
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(T12),
            capture_output=True,
            text=True,
            timeout=90,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return {
            "available": True,
            "pass": proc.returncode == 0,
            "returncode": proc.returncode,
            "detail_tail": out[-800:],
            "note": "T8 shift ΔL + hum persistence regression lock",
        }
    except Exception as e:  # noqa: BLE001
        return {"available": True, "pass": False, "detail": repr(e)}


def _arm_map():
    return {
        "fixed_t7": FixedStub(LOCK.fixed_t),
        "act": ACTStub(),
        "convergence": ConvergenceThresholdStub(),
        "coordinator_regime": CoordinatorRegimeStub(),
        "coordinator_regime_traj": CoordinatorRegimeTrajStub(),
        "oracle": OracleStub(),
        "stateless_regime": StatelessRegimeStub(),
    }


def run_arm_on_clean(seed: int) -> Dict[str, Any]:
    """Evaluate all primary arms on clean easy/hard."""
    data = build_easy_hard(seed=seed, n_per_task=N_PER_TASK, seq_len=SEQ_LEN)
    arms = _arm_map()
    depths_by_arm_reg: Dict[str, Dict[str, List[int]]] = {
        a: {"easy": [], "hard": []} for a in arms
    }
    acc_by_arm: Dict[str, List[float]] = {a: [] for a in arms}
    nll_by_arm: Dict[str, List[float]] = {a: [] for a in arms}
    per_arm: Dict[str, Any] = {}

    for diff in ("easy", "hard"):
        seqs = data[diff]["sequences"]
        for arm_name, ctrl in arms.items():
            dec = ctrl.decide(difficulty=diff, silenced=False, difficulty_true=diff)
            depth = int(dec.depth)
            depths_by_arm_reg[arm_name][diff].append(depth)
            metrics = eval_at_depth(seqs, diff, depth, width=WIDTH, seed=seed)
            acc_by_arm[arm_name].append(metrics["acc"])
            nll_by_arm[arm_name].append(metrics["nll"])

    for arm_name in arms:
        dlists = depths_by_arm_reg[arm_name]
        all_d = dlists["easy"] + dlists["hard"]
        acc = float(np.mean(acc_by_arm[arm_name]))
        nll = float(np.mean(nll_by_arm[arm_name]))
        c_bar = float(np.mean(all_d))
        per_arm[arm_name] = {
            "ACC": acc,
            "NLL": nll,
            "C_bar": c_bar,
            "LPC": float(acc / max(c_bar, 1e-9)),
            "T_b_easy": float(np.mean(dlists["easy"])),
            "T_b_hard": float(np.mean(dlists["hard"])),
            "depths_by_regime": dlists,
            "within_c_bar_tol": within_c_bar(c_bar),
        }
    return {
        "per_arm": per_arm,
        "coord_depths": depths_by_arm_reg[PRIMARY_ARM],
        "data_meta": {"n_easy": len(data["easy"]["sequences"]), "n_hard": len(data["hard"]["sequences"])},
    }


def run_silence_sd(seed: int) -> Dict[str, Any]:
    """H4 SD on silence cell: coordinator vs stateless (both gated)."""
    harness = ParityHarness(n_per_task=N_PER_TASK, seq_len=SEQ_LEN)
    batches = [
        b
        for b in harness.iter_arm_batches(seed, cells=["silence"])
        if b.arm in ("coordinator_regime", "stateless_regime")
    ]
    # ParityHarness ARMS_DEFAULT may omit traj/oracle — filter by name match on available
    # Rebuild decide using our stubs with silence kwargs from batches that exist,
    # or synthesize silence cells if harness arms differ.
    coord = CoordinatorRegimeStub()
    stateless = StatelessRegimeStub()
    coord_accs, stat_accs = [], []
    coord_cs, stat_cs = [], []

    if not batches:
        # Fall back: build silence batches via decide_kwargs manually
        from controllers.silencing import SilencingGate, apply_silence_to_decide_kwargs
        from generators.disturbance import build_grid_batch

        grid = build_grid_batch(seed, cells=["silence"], n_per_task=N_PER_TASK, seq_len=SEQ_LEN)
        for diff in ("easy", "hard"):
            payload = grid[diff]["silence"]
            gate = SilencingGate(mode="zero")
            ctrl_in = gate.prepare(difficulty=diff)
            kwargs = apply_silence_to_decide_kwargs(ctrl_in)
            for arm_name, ctrl in (("coordinator_regime", coord), ("stateless_regime", stateless)):
                dec = ctrl.decide(**kwargs)
                depth = int(dec.depth)
                m = eval_at_depth(payload["sequences"], diff, depth, width=WIDTH, seed=seed)
                if arm_name == "coordinator_regime":
                    coord_accs.append(m["acc"])
                    coord_cs.append(depth)
                else:
                    stat_accs.append(m["acc"])
                    stat_cs.append(depth)
    else:
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


def run_seed(seed: int) -> Dict[str, Any]:
    clean = run_arm_on_clean(seed)
    per_arm = clean["per_arm"]
    coord_depths = clean["coord_depths"]
    primary = per_arm[PRIMARY_ARM]
    data = build_easy_hard(seed=seed, n_per_task=N_PER_TASK, seq_len=SEQ_LEN)
    ot = overthinking_curve(
        {"easy": data["easy"]["sequences"], "hard": data["hard"]["sequences"]},
        depths=range(1, 13),
        width=WIDTH,
        seed=seed,
    )
    sd = run_silence_sd(seed)
    primary_depths = coord_depths["easy"] + coord_depths["hard"]
    bundle = summarize_batch(
        acc=primary["ACC"],
        nll=primary["NLL"],
        depths=primary_depths,
        depths_by_regime=coord_depths,
        ot=ot,
        sd=sd,
        t_b_by_example=primary_depths,
    )
    return {
        "seed": seed,
        "primary_arm": PRIMARY_ARM,
        "measures": bundle.as_dict(),
        "per_arm": per_arm,
        "calibration_ref": {
            "C_bar_target": LOCK.c_bar_target,
            "c_bar_tol": LOCK.c_bar_tol,
            "primary_C_bar": bundle.C_bar,
            "within_tol": within_c_bar(bundle.C_bar),
            "fixed_t": LOCK.fixed_t,
            "act_bp_centers": [LOCK.act_easy_center, LOCK.act_hard_center],
        },
    }


def aggregate(seed_receipts: List[Dict[str, Any]]) -> Dict[str, Any]:
    arms = list(seed_receipts[0]["per_arm"].keys())
    agg_arms: Dict[str, Any] = {}
    for arm in arms:
        accs = [r["per_arm"][arm]["ACC"] for r in seed_receipts]
        cs = [r["per_arm"][arm]["C_bar"] for r in seed_receipts]
        lpcs = [r["per_arm"][arm]["LPC"] for r in seed_receipts]
        nlls = [r["per_arm"][arm]["NLL"] for r in seed_receipts]
        te = [r["per_arm"][arm]["T_b_easy"] for r in seed_receipts]
        th = [r["per_arm"][arm]["T_b_hard"] for r in seed_receipts]
        agg_arms[arm] = {
            "ACC_mean": float(np.mean(accs)),
            "ACC_std": float(np.std(accs)),
            "C_bar_mean": float(np.mean(cs)),
            "C_bar_std": float(np.std(cs)),
            "LPC_mean": float(np.mean(lpcs)),
            "LPC_std": float(np.std(lpcs)),
            "NLL_mean": float(np.mean(nlls)),
            "T_b_easy_mean": float(np.mean(te)),
            "T_b_hard_mean": float(np.mean(th)),
            "n_seeds": len(seed_receipts),
            "per_seed": {
                "ACC": accs,
                "C_bar": cs,
                "LPC": lpcs,
            },
        }

    sds = [r["measures"]["SD"] for r in seed_receipts]
    sd_agg = {
        "delta_acc_mean": float(np.mean([s["delta_acc"] for s in sds])),
        "delta_acc_std": float(np.std([s["delta_acc"] for s in sds])),
        "coord_beats_stateless_acc_n": int(sum(1 for s in sds if s.get("coord_beats_stateless_acc"))),
        "per_seed": sds,
    }
    bps = [r["measures"]["BP"].get("separation", {}) for r in seed_receipts]
    bp_agg = {
        "easy_mean": float(np.mean([b.get("easy_mean", float("nan")) for b in bps])),
        "hard_mean": float(np.mean([b.get("hard_mean", float("nan")) for b in bps])),
        "gap_mean": float(np.mean([b.get("gap", float("nan")) for b in bps])),
        "separates_n": int(sum(1 for b in bps if b.get("separates"))),
        "n_seeds": len(bps),
    }
    # OT summary across seeds
    ot_easy_deg = [
        bool(r["measures"]["OT_T"]["by_regime"]["easy"].get("_ot", {}).get("degrades_past"))
        for r in seed_receipts
    ]
    ot_hard_deg = [
        bool(r["measures"]["OT_T"]["by_regime"]["hard"].get("_ot", {}).get("degrades_past"))
        for r in seed_receipts
    ]
    return {
        "per_arm": agg_arms,
        "primary_arm": PRIMARY_ARM,
        "primary": agg_arms[PRIMARY_ARM],
        "SD": sd_agg,
        "BP_separation": bp_agg,
        "OT_summary": {
            "easy_degrades_past_n": int(sum(ot_easy_deg)),
            "hard_degrades_past_n": int(sum(ot_hard_deg)),
            "n_seeds": len(seed_receipts),
        },
        "comparisons_numeric": {
            "H1_coord_vs_fixed": {
                "dACC": agg_arms[PRIMARY_ARM]["ACC_mean"] - agg_arms["fixed_t7"]["ACC_mean"],
                "dLPC": agg_arms[PRIMARY_ARM]["LPC_mean"] - agg_arms["fixed_t7"]["LPC_mean"],
                "coord_C": agg_arms[PRIMARY_ARM]["C_bar_mean"],
                "fixed_C": agg_arms["fixed_t7"]["C_bar_mean"],
            },
            "H2_traj_vs_act": {
                "dACC": agg_arms["coordinator_regime_traj"]["ACC_mean"] - agg_arms["act"]["ACC_mean"],
                "dLPC": agg_arms["coordinator_regime_traj"]["LPC_mean"] - agg_arms["act"]["LPC_mean"],
            },
            "H2_traj_vs_convergence": {
                "dACC": agg_arms["coordinator_regime_traj"]["ACC_mean"]
                - agg_arms["convergence"]["ACC_mean"],
                "dLPC": agg_arms["coordinator_regime_traj"]["LPC_mean"]
                - agg_arms["convergence"]["LPC_mean"],
            },
            "note": "Numeric deltas only — no support/null verdict (T7–T8 later)",
        },
    }


def write_md(agg: Dict[str, Any], pins: Dict[str, Any], seeds: List[int], pt: str, reglock: Dict[str, Any]) -> str:
    pa = agg["per_arm"]
    lines = [
        "# T12 T6 Primary Receipt",
        "",
        f"- **Ticket:** T6 (one primary × 5 seeds)",
        f"- **PT:** {pt}",
        f"- **PREREG.sha256:** `{LOCKED_PREREG_SHA}`",
        f"- **Seeds:** {seeds} (T8/T11 family via `harness.parity.DRAFT_SEEDS`; FREEZE_META had no seed list)",
        f"- **Width / n_per_task / seq_len:** {WIDTH} / {N_PER_TASK} / {SEQ_LEN}",
        f"- **Primary arm:** `{PRIMARY_ARM}`",
        f"- **Affinity / C₃ / hum:** UNTOUCHED (Affinity sha `{AFFINITY_SHA[:8]}…` match={pins['affinity_untouched']})",
        f"- **Gate defect:** {pins['gate_defect']} (→ T12.2 only if true)",
        f"- **Regression lock pass:** {reglock.get('pass')}",
        "",
        "## Aggregate ACC / C̄ / LPC by arm",
        "",
        "| Arm | ACC̄ | C̄ | LPC̄ | T_b easy/hard |",
        "|-----|-----:|---:|-----:|--------------:|",
    ]
    for arm, row in sorted(pa.items()):
        lines.append(
            f"| `{arm}` | {row['ACC_mean']:.4f} | {row['C_bar_mean']:.4f} | "
            f"{row['LPC_mean']:.4f} | {row['T_b_easy_mean']:.1f}/{row['T_b_hard_mean']:.1f} |"
        )
    prim = agg["primary"]
    lines += [
        "",
        "## Primary arm vs Fixed / ACT / baselines (numbers only)",
        "",
        f"- coordinator_regime: ACC={prim['ACC_mean']:.4f} C̄={prim['C_bar_mean']:.4f} LPC={prim['LPC_mean']:.4f}",
        f"- vs fixed_t7 ΔACC={agg['comparisons_numeric']['H1_coord_vs_fixed']['dACC']:+.4f} "
        f"ΔLPC={agg['comparisons_numeric']['H1_coord_vs_fixed']['dLPC']:+.4f}",
        f"- regime+traj vs ACT ΔACC={agg['comparisons_numeric']['H2_traj_vs_act']['dACC']:+.4f} "
        f"ΔLPC={agg['comparisons_numeric']['H2_traj_vs_act']['dLPC']:+.4f}",
        f"- regime+traj vs CONV ΔACC={agg['comparisons_numeric']['H2_traj_vs_convergence']['dACC']:+.4f} "
        f"ΔLPC={agg['comparisons_numeric']['H2_traj_vs_convergence']['dLPC']:+.4f}",
        f"- BP separation: easy={agg['BP_separation']['easy_mean']:.2f} hard={agg['BP_separation']['hard_mean']:.2f} "
        f"gap={agg['BP_separation']['gap_mean']:.2f} (separates_n={agg['BP_separation']['separates_n']}/{agg['BP_separation']['n_seeds']})",
        f"- SD ΔACC mean={agg['SD']['delta_acc_mean']:+.4f} (coord_beats_n={agg['SD']['coord_beats_stateless_acc_n']}/{len(seeds)})",
        f"- OT easy degrades_n={agg['OT_summary']['easy_degrades_past_n']} hard degrades_n={agg['OT_summary']['hard_degrades_past_n']}",
        "",
        "## Notes",
        "",
        "- No closing verdict (T7–T8 later). Fixed sweep = OT(T) measurement.",
        "- Option-1 locks: C̄=7±0.25; ACT/BP 4/10; Fixed T=7.",
        "- Affinity/C₃/hum **not** retuned. Parent T11.* PREREGs untouched. t12b untouched.",
        "",
        "## Re-run",
        "",
        "```bash",
        "cd /workspace/scp/experiments/phase-routing/t12",
        "python3 scripts/run_t6_primary.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pins = pin_checks()
    if pins["gate_defect"]:
        # Still emit a fail receipt; stopping rule says T12.2 only on freeze gate defect
        fail = {
            "ticket": "T6",
            "status": "GATE_DEFECT",
            "pins": pins,
            "pt": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT"),
        }
        (OUT_DIR / "T6_PRIMARY_AGGREGATE.json").write_text(
            json.dumps(fail, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(fail, indent=2))
        return 2

    seed_receipts: List[Dict[str, Any]] = []
    for seed in PRIMARY_SEEDS:
        print(f"[T6] seed {seed} …", flush=True)
        rec = run_seed(seed)
        seed_path = OUT_DIR / f"T6_SEED_{seed}.json"
        seed_path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seed_receipts.append(rec)
        print(
            f"  ACC={rec['measures']['ACC']:.4f} C̄={rec['measures']['C_bar']:.4f} "
            f"LPC={rec['measures']['LPC']:.4f} → {seed_path.name}",
            flush=True,
        )

    print("[T6] regression lock …", flush=True)
    reglock = regression_lock_status()
    agg = aggregate(seed_receipts)
    pt = datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT")

    receipt: Dict[str, Any] = {
        "ticket": "T6",
        "status": "COMPLETE",
        "pt": pt,
        "prereg_sha256": LOCKED_PREREG_SHA,
        "seeds": PRIMARY_SEEDS,
        "seeds_source": "harness.parity.DRAFT_SEEDS (T8/T11 family); FREEZE_META listed none",
        "width": WIDTH,
        "n_per_task": N_PER_TASK,
        "seq_len": SEQ_LEN,
        "init_family": INIT_FAMILY,
        "arms": list(ARMS_PRIMARY),
        "primary_arm": PRIMARY_ARM,
        "aggregate": agg,
        "pins": pins,
        "affinity_c3_hum": "UNTOUCHED",
        "regression_locks": reglock,
        "option1_locks": {
            "C_bar": "7±0.25",
            "fixed_t": 7,
            "act_bp_centers": [4, 10],
        },
        "stopping": {
            "primary_runs": 1,
            "seeds": 5,
            "t12_2": False,
            "gate_defect": False,
            "note": "No T12.2 — pins OK; Affinity untouched",
        },
        "verdict": None,
        "verdict_note": "No closing verdict beyond numbers (T7–T8 later)",
        "train": False,
        "t12b_touched": False,
        "run_cmd": "python3 scripts/run_t6_primary.py",
        "per_seed_receipts": [f"T6_SEED_{s}.json" for s in PRIMARY_SEEDS],
    }

    json_path = OUT_DIR / "T6_PRIMARY_AGGREGATE.json"
    json_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = OUT_DIR / "T6_PRIMARY_AGGREGATE.md"
    md_path.write_text(write_md(agg, pins, PRIMARY_SEEDS, pt, reglock), encoding="utf-8")

    # compact headline
    headline = {
        "status": receipt["status"],
        "seeds": PRIMARY_SEEDS,
        "primary": agg["primary"],
        "vs_fixed_dACC": agg["comparisons_numeric"]["H1_coord_vs_fixed"]["dACC"],
        "vs_fixed_dLPC": agg["comparisons_numeric"]["H1_coord_vs_fixed"]["dLPC"],
        "BP": agg["BP_separation"],
        "SD_delta_acc_mean": agg["SD"]["delta_acc_mean"],
        "gate_defect": False,
        "affinity_untouched": True,
        "prereg_sha256": LOCKED_PREREG_SHA,
    }
    print(json.dumps(headline, indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
