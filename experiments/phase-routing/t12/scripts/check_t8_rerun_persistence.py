#!/usr/bin/env python3
"""Confirm T8 rerun persistence artifacts + key numbers from documented paths.

Does NOT retrain. Reads frozen primary RESULTS under imports/t8_results_primary
(→ experiments/phase-routing/t8/results/primary/) and checks expected locks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

T12_ROOT = Path(__file__).resolve().parents[1]
PRIMARY = (T12_ROOT / "imports" / "t8_results_primary").resolve()
OUT_DIR = T12_ROOT / "results" / "t0"

# Reference numbers from RESULTS.md / score_summary.json (rerun 2026-09-23)
REF = {
    "prereg_sha256": "d975d96cf281badd2c0a6fc3986518b6338511cfcb7a9a9bf18be6b5e060b395",
    "hum_T_hold_seed_mean": 26.846875,
    "hum_L_seed_mean": 1.3402504453583368,
    "hum_delta_L_shift_0.3": 0.01707748666642206,
    "H2_verdict": "supported",
    "lambda_means": {
        "0.012740234375": 28.461328125,
        "0.02548046875": 26.846875,
        "0.0509609375": 24.23359375,
    },
    # Dynamics-only acceptance (homogeneous) from t11/hum
    "dynamics_T_hold_homog": 39.0,
    "dynamics_inv_lambda": 39.245745822474326,
}


def nearly(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    required = [
        PRIMARY / "RESULTS.md",
        PRIMARY / "SMOKE_HUM.md",
        PRIMARY / "score_summary.json",
        PRIMARY / "run_meta.json",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        print("FAIL: missing T8 primary artifacts:")
        for m in missing:
            print(f"  - {m}")
        return 1

    summary = json.loads((PRIMARY / "score_summary.json").read_text())
    sm = summary.get("seed_means", {})
    hum = sm.get("hum", {})
    verdicts = summary.get("verdicts", {})
    h2 = summary.get("H2", {})

    checks = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "pass": ok, "detail": detail})

    add(
        "artifacts_present",
        True,
        f"PRIMARY={PRIMARY}",
    )
    add(
        "prereg_sha",
        summary.get("prereg_sha256") == REF["prereg_sha256"]
        or REF["prereg_sha256"][:12] in (PRIMARY / "RESULTS.md").read_text(),
        f"got prereg_sha256={summary.get('prereg_sha256')}",
    )
    add(
        "hum_T_hold_seed_mean",
        nearly(hum.get("T_hold", -1), REF["hum_T_hold_seed_mean"]),
        f"got {hum.get('T_hold')} ref {REF['hum_T_hold_seed_mean']}",
    )
    add(
        "hum_delta_L_shift_0.3",
        nearly(hum.get("delta_L_shift_0.3", -1), REF["hum_delta_L_shift_0.3"]),
        f"got {hum.get('delta_L_shift_0.3')} (T8 shift regression lock reference)",
    )
    add(
        "H2_verdict",
        verdicts.get("H2") == REF["H2_verdict"] or h2.get("verdict") == REF["H2_verdict"],
        f"verdicts={verdicts.get('H2')} H2.verdict={h2.get('verdict')}",
    )
    lam = hum.get("T_hold_lambda") or h2.get("lambda_seed_means") or {}
    lam_ok = all(
        nearly(lam.get(k, -1), v) for k, v in REF["lambda_means"].items()
    )
    add("H2_lambda_monotone_means", lam_ok, f"lambda_means={lam}")

    # Dynamics acceptance copy if present
    acc_path = OUT_DIR / "ACCEPTANCE_RESULT.json"
    if acc_path.is_file():
        acc = json.loads(acc_path.read_text())
        add(
            "dynamics_acceptance_T_hold",
            nearly(acc.get("T_hold", -1), REF["dynamics_T_hold_homog"]),
            f"got {acc.get('T_hold')} (ref homog 39 ≈ 1/λ)",
        )
        add(
            "dynamics_acceptance_verdict",
            acc.get("verdict") == "PASS",
            f"verdict={acc.get('verdict')}",
        )
    else:
        add("dynamics_acceptance_present", False, "run run_t8_acceptance.py first")

    overall = all(c["pass"] for c in checks)
    report = {
        "verdict": "PASS" if overall else "FAIL",
        "primary_path": str(PRIMARY),
        "reference": REF,
        "checks": checks,
        "hum_seed_mean": hum,
        "verdicts": verdicts,
    }
    out = OUT_DIR / "T8_RERUN_PERSISTENCE_CHECK.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"T8 rerun persistence check: {report['verdict']}")
    for c in checks:
        flag = "PASS" if c["pass"] else "FAIL"
        print(f"  [{flag}] {c['name']}: {c['detail']}")
    print(f"Wrote {out}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
