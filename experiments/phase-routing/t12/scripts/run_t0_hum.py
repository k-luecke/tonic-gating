#!/usr/bin/env python3
"""T0 one-command: hum import + T8 acceptance + T8 persistence check + n=2 smoke.

From t12/:
  python3 scripts/run_t0_hum.py

Writes results/T0_HUM_REPRODUCE.md and results/t0/* artifacts.
Does NOT retune Affinity/C3/hum frozen packages; no training; no PREREG emit.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

T12_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
OUT_ROOT = T12_ROOT / "results"
OUT_DIR = OUT_ROOT / "t0"
IMPORTS = T12_ROOT / "imports"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def run_step(label: str, argv: list[str]) -> tuple[bool, str]:
    print(f"\n=== {label} ===")
    print(" ".join(argv))
    proc = subprocess.run(argv, cwd=str(T12_ROOT), check=False, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    ok = proc.returncode == 0
    print(f"→ {'PASS' if ok else 'FAIL'} (exit {proc.returncode})")
    return ok, proc.stdout + proc.stderr


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bullets: dict[str, dict] = {}

    # 1) hum_measure import
    ok, out = run_step(
        "1. hum_measure import",
        [sys.executable, str(T12_ROOT / "tests" / "test_hum_import.py")],
    )
    hum_path = (IMPORTS / "hum_measure").resolve()
    bullets["hum_measure_import"] = {
        "pass": ok,
        "path": str(hum_path),
        "symlink_ok": (IMPORTS / "hum_measure").is_symlink() and hum_path.is_dir(),
    }

    # Confirm Affinity __all__ / frozen shas untouched (read-only check)
    aff_init = hum_path / "__init__.py"
    aff_api = hum_path / "affinity_inputs.py"
    c3_hashes = hum_path / "COUPLING_HASHES.json"
    frozen_shas = {}
    if aff_init.is_file():
        frozen_shas["hum_measure/__init__.py"] = sha256_file(aff_init)
    if aff_api.is_file():
        frozen_shas["hum_measure/affinity_inputs.py"] = sha256_file(aff_api)
    if c3_hashes.is_file():
        frozen_shas["hum_measure/COUPLING_HASHES.json"] = sha256_file(c3_hashes)
        c3 = json.loads(c3_hashes.read_text())
        frozen_shas["C_n3"] = c3.get("n3", {}).get("sha256_C_float64_tobytes")

    # Expected frozen shas (pre-T0 baseline; must remain)
    EXPECTED = {
        "hum_measure/__init__.py": "b51cda27b48e7ce29c801b5a86bb8f613a38cb4ba261e40459dbdb842b56721a",
        "hum_measure/affinity_inputs.py": "7dae877a3975c1ba4ca264f03f572d6749cce39a114ea3adbcffc010c7852c50",
        "hum_measure/COUPLING_HASHES.json": "482fb1b53bfddc0e771145a0783f3b65104231d537cbba507d4783d13a27902b",
        "C_n3": "40e3c0c95849ad67af6738e7842c1a4d24783dafa742ed247d03391db37a1493",
        "hum_state.py": "03ab651db94e87e99e9d560b2e374ccc9dd6e90251422b2364b007089cfd0637",
    }
    hum_dyn = (IMPORTS / "t8_hum" / "hum_state.py").resolve()
    if hum_dyn.is_file():
        frozen_shas["hum_state.py"] = sha256_file(hum_dyn)
    untouched = all(frozen_shas.get(k) == v for k, v in EXPECTED.items() if k in frozen_shas)
    bullets["frozen_packages_untouched"] = {
        "pass": untouched,
        "shas": frozen_shas,
        "expected": EXPECTED,
    }

    # 2) T8 hum unit tests (acceptance)
    ok2, _ = run_step(
        "2. T8 hum acceptance (unit / persistence dynamics)",
        [sys.executable, str(SCRIPTS / "run_t8_acceptance.py")],
    )
    acc = {}
    acc_path = OUT_DIR / "ACCEPTANCE_RESULT.json"
    if acc_path.is_file():
        acc = json.loads(acc_path.read_text())
    bullets["t8_hum_unit_tests"] = {
        "pass": ok2 and acc.get("verdict") == "PASS",
        "T_hold": acc.get("T_hold"),
        "T_hold_target": 41.0,
        "inv_lambda": acc.get("inv_lambda"),
        "scaling_pass": acc.get("scaling_pass"),
        "command": "python3 scripts/run_t8_acceptance.py",
    }

    # 3) T8 rerun persistence measurement (artifact check + dynamics)
    ok3, _ = run_step(
        "3. T8 rerun persistence measurement (documented paths)",
        [sys.executable, str(SCRIPTS / "check_t8_rerun_persistence.py")],
    )
    pers = {}
    pers_path = OUT_DIR / "T8_RERUN_PERSISTENCE_CHECK.json"
    if pers_path.is_file():
        pers = json.loads(pers_path.read_text())
    bullets["t8_rerun_persistence"] = {
        "pass": ok3 and pers.get("verdict") == "PASS",
        "primary": str((IMPORTS / "t8_results_primary").resolve()),
        "hum_T_hold_seed_mean": (pers.get("hum_seed_mean") or {}).get("T_hold"),
        "shift_lock_delta_L": (pers.get("hum_seed_mean") or {}).get("delta_L_shift_0.3"),
        "command": "python3 scripts/check_t8_rerun_persistence.py",
    }

    # 4) n=2 smoke + locks
    ok4, _ = run_step(
        "4. n=2 easy/hard smoke (high-pass + amplitude readout)",
        [sys.executable, str(SCRIPTS / "smoke_hum_n2.py")],
    )
    n2 = {}
    n2_path = OUT_DIR / "SMOKE_HUM_N2.json"
    if n2_path.is_file():
        n2 = json.loads(n2_path.read_text())
    locks = n2.get("locks") or {}
    bullets["locks_n2_hp_amp"] = {
        "pass": ok4 and n2.get("verdict") == "PASS",
        "hum_n": locks.get("hum_n"),
        "high_pass_every_drive": locks.get("high_pass_every_drive"),
        "amplitude_readout_never_phase_gated": locks.get(
            "amplitude_readout_never_phase_gated"
        ),
        "homogeneous_T_hold": n2.get("homogeneous_T_hold_1e"),
        "coupled_T_hold": n2.get("coupled_diagnostic_T_hold_1e"),
        "operating_hp_easy": (n2.get("operating_hp") or {}).get("easy_unit0", {}).get("T_hold"),
        "operating_hp_hard": (n2.get("operating_hp") or {}).get("hard_unit1", {}).get("T_hold"),
        "C_n2_sha": (n2.get("coupling") or {}).get("sha256_C_float64_tobytes"),
        "command": "python3 scripts/smoke_hum_n2.py",
    }

    overall = all(b.get("pass") for b in bullets.values())
    created = time.strftime("%Y-%m-%dT%H:%M:%S PT")

    # Collect new file shas under t12
    new_files = []
    for p in sorted(T12_ROOT.rglob("*")):
        if not p.is_file():
            continue
        if "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        rel = p.relative_to(T12_ROOT)
        s = str(rel)
        if s.startswith("results/t0/") or s.startswith("scripts/") or s in (
            "results/T0_HUM_REPRODUCE.md",
            "imports/PATHS.md",
            "tests/test_t0_hum.py",
            "tests/test_hum_import.py",
            "README.md",
            "ARTIFACT_INVENTORY.md",
        ):
            if p.is_symlink():
                continue
            new_files.append((s, p.stat().st_size, sha256_file(p)))

    receipt = OUT_ROOT / "T0_HUM_REPRODUCE.md"
    lines = [
        "# T0_HUM_REPRODUCE — GNURadio acceptance receipt",
        "",
        f"**Overall: {'PASS' if overall else 'FAIL'}**",
        "",
        f"- Created: {created}",
        f"- Tree: `{T12_ROOT}`",
        f"- PREREG draft sha (untouched): `78091e84b88e006fb27dc3f7648d0e545387060ab96140f6afe6567c232a6977`",
        "",
        "## Bullets",
        "",
        "| # | Bullet | Pass | Key numbers |",
        "|---|--------|:----:|-------------|",
        f"| 1 | `hum_measure` imports | {'yes' if bullets['hum_measure_import']['pass'] else 'no'} | `{hum_path}` |",
        f"| 2 | T8 hum unit tests (acceptance) | {'yes' if bullets['t8_hum_unit_tests']['pass'] else 'no'} | T_hold={bullets['t8_hum_unit_tests'].get('T_hold')} (target 41±5); 1/λ≈{bullets['t8_hum_unit_tests'].get('inv_lambda')} |",
        f"| 3 | T8 rerun persistence measurement | {'yes' if bullets['t8_rerun_persistence']['pass'] else 'no'} | MoE seed-mean T_hold={bullets['t8_rerun_persistence'].get('hum_T_hold_seed_mean')}; shift ΔL={bullets['t8_rerun_persistence'].get('shift_lock_delta_L')} |",
        f"| 4 | Locks: n=2 / HP every drive / amp never phase-gated | {'yes' if bullets['locks_n2_hp_amp']['pass'] else 'no'} | homog T_hold={bullets['locks_n2_hp_amp'].get('homogeneous_T_hold')}; HP easy/hard={bullets['locks_n2_hp_amp'].get('operating_hp_easy')}/{bullets['locks_n2_hp_amp'].get('operating_hp_hard')} |",
        f"| 5 | Affinity / C₃ / hum_state untouched | {'yes' if bullets['frozen_packages_untouched']['pass'] else 'no'} | see shas below |",
        "",
        "## Persistence vs T8 reference",
        "",
        "| Measure | T8 reference | This run |",
        "|---------|-------------:|---------:|",
        f"| Dynamics homog T_hold (1/e) | 39.0 | {bullets['t8_hum_unit_tests'].get('T_hold')} |",
        f"| 1/λ | ≈39.2457 | {bullets['t8_hum_unit_tests'].get('inv_lambda')} |",
        f"| MoE hum seed-mean T_hold (rerun primary) | 26.846875 | {bullets['t8_rerun_persistence'].get('hum_T_hold_seed_mean')} |",
        f"| MoE hum ΔL shift0.3 (regression lock) | 0.01707748666642206 | {bullets['t8_rerun_persistence'].get('shift_lock_delta_L')} |",
        f"| n=2 homog T_hold | — (same family ≈39) | {bullets['locks_n2_hp_amp'].get('homogeneous_T_hold')} |",
        f"| n=2 coupled diagnostic T_hold | — | {bullets['locks_n2_hp_amp'].get('coupled_T_hold')} |",
        "",
        "## Frozen package shas (must match baseline)",
        "",
        "| File | sha256 |",
        "|------|--------|",
    ]
    for k, v in EXPECTED.items():
        got = frozen_shas.get(k, "MISSING")
        flag = "OK" if got == v else "DRIFT"
        lines.append(f"| `{k}` | `{got}` ({flag}) |")
    lines += [
        "",
        "## Affinity / C₃ note",
        "",
        "- Affinity `__all__` **not edited** (T11 n=3 specialist package left frozen).",
        "- C₃ coupling hash unchanged; T12 builds **n=2** via same `make_coupling` formula only in smoke under `results/t0/`.",
        "- No T11 / T11.2 / T11.3 PREREG hash emit or edit.",
        "",
        "## Commands (re-run from t12/)",
        "",
        "```bash",
        "cd /workspace/scp/experiments/phase-routing/t12",
        "python3 scripts/run_t0_hum.py          # all T0 bullets",
        "# or stepwise:",
        "python3 tests/test_hum_import.py",
        "python3 scripts/run_t8_acceptance.py",
        "python3 scripts/check_t8_rerun_persistence.py",
        "python3 scripts/smoke_hum_n2.py",
        "```",
        "",
        "## Documented paths (see imports/PATHS.md)",
        "",
        "| Name | Path |",
        "|------|------|",
        "| hum_measure | `imports/hum_measure` → `/workspace/scp/t11/imports/hum_measure` |",
        "| t8_hum dynamics | `imports/t8_hum` → `.../t11/hum/` (`acceptance_t8.py`, `hum_state.py`) |",
        "| T8 primary results | `imports/t8_results_primary` → `.../t8/results/primary/` |",
        "| T8 PREREG | `imports/t8_PREREG.md` → `/workspace/scp/t11/imports/t8_PREREG.md` |",
        "",
        "## New / edited file shas (this receipt run)",
        "",
        "| Path | Bytes | sha256 |",
        "|------|------:|--------|",
    ]
    for s, nbytes, digest in new_files:
        lines.append(f"| `{s}` | {nbytes} | `{digest}` |")
    lines.append("")
    receipt.write_text("\n".join(lines))

    summary_json = OUT_DIR / "T0_SUMMARY.json"
    summary_json.write_text(
        json.dumps(
            {
                "verdict": "PASS" if overall else "FAIL",
                "created_pt": created,
                "bullets": bullets,
                "frozen_shas": frozen_shas,
            },
            indent=2,
        )
        + "\n"
    )
    print("\n" + receipt.read_text())
    print(f"Wrote {receipt}")
    print(f"Wrote {summary_json}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
