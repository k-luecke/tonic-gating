#!/usr/bin/env python3
"""T12 Rule 19 — mechanical freeze gates (T5).

Refuses to emit / update ``experiments/phase-routing/t12/PREREG.sha256``
unless every gate prints PASS.

Gates (Freeze gate note + draft §4.2 T5 + Kyle Option-1 lock):
  1. Draft content sha == 8e04b801… (Option-1 C̄=7 tip)
  2. Affinity / C₃ / hum package shas untouched (no retune)
  3. Hum import (imports/hum_measure + test_hum_import)
  4. T8 shift / persistence regression lock PASS
  5. Calibration bounds C̄=7±0.25; ACT/BP 4/10; Fixed T=7
  6. Arm parity smoke (T2) PASS
  7. T0–T4 ticket receipts all PASS
  8. Parent T11.2 / T11.3 PREREG hashes untouched

Hum / AffinityProvider / C₃ are never retuned here.
Usage:
  python3 scripts/rule19_freeze_gates.py
  python3 scripts/rule19_freeze_gates.py --emit-hash   # only if all PASS
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]  # /workspace/scp
T12 = Path(__file__).resolve().parents[1]
SCRIPTS = T12 / "scripts"
IMPORTS = T12 / "imports"
RESULTS = T12 / "results"
DOCS = T12 / "docs"
PT = ZoneInfo("America/Chicago")

DRAFT_PATH = DOCS / "T12_PREREG_DRAFT.md"
EXPECTED_DRAFT_SHA = (
    "8e04b801d9da5e1c94221ecff07d3555dc3c4f630599434160173f61388ffbd1"
)
PREREG_MD = T12 / "PREREG.md"
PREREG_SHA_PATH = T12 / "PREREG.sha256"
FREEZE_META = T12 / "FREEZE_META.json"

T11_2_PREREG_SHA = "593762c4c19e85d1720f6b46edbb93869258c4047ad023e42e144416aaed20d1"
T11_3_PREREG_SHA = "27ab4a9573ec8c01ed043d72ef6c83ba47fdd4a764a373acb7453381bb9bd442"

EXPECTED_FROZEN = {
    "hum_measure/__init__.py": "b51cda27b48e7ce29c801b5a86bb8f613a38cb4ba261e40459dbdb842b56721a",
    "hum_measure/affinity_inputs.py": "7dae877a3975c1ba4ca264f03f572d6749cce39a114ea3adbcffc010c7852c50",
    "hum_measure/COUPLING_HASHES.json": "482fb1b53bfddc0e771145a0783f3b65104231d537cbba507d4783d13a27902b",
    "C_n3": "40e3c0c95849ad67af6738e7842c1a4d24783dafa742ed247d03391db37a1493",
    "hum_state.py": "03ab651db94e87e99e9d560b2e374ccc9dd6e90251422b2364b007089cfd0637",
}

REQUIRED_ARMS = {
    "fixed_t7",
    "act",
    "convergence",
    "coordinator_regime",
    "stateless_regime",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        tag = "PASS" if self.passed else "FAIL"
        return f"[{tag}] {self.name}: {self.detail}"


def gate_draft_sha() -> GateResult:
    if not DRAFT_PATH.is_file():
        return GateResult("draft_content_sha", False, f"missing {DRAFT_PATH}")
    digest = sha256_file(DRAFT_PATH)
    ok = digest == EXPECTED_DRAFT_SHA
    return GateResult(
        "draft_content_sha",
        ok,
        f"got {digest[:16]}… want {EXPECTED_DRAFT_SHA[:16]}…",
        {"sha256": digest, "expected": EXPECTED_DRAFT_SHA},
    )


def gate_affinity_untouched() -> GateResult:
    hum_path = (IMPORTS / "hum_measure").resolve()
    got: Dict[str, Optional[str]] = {}
    aff_init = hum_path / "__init__.py"
    aff_api = hum_path / "affinity_inputs.py"
    c3_hashes = hum_path / "COUPLING_HASHES.json"
    if aff_init.is_file():
        got["hum_measure/__init__.py"] = sha256_file(aff_init)
    if aff_api.is_file():
        got["hum_measure/affinity_inputs.py"] = sha256_file(aff_api)
    if c3_hashes.is_file():
        got["hum_measure/COUPLING_HASHES.json"] = sha256_file(c3_hashes)
        c3 = json.loads(c3_hashes.read_text(encoding="utf-8"))
        got["C_n3"] = c3.get("n3", {}).get("sha256_C_float64_tobytes")
    hum_dyn = (IMPORTS / "t8_hum" / "hum_state.py").resolve()
    if hum_dyn.is_file():
        got["hum_state.py"] = sha256_file(hum_dyn)
    mismatches = {
        k: {"got": got.get(k), "want": v}
        for k, v in EXPECTED_FROZEN.items()
        if got.get(k) != v
    }
    ok = not mismatches and all(k in got for k in EXPECTED_FROZEN)
    return GateResult(
        "affinity_c3_hum_untouched",
        ok,
        "Affinity/C₃/hum shas match frozen baseline"
        if ok
        else f"mismatch={list(mismatches)}",
        {"shas": got, "expected": EXPECTED_FROZEN, "mismatches": mismatches},
    )


def gate_hum_import() -> GateResult:
    test = T12 / "tests" / "test_hum_import.py"
    if not test.is_file():
        return GateResult("hum_import", False, f"missing {test}")
    proc = subprocess.run(
        [sys.executable, str(test)],
        cwd=str(T12),
        capture_output=True,
        text=True,
    )
    link_ok = (IMPORTS / "hum_measure").is_symlink() and (
        IMPORTS / "hum_measure"
    ).resolve().is_dir()
    ok = proc.returncode == 0 and link_ok
    return GateResult(
        "hum_import",
        ok,
        f"test_hum_import rc={proc.returncode} symlink_ok={link_ok}",
        {
            "returncode": proc.returncode,
            "symlink_ok": link_ok,
            "stdout_tail": (proc.stdout or "")[-400:],
            "stderr_tail": (proc.stderr or "")[-400:],
        },
    )


def gate_shift_regression() -> GateResult:
    t4 = RESULTS / "t4" / "T4_MEASURES_RECEIPT.json"
    t8 = RESULTS / "t0" / "T8_RERUN_PERSISTENCE_CHECK.json"
    if not t4.is_file():
        return GateResult("shift_regression_lock", False, f"missing {t4}")
    d4 = load_json(t4)
    locks = d4.get("regression_locks") or {}
    t4_ok = bool(locks.get("pass")) and bool(locks.get("available"))
    t8_ok = True
    t8_detail = "no separate T8 json"
    if t8.is_file():
        d8 = load_json(t8)
        # accept verdict/pass/status
        v = d8.get("verdict") or d8.get("status")
        if "pass" in d8:
            t8_ok = bool(d8.get("pass"))
        elif isinstance(v, str):
            t8_ok = v.upper() == "PASS"
        else:
            # fall back to all checks true if present
            checks = d8.get("checks") or d8.get("results")
            if isinstance(checks, list):
                t8_ok = all(
                    (c.get("pass") if isinstance(c, dict) else True) for c in checks
                )
        t8_detail = f"T8_RERUN present ok={t8_ok}"
    ok = t4_ok and t8_ok
    return GateResult(
        "shift_regression_lock",
        ok,
        f"T4 regression_locks.pass={t4_ok}; {t8_detail}",
        {"t4_regression_locks": locks, "t8_path": str(t8) if t8.is_file() else None},
    )


def gate_calibration_bounds() -> GateResult:
    t3 = RESULTS / "t3" / "T3_CALIBRATION_RECEIPT.json"
    if not t3.is_file():
        return GateResult("calibration_bounds", False, f"missing {t3}")
    d = load_json(t3)
    # Prefer live LOCK constants
    sys.path.insert(0, str(T12))
    from controllers.calibration import LOCK, within_c_bar  # noqa: E402

    primary = float(d.get("primary_mean_C_bar", float("nan")))
    within = bool(d.get("primary_within_tol")) and within_c_bar(primary)
    checks = d.get("checks") or {}
    act_ok = bool(checks.get("act_C_bar"))
    coord_ok = bool(checks.get("coord_C_bar"))
    fixed_ok = bool(checks.get("fixed_C_bar"))
    sep_ok = bool(checks.get("easy_hard_separation_retained"))
    lock_ok = (
        LOCK.c_bar_target == 7.0
        and LOCK.c_bar_tol == 0.25
        and LOCK.act_easy_center == 4.0
        and LOCK.act_hard_center == 10.0
        and LOCK.fixed_t == 7
    )
    status_ok = str(d.get("status", "")).upper() == "PASS"
    ok = within and act_ok and coord_ok and fixed_ok and sep_ok and lock_ok and status_ok
    return GateResult(
        "calibration_bounds",
        ok,
        (
            f"C̄={primary} within_tol={within}; ACT/BP centers "
            f"{LOCK.act_easy_center}/{LOCK.act_hard_center}; Fixed T={LOCK.fixed_t}; "
            f"status={d.get('status')}"
        ),
        {
            "primary_mean_C_bar": primary,
            "checks": checks,
            "lock": LOCK.as_dict(),
        },
    )


def gate_arm_parity() -> GateResult:
    t2 = RESULTS / "t2" / "T2_HARNESS_RECEIPT.json"
    if not t2.is_file():
        return GateResult("arm_parity_smoke", False, f"missing {t2}")
    d = load_json(t2)
    parity = d.get("parity")
    # parity may be list or dict
    parity_pass = False
    arms_seen: set = set()
    if isinstance(parity, list) and parity:
        first = parity[0]
        parity_pass = bool(first.get("pass"))
        arms_seen = set(first.get("arms") or [])
    elif isinstance(parity, dict):
        parity_pass = bool(parity.get("pass"))
        arms_seen = set(parity.get("arms") or [])
    status_ok = str(d.get("status", "")).upper() == "PASS"
    arms_ok = REQUIRED_ARMS.issubset(arms_seen)
    ok = parity_pass and arms_ok and status_ok
    return GateResult(
        "arm_parity_smoke",
        ok,
        f"parity.pass={parity_pass} arms={sorted(arms_seen)} status={d.get('status')}",
        {"arms": sorted(arms_seen), "required": sorted(REQUIRED_ARMS)},
    )


def gate_tickets_t0_t4() -> GateResult:
    paths = {
        "T0": RESULTS / "t0" / "T0_SUMMARY.json",
        "T1": RESULTS / "t1" / "T1_SWEEP_RECEIPT.json",
        "T2": RESULTS / "t2" / "T2_HARNESS_RECEIPT.json",
        "T3": RESULTS / "t3" / "T3_CALIBRATION_RECEIPT.json",
        "T4": RESULTS / "t4" / "T4_MEASURES_RECEIPT.json",
    }
    detail: Dict[str, Any] = {}
    all_ok = True
    for name, p in paths.items():
        if not p.is_file():
            detail[name] = {"pass": False, "error": "missing"}
            all_ok = False
            continue
        d = load_json(p)
        st = d.get("status") or d.get("verdict")
        ok = str(st).upper() == "PASS"
        detail[name] = {"pass": ok, "status": st, "path": str(p)}
        all_ok = all_ok and ok
    return GateResult(
        "tickets_T0_T4_PASS",
        all_ok,
        "; ".join(f"{k}={'PASS' if v.get('pass') else 'FAIL'}" for k, v in detail.items()),
        detail,
    )


def gate_parent_preregs() -> GateResult:
    t112 = ROOT / "t11" / "PREREG.md"
    t112_sha_file = ROOT / "t11" / "PREREG.sha256"
    t113 = ROOT / "experiments" / "phase-routing" / "t11_3" / "PREREG.md"
    t113_sha_file = ROOT / "experiments" / "phase-routing" / "t11_3" / "PREREG.sha256"
    got112 = sha256_file(t112) if t112.is_file() else None
    got113 = sha256_file(t113) if t113.is_file() else None
    file112 = (
        t112_sha_file.read_text(encoding="utf-8").split()[0].strip()
        if t112_sha_file.is_file()
        else None
    )
    file113 = (
        t113_sha_file.read_text(encoding="utf-8").split()[0].strip()
        if t113_sha_file.is_file()
        else None
    )
    ok = (
        got112 == T11_2_PREREG_SHA
        and got113 == T11_3_PREREG_SHA
        and file112 == T11_2_PREREG_SHA
        and file113 == T11_3_PREREG_SHA
    )
    return GateResult(
        "parent_prereg_untouched",
        ok,
        f"T11.2={got112 and got112[:12]}… T11.3={got113 and got113[:12]}…",
        {
            "t11_2_md": got112,
            "t11_3_md": got113,
            "t11_2_sha_file": file112,
            "t11_3_sha_file": file113,
        },
    )


def promote_draft_to_prereg(*, frozen_at: str) -> str:
    """Write PREREG.md from draft with Status flipped to FROZEN. Return sha256."""
    text = DRAFT_PATH.read_text(encoding="utf-8")
    # Title
    text = text.replace(
        "# PREREG DRAFT — T12 The Loop Budget",
        "# PREREG — T12 The Loop Budget",
        1,
    )
    # Status row
    text = text.replace(
        "| **Status** | **DRAFT** — do **not** freeze; do **not** emit `PREREG.sha256` |",
        f"| **Status** | **FROZEN** {frozen_at} — Rule-19 T12 T5 PASS; emit only via freeze_prereg |",
        1,
    )
    # Lead callout
    text = text.replace(
        "> **DRAFT.** Sections below lock **intent** for freeze review (T5). "
        "Emitting a hash before gates + Kyle greenlight voids the draft for that run.",
        "> **FROZEN.** Rule-19 T12 T5 gates PASS. Hum/C₃/Affinity `__all__` untouched. "
        "Parent T11.2 / T11.3 PREREGs not re-emitted.",
        1,
    )
    # Option-1 amendment still says DRAFT — flip that clause
    text = text.replace(
        "Affinity/C₃/hum still frozen. Status remains **DRAFT** (no `PREREG.sha256`).",
        "Affinity/C₃/hum still frozen. Status **FROZEN** via Rule-19 T5.",
        1,
    )
    # Freeze gate note at end
    text = text.replace(
        "## Freeze gate note (T5 later)\n\n"
        "`scripts/freeze_prereg.py` is a **stub**. It must refuse to emit `PREREG.sha256` "
        "until Rule-19-style gates (hum import, shift regression, calibration bounds, "
        "arm parity smoke) PASS and Kyle greenlights. This file remains **DRAFT** until then.",
        "## Freeze gate note\n\n"
        "`scripts/freeze_prereg.py` + `scripts/rule19_freeze_gates.py` emitted this "
        "frozen PREREG after Rule-19 T5 gates PASS. Do not re-emit casually; Affinity/C₃/hum "
        "remain untouched.",
        1,
    )
    PREREG_MD.write_text(text, encoding="utf-8")
    digest = sha256_file(PREREG_MD)
    PREREG_SHA_PATH.write_text(digest + "  PREREG.md\n", encoding="utf-8")
    return digest


def run_all_gates() -> List[GateResult]:
    return [
        gate_draft_sha(),
        gate_affinity_untouched(),
        gate_hum_import(),
        gate_shift_regression(),
        gate_calibration_bounds(),
        gate_arm_parity(),
        gate_tickets_t0_t4(),
        gate_parent_preregs(),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="T12 Rule 19 freeze gates (T5)")
    ap.add_argument(
        "--emit-hash",
        action="store_true",
        help="Promote draft→PREREG.md + write PREREG.sha256 iff all PASS",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=RESULTS / "t5" / "T5_RULE19_GATES.json",
    )
    args = ap.parse_args()

    print("=== T12 Rule 19 freeze gates (T5) ===")
    print(f"draft: {DRAFT_PATH}")
    print(f"expected draft sha: {EXPECTED_DRAFT_SHA[:16]}…")
    print("hum/C3/Affinity __all__: FROZEN (must remain untouched)")
    print()

    results = run_all_gates()
    for r in results:
        print(r.line())

    all_pass = all(r.passed for r in results)
    print()
    if all_pass:
        print("ALL T12 T5 GATES PASS")
    else:
        print("GATES FAILED — refusing T12 PREREG.sha256 emit")

    frozen_at = datetime.now(PT).strftime("%Y-%m-%d %H:%M PT")
    report: Dict[str, Any] = {
        "rule": "19_t12_t5",
        "ticket": "T5",
        "all_pass": all_pass,
        "pt": frozen_at,
        "expected_draft_sha": EXPECTED_DRAFT_SHA,
        "gates": [
            {
                "name": r.name,
                "pass": r.passed,
                "detail": r.detail,
                "metrics": r.metrics,
            }
            for r in results
        ],
        "hash_emitted": False,
        "prereg_sha256": None,
        "t11_2_prereg_untouched": T11_2_PREREG_SHA,
        "t11_3_prereg_untouched": T11_3_PREREG_SHA,
        "affinity_c3_hum": "UNTOUCHED",
    }

    if args.emit_hash:
        if not all_pass:
            print("ERROR: --emit-hash blocked (gates not all PASS)")
            report["block_reason"] = "gates_failed"
        else:
            digest = promote_draft_to_prereg(frozen_at=frozen_at)
            report["hash_emitted"] = True
            report["prereg_sha256"] = digest
            report["prereg_path"] = str(PREREG_MD)
            meta = {
                "ticket": "t12",
                "status": "FROZEN",
                "frozen_at_pt": frozen_at,
                "prereg_path": str(PREREG_MD),
                "prereg_sha256": digest,
                "draft_content_sha_at_freeze": EXPECTED_DRAFT_SHA,
                "rule19": "PASS",
                "rule19_script": str(Path(__file__).resolve()),
                "t11_2_prereg_untouched": T11_2_PREREG_SHA,
                "t11_3_prereg_untouched": T11_3_PREREG_SHA,
                "note": (
                    "Emitted only via freeze_prereg.py after Rule 19 PASS; "
                    "hum/provider/C3 untouched. Option-1 C̄=7±0.25; ACT/BP 4/10; Fixed T=7."
                ),
            }
            FREEZE_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            print(f"T12 PREREG.md + PREREG.sha256 written: {digest}")
            print(f"FREEZE_META.json updated")
    else:
        print("No hash emit (pass --emit-hash only after all PASS via freeze_prereg).")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.json_out}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
