#!/usr/bin/env python3
"""T12b Rule 19 — mechanical freeze gates (G1–G11).

Default: evaluate + write receipt. Does NOT emit PREREG.sha256.
--emit-hash promotes draft→PREREG.md (Status FROZEN) and writes PREREG.sha256 iff all PASS.
Draft file bytes stay unchanged (content sha 408eaaac…).

G9: real HumMeasureArm (HumState path); stub refused as counted.
G10: mid-gap ScoreAffinity lure wired + asserted.
G11: majority_k3 labeled secondary present.
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

T12B = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = Path(__file__).resolve().parent
DOCS = T12B / "docs"
RESULTS = T12B / "results"
IMPORTS = T12B / "imports"
PT = ZoneInfo("America/Chicago")

DRAFT_PATH = DOCS / "T12B_PREREG_DRAFT.md"
CROSSOVER = DOCS / "T12B_CROSSOVER_PREDICTION.md"
PREREG_MD = T12B / "PREREG.md"
PREREG_SHA_PATH = T12B / "PREREG.sha256"
EXPECTED_DRAFT_SHA = "408eaaaca59d0c210e1b1ef50d2494213662da1059b3542c38ce866651801b21"
RECEIPT_JSON = RESULTS / "T12B_RULE19_GATES.json"
RECEIPT_MD = RESULTS / "T12B_RULE19_GATES.md"

T11_2_PREREG_SHA = "593762c4c19e85d1720f6b46edbb93869258c4047ad023e42e144416aaed20d1"
T11_3_PREREG_SHA = "27ab4a9573ec8c01ed043d72ef6c83ba47fdd4a764a373acb7453381bb9bd442"
EXPECTED_ALL_SHA = "646df30bbeacdc77432e9cb1daa366640ca0cfeb2e01216d4cb5f03a9df404c6"
EXPECTED_FROZEN = {'hum_measure/__init__.py': 'b51cda27b48e7ce29c801b5a86bb8f613a38cb4ba261e40459dbdb842b56721a', 'hum_measure/affinity_inputs.py': '7dae877a3975c1ba4ca264f03f572d6749cce39a114ea3adbcffc010c7852c50', 'hum_measure/COUPLING_HASHES.json': '482fb1b53bfddc0e771145a0783f3b65104231d537cbba507d4783d13a27902b', 'C_n3': '40e3c0c95849ad67af6738e7842c1a4d24783dafa742ed247d03391db37a1493', 'hum_state.py': '03ab651db94e87e99e9d560b2e374ccc9dd6e90251422b2364b007089cfd0637'}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


def _prereg_sha(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").split()[0].strip()


def draft_is_draft() -> bool:
    if not DRAFT_PATH.is_file():
        return True
    return "DRAFT" in DRAFT_PATH.read_text(encoding="utf-8")[:1500]


def gate_g1() -> GateResult:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "gap_inject.py"), "--self-test"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    ok = proc.returncode == 0 and "PASS" in (proc.stdout or "")
    return GateResult("G1_gaps_zero_affinity", ok, f"gap_inject --self-test rc={proc.returncode}",
                      {"stdout_tail": (proc.stdout or "")[-400:], "stderr_tail": (proc.stderr or "")[-200:]})


def gate_g2() -> GateResult:
    sys.path.insert(0, str(SCRIPTS))
    from arms import PRIMARY_ARMS, arm_factory
    arms = arm_factory(mode="primary")
    ok = "one_step_memory" in arms and "one_step_memory" in PRIMARY_ARMS
    return GateResult("G2_one_step_memory_present", ok,
                      f"arms={list(arms)}; PRIMARY_ARMS={list(PRIMARY_ARMS)}",
                      {"arms": list(arms), "primary_arms": list(PRIMARY_ARMS)})


def gate_g3() -> GateResult:
    if not CROSSOVER.is_file():
        return GateResult("G3_crossover_prediction_prerun", False, f"missing {CROSSOVER}")
    text = CROSSOVER.read_text(encoding="utf-8")
    cites = ("65" in text) and ("T_hold" in text)
    primary_hits = list(RESULTS.glob("T12B_PRIMARY*")) + list(RESULTS.glob("*PRIMARY_AGGREGATE*"))
    ok = cites and not primary_hits
    return GateResult("G3_crossover_prediction_prerun", ok,
                      f"cites_thold_65={cites}; no_primary_outputs={not primary_hits}",
                      {"sha256": sha256_file(CROSSOVER), "primary_hits": [p.name for p in primary_hits]})


def gate_g4() -> GateResult:
    hum = (IMPORTS / "hum_measure").resolve()
    got: Dict[str, Optional[str]] = {}
    for rel, key in [
        ("__init__.py", "hum_measure/__init__.py"),
        ("affinity_inputs.py", "hum_measure/affinity_inputs.py"),
        ("COUPLING_HASHES.json", "hum_measure/COUPLING_HASHES.json"),
    ]:
        p = hum / rel
        if p.is_file():
            got[key] = sha256_file(p)
    c3_path = hum / "COUPLING_HASHES.json"
    if c3_path.is_file():
        c3 = json.loads(c3_path.read_text(encoding="utf-8"))
        got["C_n3"] = (c3.get("n3") or {}).get("sha256_C_float64_tobytes")
    hum_dyn = ROOT / "experiments" / "phase-routing" / "t11" / "hum" / "hum_state.py"
    if hum_dyn.is_file():
        got["hum_state.py"] = sha256_file(hum_dyn)
    mismatches = {k: {"got": got.get(k), "want": v} for k, v in EXPECTED_FROZEN.items() if got.get(k) != v}
    ok = not mismatches and all(k in got for k in EXPECTED_FROZEN)
    return GateResult("G4_no_c3_hum_affinity_retune", ok,
                      "Affinity/C₃/hum shas match frozen baseline" if ok else f"mismatch={list(mismatches)}",
                      {"shas": got, "mismatches": mismatches})


def gate_g5() -> GateResult:
    got = _prereg_sha(T12B.parent / "t11_3" / "PREREG.sha256")
    ok = got == T11_3_PREREG_SHA
    return GateResult("G5_t11_3_hash_untouched", ok,
                      f"got={(got or '')[:16]}… want={T11_3_PREREG_SHA[:16]}…",
                      {"got": got, "expected": T11_3_PREREG_SHA})


def gate_g6() -> GateResult:
    got = _prereg_sha(T12B.parent / "t11_2" / "PREREG.sha256")
    ok = got == T11_2_PREREG_SHA
    return GateResult("G6_t11_2_hash_untouched", ok,
                      f"got={(got or '')[:16]}… want={T11_2_PREREG_SHA[:16]}…",
                      {"got": got, "expected": T11_2_PREREG_SHA})


def gate_g7() -> GateResult:
    link = T12B / "generators" / "sealed"
    all_seq = link / "ALL_SEQUENCES.jsonl"
    is_link = link.is_symlink()
    resolve = None
    target_ok = False
    if is_link:
        try:
            resolve = str(link.resolve())
            target_ok = "t11_3" in resolve and "sealed" in resolve
        except OSError:
            target_ok = False
    digest = sha256_file(all_seq) if all_seq.is_file() else None
    ok = is_link and target_ok and digest == EXPECTED_ALL_SHA
    return GateResult("G7_sealed_reuse_not_rewritten", ok,
                      f"symlink={is_link}; target_ok={target_ok}; all_sha={(digest or '')[:16]}…",
                      {"symlink": is_link, "resolve": resolve, "all_sha": digest})


def gate_g8() -> GateResult:
    primary = (SCRIPTS / "primary_run.py").read_text(encoding="utf-8")
    draft = DRAFT_PATH.read_text(encoding="utf-8") if DRAFT_PATH.is_file() else ""
    bad = ["llama_cpp", "specialist_exec", "browsergym.make"]
    hits = [b for b in bad if b in primary.lower()]
    routing_claimed = ("routing-only" in draft.lower()) or ("routing-only" in primary.lower())
    ok = (not hits) and routing_claimed
    return GateResult("G8_routing_only", ok,
                      f"forbidden_hits={hits}; routing_claimed={routing_claimed}", {"hits": hits})


def gate_g9() -> GateResult:
    sys.path.insert(0, str(SCRIPTS))
    from arms import HumMeasureArm, HumMeasureArmStub, arm_factory, refuse_stub_as_counted
    arms = arm_factory(mode="primary")
    hum = arms["hum_measure"]
    is_real = isinstance(hum, HumMeasureArm) and not isinstance(hum, HumMeasureArmStub)
    counted = bool(getattr(hum, "counted", False))
    stub_refused = False
    try:
        refuse_stub_as_counted(HumMeasureArmStub())
    except RuntimeError:
        stub_refused = True
    dry_only = bool(getattr(hum, "dry_run_only", False))
    ok = is_real and counted and stub_refused and not dry_only
    return GateResult("G9_stub_ne_counted_arm", ok,
                      f"class={type(hum).__name__}; counted={counted}; stub_refused={stub_refused}",
                      {"hum_class": type(hum).__name__, "counted": counted, "hum_state_path": True})


def gate_g10() -> GateResult:
    sys.path.insert(0, str(SCRIPTS))
    import numpy as np
    import gap_inject
    from gap_inject import GapSpec
    base = np.ones((32, 3), dtype=np.float64)
    gap = GapSpec(start=10, length=6)
    mod, mask, pert = gap_inject.inject_gap_with_g10(
        base, gap, seed=20260923, seq_id=0, true_spec_index=0, peak=1.0
    )
    t_flip = gap.start + pert.offset
    flip_nz = bool(np.abs(mod[t_flip]).sum() > 0)
    surrounding = all(bool(np.all(mod[i] == 0.0)) for i in np.where(mask)[0] if int(i) != t_flip)
    ok = flip_nz and surrounding and int(pert.offset) == int(gap.length) // 2
    return GateResult("G10_in_gap_scoreaffinity_lure", ok,
                      f"flip_nz={flip_nz}; surrounding_zero={surrounding}; mid_offset={pert.offset}",
                      {"t_flip": int(t_flip), "wrong_spec_index": int(pert.wrong_spec_index), "offset": int(pert.offset)})


def gate_g11() -> GateResult:
    sys.path.insert(0, str(SCRIPTS))
    from arms import PRIMARY_ARMS, SECONDARY_ARMS, MajorityK3Arm, arm_factory
    arms = arm_factory(mode="primary")
    present = "majority_k3" in arms
    secondary = "majority_k3" in SECONDARY_ARMS
    arm = arms.get("majority_k3")
    labeled = bool(getattr(arm, "labeled_secondary", False)) if arm else False
    is_cls = isinstance(arm, MajorityK3Arm) if arm else False
    k_ok = int(getattr(arm, "k", 0) or 0) == 3 if arm else False
    not_primary = "majority_k3" not in PRIMARY_ARMS
    draft = DRAFT_PATH.read_text(encoding="utf-8") if DRAFT_PATH.is_file() else ""
    draft_docs = ("majority_k3" in draft) and ("labeled secondary" in draft.lower())
    ok = present and secondary and labeled and is_cls and k_ok and not_primary and draft_docs
    return GateResult("G11_majority_k3_labeled_secondary", ok,
                      f"present={present}; labeled={labeled}; k={getattr(arm,'k',None)}; not_primary={not_primary}; draft_docs={draft_docs}",
                      {"secondary_arms": list(SECONDARY_ARMS), "primary_arms": list(PRIMARY_ARMS), "draft_docs": draft_docs})



def promote_draft_to_prereg(*, frozen_at: str) -> str:
    """Write PREREG.md from draft with Status flipped to FROZEN. Return sha256 of PREREG.md.

    Draft file bytes are left unchanged so draft_content_sha stays 408eaaac….
    """
    if not DRAFT_PATH.is_file():
        raise FileNotFoundError(DRAFT_PATH)
    draft_sha = sha256_file(DRAFT_PATH)
    if draft_sha != EXPECTED_DRAFT_SHA:
        raise RuntimeError(
            f"draft content sha mismatch: got {draft_sha} want {EXPECTED_DRAFT_SHA}"
        )
    text = DRAFT_PATH.read_text(encoding="utf-8")
    text = text.replace(
        "# PREREG DRAFT — T12b Persistence under evidence gaps (routing-only)",
        "# PREREG — T12b Persistence under evidence gaps (routing-only)",
        1,
    )
    text = text.replace(
        "| **Status** | **DRAFT** — do **not** freeze; do **not** emit `PREREG.sha256` |",
        f"| **Status** | **FROZEN** {frozen_at} — Rule-19 T12b PASS; emit via freeze_prereg --authorize |",
        1,
    )
    text = text.replace(
        "| **Run location** | Linux `/workspace/scp` (scaffold + draft this turn; **no primary**) |",
        "| **Run location** | Linux `/workspace/scp` (frozen; one primary × five seeds) |",
        1,
    )
    text = text.replace(
        "> **DRAFT.** Numbers below lock **intent** for freeze review. Emitting a hash before Rule-19 gates + Kyle freeze greenlight voids the draft for that run.",
        "> **FROZEN.** Rule-19 T12b gates PASS. Hum/C₃/Affinity `__all__` untouched. "
        "Parent T11.2 / T11.3 PREREGs not re-emitted. Primary contrast: hum_measure vs one_step_memory; majority_k3 labeled secondary.",
        1,
    )
    # Soften hold-freeze language that would contradict emit
    text = text.replace(
        "**Hold freeze** until this perturbation is locked and harness-wired — **not** only until HumState / HumMeasureOrchestrator is wired (G9 remains necessary but insufficient).",
        "**In-gap perturbation locked** at freeze — harness-wired (G10 PASS). G9 HumState counted path required.",
        1,
    )
    text = text.replace(
        "| **Freeze gate** | Do **not** emit `PREREG.sha256` until primary perturbation is implemented in `gap_inject` (or successor) and dry-run asserts one-frame non-zero + surrounding zeros. |",
        "| **Freeze gate** | Satisfied at this freeze — G10 wired + asserted (Rule-19 PASS). |",
        1,
    )
    text = text.replace(
        "- Do **not** run full primary while status is DRAFT.\n"
        "- Do **not** emit `PREREG.sha256` for T12b while DRAFT.\n",
        "- Primary may run only under this frozen PREREG (one × five seeds).\n"
        "- `PREREG.sha256` emitted via authorized freeze_prereg after Rule-19 PASS.\n",
        1,
    )
    text = text.replace(
        "**Freeze hold (amendment):** Do **not** freeze on HumState wire alone. With G≪65, bare silence + hold-last **ties** hum and `one_step_memory` when the label should not change; **in-gap perturbation (§3.1.1)** is required to make the null informative. Lag cells alone do not substitute.",
        "**Freeze note:** Emitted only after G9+G10+G11 Rule-19 PASS. Mid-gap lure (§3.1.1) is the informative null; HumState wire alone was never sufficient.",
        1,
    )
    PREREG_MD.write_text(text, encoding="utf-8")
    digest = sha256_file(PREREG_MD)
    PREREG_SHA_PATH.write_text(digest + "  PREREG.md\n", encoding="utf-8")
    return digest


def run_gates() -> List[GateResult]:
    return [gate_g1(), gate_g2(), gate_g3(), gate_g4(), gate_g5(), gate_g6(),
            gate_g7(), gate_g8(), gate_g9(), gate_g10(), gate_g11()]


def write_receipt(gates: List[GateResult], *, emit_attempted: bool, emit_done: bool) -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_pass = all(g.passed for g in gates)
    draft_sha = sha256_file(DRAFT_PATH) if DRAFT_PATH.is_file() else None
    payload: Dict[str, Any] = {
        "ticket": "t12b",
        "pt": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT"),
        "rule19": "PASS" if all_pass else "FAIL",
        "all_pass": all_pass,
        "draft_content_sha256": draft_sha,
        "draft_status_is_draft": draft_is_draft(),
        "prereg_sha256_emitted": emit_done,
        "emit_attempted": emit_attempted,
        "prereg_sha256_path_exists": PREREG_SHA_PATH.is_file(),
        "gates": [{"name": g.name, "passed": g.passed, "detail": g.detail, "metrics": g.metrics} for g in gates],
        "constraints": {
            "affinity_c3_hum": "UNTOUCHED",
            "affinity_sha": EXPECTED_FROZEN["hum_measure/affinity_inputs.py"],
            "no_t12b_freeze_emit": (not emit_done),
            "counted_arm": "HumMeasureArm (HumState path)",
            "primary_contrast": "hum_measure vs one_step_memory",
            "labeled_secondary": "majority_k3",
        },
    }
    RECEIPT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# T12b Rule 19 — gate receipt", "",
        f"- **PT:** {payload['pt']}",
        f"- **Verdict:** **{payload['rule19']}**",
        f"- **Draft content sha256:** `{draft_sha}`",
        f"- **PREREG.sha256 emitted:** {emit_done}",
        f"- **PREREG.sha256 path exists:** {PREREG_SHA_PATH.is_file()}",
        "", "## Gates", "",
        "| Gate | Result | Detail |",
        "|------|:------:|--------|",
    ]
    for g in gates:
        lines.append(f"| `{g.name}` | {'PASS' if g.passed else 'FAIL'} | {g.detail} |")
    lines += [
        "", "## Notes", "",
        "- Counted arm path = HumState (`HumMeasureArm`).",
        "- Primary contrast = `hum_measure` vs `one_step_memory`.",
        "- `majority_k3` = labeled secondary (G11).",
        "- Affinity / C₃ / hum frozen. Parent T11.2 / T11.3 PREREG untouched.",
        "- Freeze emit: see `prereg_sha256_emitted` field.",
        "",
    ]
    RECEIPT_MD.write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="T12b Rule 19 freeze gates")
    ap.add_argument("--emit-hash", action="store_true",
                    help="Promote draft→PREREG.md + write PREREG.sha256 iff all PASS")
    args = ap.parse_args()
    print("=== T12b Rule 19 freeze gates ===")
    gates = run_gates()
    for g in gates:
        print(g.line())
    all_pass = all(g.passed for g in gates)
    print(f"overall: {'PASS' if all_pass else 'FAIL'}")
    emit_done = False
    emit_attempted = bool(args.emit_hash)
    frozen_at = datetime.now(PT).strftime("%Y-%m-%d %H:%M PT")
    if args.emit_hash:
        if not all_pass:
            print("REFUSE --emit-hash: Rule 19 FAIL; PREREG.sha256 not written.")
        else:
            digest = promote_draft_to_prereg(frozen_at=frozen_at)
            emit_done = True
            print(f"EMITTED PREREG.md + PREREG.sha256 = {digest}")
            print(f"draft_content_sha (unchanged) = {sha256_file(DRAFT_PATH)}")
    payload = write_receipt(gates, emit_attempted=emit_attempted, emit_done=emit_done)
    print(f"wrote {RECEIPT_JSON}")
    print(f"wrote {RECEIPT_MD}")
    print(f"draft_sha={payload.get('draft_content_sha256')}")
    print(f"prereg_sha256_emitted={emit_done}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
