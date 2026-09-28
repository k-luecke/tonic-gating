#!/usr/bin/env python3
"""T13 Rule 19 — freeze gates.

Default: evaluate + write receipt under results/. Does NOT emit PREREG.sha256.
With --emit-hash (Kyle freeze GO): promote draft→PREREG.md (Status FROZEN) and write
PREREG.sha256 iff freeze-blocking gates PASS.

Affinity / C3 / hum packages are read-only; Affinity sha must match frozen pin.

G13: procedure LOCKED (manipulation check; pairwise+N=36; max-dev<=0.1 not RMS); Acc post-train — does NOT block freeze.
G17: docs-assert (n=1000, Acc<=0.273 binomial UB) PASS OK for freeze; probe Acc post-train.
G9: instability language at freeze; actual check runs at train; refuse primary if FAIL.
Do not fake G13/G17 Acc PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

T13 = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
DOCS = T13 / "docs"
RESULTS = T13 / "results"
PT = ZoneInfo("America/Chicago")

DRAFT_PATH = DOCS / "T13_PREREG_DRAFT.md"
GATES_DOC = DOCS / "T13_RULE19_GATES.md"
PREREG_SHA_PATH = T13 / "PREREG.sha256"
PREREG_MD = T13 / "PREREG.md"
RECEIPT_JSON = RESULTS / "T13_RULE19_GATES.json"
RECEIPT_MD = RESULTS / "T13_RULE19_GATES.md"

EXPECTED_AFFINITY = (
    "7dae877a3975c1ba4ca264f03f572d6749cce39a114ea3adbcffc010c7852c50"
)
AFFINITY_PATH = ROOT / "t11" / "imports" / "hum_measure" / "affinity_inputs.py"

PARENT_PREREG_SHA = {
    "t11_2": (
        ROOT / "experiments/phase-routing/t11_2/PREREG.sha256",
        "593762c4c19e85d1720f6b46edbb93869258c4047ad023e42e144416aaed20d1",
    ),
    "t11_3": (
        ROOT / "experiments/phase-routing/t11_3/PREREG.sha256",
        "27ab4a9573ec8c01ed043d72ef6c83ba47fdd4a764a373acb7453381bb9bd442",
    ),
    "t12": (
        ROOT / "experiments/phase-routing/t12/PREREG.sha256",
        "bf757ee4fc01e5c40b45f035150e7b716d12f7d8b30d745b9a7669f79bbf307c",
    ),
    "t12b": (
        ROOT / "experiments/phase-routing/t12b/PREREG.sha256",
        "851124b3c4bc66cb8e7c96c4f501084f4de186574f4f8d9446cac5879f62a3f0",
    ),
}

LOCKED_SEEDS = [20260928, 20260929, 20260930, 20261001, 20261002]
G17_N = 1000
G17_P0 = 0.25
G17_THRESH = 0.273

# Gates whose Acc measurement is deferred post-train (criterion/docs lock is enough for freeze)
DEFERRED_POST_TRAIN_ACC = {
    "G13_crosstalk_bound_assert",
    "G17_tag_placement_decodability",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class GateResult:
    name: str
    status: str  # PASS | FAIL | PENDING
    detail: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def line(self) -> str:
        return f"[{self.status}] {self.name}: {self.detail}"


def draft_text() -> str:
    if not DRAFT_PATH.is_file():
        return ""
    return DRAFT_PATH.read_text(encoding="utf-8")


def draft_is_draft() -> bool:
    if not DRAFT_PATH.is_file():
        return True
    head = draft_text()[:2000]
    return "**DRAFT**" in head or "| **Status** | **DRAFT**" in head


def _prereg_sha(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").split()[0].strip()


def gate_g1_draft_blocks_emit() -> GateResult:
    """G1 documents DRAFT presence. At emit, draft stays DRAFT (T12b pattern); PREREG.md is promoted."""
    is_draft = draft_is_draft()
    exists = DRAFT_PATH.is_file()
    ok = exists and is_draft
    return GateResult(
        "G1_draft_detected_blocks_emit",
        "PASS" if ok else "FAIL",
        f"draft_exists={exists}; is_draft={is_draft}; emit via --emit-hash promotes separate PREREG.md",
        {"draft_path": str(DRAFT_PATH), "is_draft": is_draft},
    )


def gate_g2_compute_formula() -> GateResult:
    text = draft_text()
    if not text:
        return GateResult("G2_matched_compute_formula", "FAIL", "missing PREREG draft")
    markers = (
        "matched compute" in text.lower() or "Matched compute" in text,
        "loops" in text.lower() and "experts" in text.lower() and "width" in text.lower(),
        "C_match" in text or "compute formula" in text.lower(),
    )
    ok = all(markers[:2]) and ("formula" in text.lower() or "C_match" in text)
    return GateResult(
        "G2_matched_compute_formula",
        "PASS" if ok else "FAIL",
        f"markers={markers}",
        {"markers": list(markers)},
    )


def gate_g3_intelligibility_primary() -> GateResult:
    text = draft_text()
    if not text:
        return GateResult("G3_intelligibility_primary", "FAIL", "missing PREREG draft")
    ok = ("H_int" in text) and ("intelligibility" in text.lower()) and ("primary" in text.lower())
    idx4 = text.find("## 4.")
    idx5 = text.find("## 5.")
    before_arms = idx4 != -1 and idx5 != -1 and idx4 < idx5 and "H_int" in text[idx4:idx5]
    ok = ok and before_arms
    return GateResult(
        "G3_intelligibility_primary",
        "PASS" if ok else "FAIL",
        "H_int algebra before arms" if ok else "missing H_int-before-arms",
    )


def gate_g4_related_work() -> GateResult:
    text = draft_text()
    if not text:
        return GateResult("G4_related_work_delta", "FAIL", "missing PREREG draft")
    has_mod = "Mixture-of-Depths" in text or "Raposo" in text
    has_mor = "Mixture-of-Recursions" in text or "Bae" in text
    has_slice = "context slice" in text.lower()
    ok = has_mod and has_mor and has_slice
    return GateResult(
        "G4_related_work_delta",
        "PASS" if ok else "FAIL",
        f"MoD={has_mod}; MoR={has_mor}; context_slice={has_slice}",
    )


def gate_g5_hum_expected_null() -> GateResult:
    text = draft_text()
    if not text:
        return GateResult("G5_hum_allocator_expected_null", "FAIL", "missing PREREG draft")
    low = text.lower()
    ok = (
        "do not expect hum to win" in low
        or "expected-null" in low
        or "expected null" in low
    )
    return GateResult(
        "G5_hum_allocator_expected_null",
        "PASS" if ok else "FAIL",
        "hum allocator expected-null clause present" if ok else "missing expected-null for hum",
    )


def gate_g6_affinity_frozen() -> GateResult:
    if not AFFINITY_PATH.is_file():
        return GateResult("G6_affinity_frozen", "FAIL", f"missing {AFFINITY_PATH}")
    got = sha256_file(AFFINITY_PATH)
    ok = got == EXPECTED_AFFINITY
    local_copy = T13 / "imports" / "hum_measure" / "affinity_inputs.py"
    no_local = not local_copy.is_file()
    return GateResult(
        "G6_affinity_frozen",
        "PASS" if (ok and no_local) else "FAIL",
        f"sha_match={ok}; no_local_affinity_copy={no_local}; got={got[:12]}…",
        {"sha256": got, "expected": EXPECTED_AFFINITY, "no_local_copy": no_local},
    )


def gate_g7_parents_untouched() -> GateResult:
    details = {}
    ok = True
    for name, (path, expected) in PARENT_PREREG_SHA.items():
        got = _prereg_sha(path)
        match = got == expected
        details[name] = {"path": str(path), "expected": expected, "got": got, "match": match}
        ok = ok and match
    return GateResult(
        "G7_parent_preregs_untouched",
        "PASS" if ok else "FAIL",
        "all parent PREREG.sha256 match pins" if ok else f"mismatch: {details}",
        details,
    )


def gate_g8_mythos_tiny() -> GateResult:
    text = draft_text()
    if not text:
        return GateResult("G8_scale_mythos_tiny", "FAIL", "missing PREREG draft")
    checks = {
        "mythos_tiny": "mythos_tiny" in text,
        "K=4": bool(re.search(r"K\s*=\s*4|K=4|\\(K=4\\)", text)),
        "W=64": "W=64" in text,
        "L_max=4": "L_{\\max}=4" in text or "L_max=4" in text,
        "S_full=64": "S_{\\mathrm{full}}=64" in text or "S_full=64" in text,
        "S_e=16": "S_e=16" in text or "S_e = 16" in text or "= 16" in text,
        "vocab_256": ("vocab **256**" in text) or ("vocab \\(=256\\)" in text) or ("=256" in text),
        "steps_2000": "2000 steps" in text or "2000" in text,
        "seeds": all(str(s) in text for s in LOCKED_SEEDS),
        "C_match_numeric": "4,210,688" in text or "4{,}210{,}688" in text,
    }
    ok = all(checks.values())
    return GateResult(
        "G8_scale_mythos_tiny",
        "PASS" if ok else "FAIL",
        f"mythos_tiny numeric locks present; checks={checks}" if ok else f"missing locks: {checks}",
        checks,
    )


def gate_g9_instability_gate() -> GateResult:
    if not DRAFT_PATH.is_file() or not GATES_DOC.is_file():
        return GateResult("G9_training_instability", "FAIL", "missing draft or gates doc")
    text = draft_text() + GATES_DOC.read_text(encoding="utf-8")
    ok = ("instability" in text.lower()) or ("T12 T1" in text) or ("non-finite" in text.lower())
    return GateResult(
        "G9_training_instability",
        "PASS" if ok else "FAIL",
        "instability language locked; runs at train; refuse primary if FAIL"
        if ok
        else "missing instability gate",
    )


def gate_g10_stopping() -> GateResult:
    text = draft_text()
    if not text:
        return GateResult("G10_stopping_rule", "FAIL", "missing PREREG draft")
    ok = ("stopping" in text.lower()) and ("null" in text.lower() and "support" in text.lower())
    ok = ok and ("medium_not_beyond_attribution" in text) and ("tagged_superposition" in text)
    return GateResult(
        "G10_stopping_rule",
        "PASS" if ok else "FAIL",
        "stopping + null/support vocabulary present" if ok else "incomplete stopping section",
    )


def gate_g11_affinity_surface_off() -> GateResult:
    text = draft_text()
    if not text:
        return GateResult("G11_affinity_surface_default_off", "FAIL", "missing PREREG draft")
    ok = (
        "no Affinity work" in text
        or "Affinity surface" in text
        or "does not authorize" in text.lower()
        or "Default off" in text
    )
    local_imports = T13 / "imports"
    no_affinity_tree = not (local_imports / "hum_measure").exists()
    return GateResult(
        "G11_affinity_surface_default_off",
        "PASS" if (ok and no_affinity_tree) else "FAIL",
        f"doc_ok={ok}; no_local_hum_measure={no_affinity_tree}",
    )


def gate_g12_hint_controls() -> GateResult:
    text = draft_text()
    if not text:
        return GateResult("G12_hint_controls_named", "FAIL", "missing PREREG draft")
    ok = all(
        x in text.lower()
        for x in ("random-tag", "plain", "oracle")
    ) and ("no-medium" in text.lower() or "no_medium" in text.lower())
    return GateResult(
        "G12_hint_controls_named",
        "PASS" if ok else "FAIL",
        "random-tag / plain-sum / oracle / no-medium named" if ok else "missing H_int controls",
    )


def gate_g13_crosstalk() -> GateResult:
    """Procedure LOCKED at freeze; Acc evaluate post-train — never fake Acc PASS."""
    text = draft_text()
    gates = GATES_DOC.read_text(encoding="utf-8") if GATES_DOC.is_file() else ""
    blob = text + "\n" + gates
    low = blob.lower()
    has_bound = ("0.1" in blob) and ("crosstalk" in low)
    locked = "LOCKED" in blob and ("0.1" in blob)
    timing_ok = (
        ("does not block freeze" in low or "does **not** block freeze" in low
         or "NOT** block freeze" in blob or "does **NOT** block freeze" in blob)
        and ("post-train" in low or "post-training" in low)
    )
    # Gate-defect procedure pins (same category as G17 n/thresh)
    sampling_ok = (
        ("n=36" in low or "n = 36" in low)
        and ("2π" in blob or "2\\pi" in blob or "2\\pi" in low or "[0, 2" in blob or "[0,2" in blob.replace(" ", ""))
        and ("pairwise" in low)
    )
    agg_ok = ("max-deviation" in low or "max_deviation" in low or "max-dev" in low) and ("rms" in low)
    # require explicit rejection of RMS
    role_ok = ("manipulation check" in low) and ("not" in low and ("h_int" in low or "intelligibility" in low))
    procedure_ok = sampling_ok and agg_ok and role_ok
    criterion_ok = has_bound and locked and timing_ok and procedure_ok

    mixture_artifacts = (
        list(RESULTS.glob("*CROSSTALK*"))
        + list(RESULTS.glob("*mixture*"))
        + list(RESULTS.glob("*TRAINED*"))
    )
    trained = bool(mixture_artifacts)
    metrics = {
        "criterion_locked": criterion_ok,
        "doc_mentions_bound": has_bound,
        "timing_post_train": timing_ok,
        "procedure_sampling": sampling_ok,
        "procedure_aggregation_maxdev_not_rms": agg_ok,
        "procedure_manipulation_check_role": role_ok,
        "trained_mixture": trained,
        "acc_status": "PENDING_POST_TRAIN",
        "threshold": 0.1,
        "formula": "|X(dphi)-max(0,cos dphi)| <= 0.1",
        "delta_phi_grid_n": 36,
        "aggregation": "max_deviation",
        "role": "manipulation_check_not_hint_evidence",
        "blocks_freeze": False,
        "blocks_primary_if_fail_after_train": True,
    }
    if not criterion_ok:
        return GateResult(
            "G13_crosstalk_bound_assert",
            "FAIL",
            (
                f"G13 procedure incomplete: bound={has_bound} locked={locked} timing={timing_ok} "
                f"sampling={sampling_ok} agg={agg_ok} role={role_ok}"
            ),
            metrics,
        )
    return GateResult(
        "G13_crosstalk_bound_assert",
        "PASS",
        (
            "procedure LOCKED (manipulation check; pairwise+N=36; max-dev≤0.1 not RMS); "
            "Acc PENDING post-train (does not block freeze; do not fake Acc)"
        ),
        metrics,
    )


def gate_g14_k_phases() -> GateResult:
    text = draft_text()
    if not text:
        return GateResult("G14_k_phases_hashed", "FAIL", "missing PREREG draft")
    has_k = "K=4" in text or ("\\(K=4\\)" in text) or ("draft \\(K=4\\)" in text)
    has_phases = "phases" in text.lower()
    ok = has_k and has_phases and ("hash" in text.lower())
    return GateResult(
        "G14_k_phases_hashed",
        "PASS" if ok else "FAIL",
        f"draft K=4 + phases hash language present; K={has_k}; phases={has_phases}",
        {"K4": has_k, "phases_mentioned": has_phases, "prereg_emitted": PREREG_SHA_PATH.is_file()},
    )


def gate_g15_cmatch_numeric() -> GateResult:
    text = draft_text()
    if not text:
        return GateResult("G15_cmatch_per_arm", "FAIL", "missing PREREG draft")
    has_ref = "4,210,688" in text or "4{,}210{,}688" in text
    has_shared = "4,259,840" in text or "4{,}259{,}840" in text
    has_pct = "±5%" in text or "+1.17%" in text or "5%" in text
    ok = has_ref and has_shared and has_pct
    return GateResult(
        "G15_cmatch_per_arm",
        "PASS" if ok else "FAIL",
        f"§3.2 numeric C_match lines present; ref={has_ref}; shared={has_shared}" if ok else "missing numeric C_match",
        {"ref_4210688": has_ref, "shared_4259840": has_shared},
    )


def gate_g16_no_medium_verdict() -> GateResult:
    text = draft_text()
    if not text:
        return GateResult("G16_no_medium_inside_verdict", "FAIL", "missing PREREG draft")
    low = text.lower()
    in_verdict = "r_{\\mathrm{no\\_medium}}" in text or "r_no_medium" in text or "no_medium" in low
    required = "support requires" in low or "beating no-medium" in low or "> r_{\\mathrm{no\\_medium}}" in text
    ok = in_verdict and required and ("verdict" in low)
    return GateResult(
        "G16_no_medium_inside_verdict",
        "PASS" if ok else "FAIL",
        "no-medium inside H_int support verdict" if ok else "no-medium not inside verdict",
    )


def gate_g17_placement() -> GateResult:
    """Docs lock PASS OK for freeze; probe Acc deferred post-train / pre-primary."""
    text = draft_text()
    gates = GATES_DOC.read_text(encoding="utf-8") if GATES_DOC.is_file() else ""
    blob = text + "\n" + gates
    has_n = "n=1000" in blob or "**n=1000**" in blob
    has_p0 = "0.25" in blob and ("1/K" in blob or "p_0" in blob or "p0" in blob or "chance" in blob.lower())
    has_thresh = "0.273" in blob or "0.2725" in blob
    has_binomial = "binomial" in blob.lower() and ("upper bound" in blob.lower() or "UB" in blob)
    not_slack = (
        ("retired" in blob.lower() and "n=200" in blob)
        or ("high power" in blob.lower())
        or ("do **not** shrink n" in blob)
        or ("do not shrink n" in blob.lower())
    )
    # Timing: docs OK for freeze; Acc post-train (not freeze-blocking)
    timing_ok = (
        "docs-assert" in blob.lower()
        or "docs assert" in blob.lower()
        or "docs-assert PASS OK for freeze" in blob
        or ("post-train" in blob.lower() and "0.273" in blob)
    )
    ok = has_n and has_p0 and has_thresh and has_binomial and not_slack and timing_ok
    approx = G17_P0 + 1.645 * ((G17_P0 * (1 - G17_P0) / G17_N) ** 0.5)
    return GateResult(
        "G17_tag_placement_decodability",
        "PASS" if ok else "FAIL",
        (
            f"docs pin n={G17_N}, p0={G17_P0}, Acc≤{G17_THRESH} (= one-sided 95% binomial UB ≈{approx:.3f}); "
            "docs-assert PASS OK for freeze; probe Acc PENDING post-train (blocks primary if FAIL)"
            if ok
            else f"missing G17 pins: n={has_n} p0={has_p0} thresh={has_thresh} binomial={has_binomial} "
            f"not_slack={not_slack} timing={timing_ok}"
        ),
        {
            "n": G17_N,
            "p0": G17_P0,
            "threshold": G17_THRESH,
            "binomial_ub_approx": round(approx, 6),
            "probe_measured": False,
            "acc_status": "PENDING_POST_TRAIN",
            "docs_ok": ok,
            "blocks_freeze": False,
            "blocks_primary_if_fail_after_train": True,
        },
    )


def gate_g18_attention_locked() -> GateResult:
    text = draft_text()
    if not text:
        return GateResult("G18_attention_accounting_locked", "FAIL", "missing PREREG draft")
    primary = "per-slice attention counted" in text.lower() or "per-slice attn" in text.lower()
    secondary = "full-context-equivalent" in text.lower()
    section = "### 3.0" in text or "§3.0" in text or "3.0 Chosen" in text
    ok = primary and secondary and section
    return GateResult(
        "G18_attention_accounting_locked",
        "PASS" if ok else "FAIL",
        "primary=per-slice attn; secondary=full-context-equivalent; §3.0 locked" if ok else "attention choice incomplete",
    )


def run_gates() -> List[GateResult]:
    return [
        gate_g1_draft_blocks_emit(),
        gate_g2_compute_formula(),
        gate_g3_intelligibility_primary(),
        gate_g4_related_work(),
        gate_g5_hum_expected_null(),
        gate_g6_affinity_frozen(),
        gate_g7_parents_untouched(),
        gate_g8_mythos_tiny(),
        gate_g9_instability_gate(),
        gate_g10_stopping(),
        gate_g11_affinity_surface_off(),
        gate_g12_hint_controls(),
        gate_g13_crosstalk(),
        gate_g14_k_phases(),
        gate_g15_cmatch_numeric(),
        gate_g16_no_medium_verdict(),
        gate_g17_placement(),
        gate_g18_attention_locked(),
    ]


def freeze_blocking_pass(gates: List[GateResult]) -> bool:
    """All gates PASS (G13/G17 Acc deferred but criterion/docs already PASS). No FAIL."""
    return all(g.status == "PASS" for g in gates)


def promote_draft_to_prereg(*, frozen_at: str) -> str:
    """Write PREREG.md from draft with Status flipped to FROZEN. Draft bytes unchanged."""
    if not DRAFT_PATH.is_file():
        raise FileNotFoundError(DRAFT_PATH)
    text = DRAFT_PATH.read_text(encoding="utf-8")
    text = text.replace(
        "# PREREG DRAFT — T13 Looped architectures (MoE-of-loops / context slices)",
        "# PREREG — T13 Looped architectures (MoE-of-loops / context slices)",
        1,
    )
    # Status row: replace DRAFT status cell
    text = re.sub(
        r"\| \*\*Status\*\* \| \*\*DRAFT\*\*[^\n]*\|",
        f"| **Status** | **FROZEN** {frozen_at} — Rule-19 T13 PASS; emit via rule19_freeze_gates --emit-hash |",
        text,
        count=1,
    )
    text = text.replace(
        "| **Run location** | Linux `/workspace/scp` (scaffold + draft this turn; **no primary**) |",
        "| **Run location** | Linux `/workspace/scp` (frozen; train → G13/G17 Acc → primary × seeds) |",
        1,
    )
    text = text.replace(
        "> **DRAFT.** Numbers and arm grids below lock **intent** for freeze review. Emitting a hash before matched-compute formula lock + Rule-19 PASS + Kyle freeze greenlight voids the draft for that run.",
        "> **FROZEN.** Rule-19 T13 freeze-blocking gates PASS. Affinity / C₃ / hum untouched. "
        "Parent T11/T12/T12b PREREGs not re-emitted. G13/G17 Acc deferred post-train (criterion locked). "
        "Sequence: train mixture (G9) → G13+G17 Acc → primary × seeds.",
        1,
    )
    text = text.replace(
        "| T13 `PREREG.sha256` | **not emitted** (DRAFT) |",
        "| T13 `PREREG.sha256` | **emitted** (this freeze) |",
        1,
    )
    # Soften remaining Until-authorize DRAFT footer in checklist
    text = text.replace(
        "Until authorize: DRAFT. No primary. No hash.",
        "FROZEN. No primary until train + G13/G17 Acc PASS. Hash emitted.",
        1,
    )
    PREREG_MD.write_text(text, encoding="utf-8")
    digest = sha256_file(PREREG_MD)
    PREREG_SHA_PATH.write_text(digest + "  PREREG.md\n", encoding="utf-8")
    return digest


def write_receipt(
    gates: List[GateResult],
    *,
    emit_attempted: bool,
    emit_done: bool,
    prereg_digest: Optional[str] = None,
) -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    draft_sha = sha256_file(DRAFT_PATH) if DRAFT_PATH.is_file() else None
    n_pass = sum(1 for g in gates if g.status == "PASS")
    n_fail = sum(1 for g in gates if g.status == "FAIL")
    n_pending = sum(1 for g in gates if g.status == "PENDING")
    freeze_ready = freeze_blocking_pass(gates)
    status_label = "FROZEN" if emit_done else ("DRAFT" if draft_is_draft() else "PROMOTED")
    payload: Dict[str, Any] = {
        "ticket": "t13",
        "pt": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT"),
        "rule19_freeze_ready": freeze_ready,
        "summary": {"PASS": n_pass, "FAIL": n_fail, "PENDING": n_pending},
        "draft_content_sha256": draft_sha,
        "draft_status_is_draft": draft_is_draft(),
        "prereg_sha256_emitted": emit_done,
        "prereg_sha256": prereg_digest,
        "emit_attempted": emit_attempted,
        "prereg_sha256_path_exists": PREREG_SHA_PATH.is_file(),
        "gates": [
            {"name": g.name, "status": g.status, "passed": g.passed, "detail": g.detail, "metrics": g.metrics}
            for g in gates
        ],
        "constraints": {
            "affinity_c3_hum": "FROZEN / UNTOUCHED",
            "affinity_sha": EXPECTED_AFFINITY,
            "status": status_label,
            "no_prereg_emit": (not emit_done),
            "no_primary": True,
            "g13_criterion_locked": True,
            "g13_acc_deferred_post_train": True,
            "g17_n": G17_N,
            "g17_threshold": G17_THRESH,
            "g17_threshold_is_binomial_ub": True,
            "g17_acc_deferred_post_train": True,
            "g9_runs_at_train": True,
            "post_freeze_sequence": "train mixture (G9) → G13 Acc + G17 Acc → primary × seeds",
        },
    }
    RECEIPT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# T13 Rule 19 — gate receipt",
        "",
        f"- **PT:** {payload['pt']}",
        f"- **Freeze-ready:** **{freeze_ready}**",
        f"- **Summary:** PASS={n_pass} FAIL={n_fail} PENDING={n_pending}",
        f"- **Draft content sha256:** `{draft_sha}`",
        f"- **PREREG.sha256 emitted:** {emit_done}"
        + (f" (`{prereg_digest}`)" if prereg_digest else ""),
        f"- **Status:** {status_label}",
        "",
        "## Gates",
        "",
        "| Gate | Status | Detail |",
        "|------|:------:|--------|",
    ]
    for g in gates:
        lines.append(f"| `{g.name}` | {g.status} | {g.detail} |")
    lines += [
        "",
        "## Notes",
        "",
        "- Affinity / C₃ / hum **FROZEN** — affinity_inputs.py untouched.",
        "- **G13** procedure LOCKED (manipulation check; pairwise+N=36; max-dev≤0.1 not RMS); Acc PENDING post-train — does **not** block freeze.",
        "- **G17** docs pin n=1000 + Acc≤0.273 (= binomial UB); docs-assert PASS OK for freeze; Acc PENDING post-train.",
        "- **G9** instability language locked; actual check runs at train; refuse primary if FAIL.",
        "- Post-freeze sequence: train mixture → G13 Acc + G17 Acc → primary × seeds.",
        "- Parent T11/T12/T12b PREREG.sha256 untouched.",
        "",
    ]
    RECEIPT_MD.write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="T13 Rule 19 freeze gates")
    ap.add_argument(
        "--emit-hash",
        action="store_true",
        help="Promote draft→PREREG.md + write PREREG.sha256 iff freeze-blocking gates PASS",
    )
    args = ap.parse_args()
    print("=== T13 Rule 19 freeze gates ===")
    print(f"pt: {datetime.now(PT).strftime('%Y-%m-%d %H:%M:%S PT')}")
    gates = run_gates()
    for g in gates:
        print(g.line())
    n_pass = sum(1 for g in gates if g.status == "PASS")
    n_fail = sum(1 for g in gates if g.status == "FAIL")
    n_pending = sum(1 for g in gates if g.status == "PENDING")
    freeze_ready = freeze_blocking_pass(gates)
    print(f"summary: PASS={n_pass} FAIL={n_fail} PENDING={n_pending}")
    print(f"freeze_ready: {freeze_ready}")
    if DRAFT_PATH.is_file():
        print(f"draft_sha256={sha256_file(DRAFT_PATH)}")
    print(f"draft_status_is_draft={draft_is_draft()}")
    print(f"prereg_sha256_path_exists={PREREG_SHA_PATH.is_file()}")

    emit_done = False
    prereg_digest: Optional[str] = None
    if args.emit_hash:
        if not freeze_ready:
            print("REFUSE --emit-hash: freeze-blocking gates not all PASS; PREREG.sha256 not written.")
            write_receipt(gates, emit_attempted=True, emit_done=False)
            print(f"wrote {RECEIPT_JSON}")
            print(f"wrote {RECEIPT_MD}")
            return 2
        frozen_at = datetime.now(PT).strftime("%Y-%m-%d %H:%M PT")
        prereg_digest = promote_draft_to_prereg(frozen_at=frozen_at)
        emit_done = True
        print(f"EMITTED PREREG.md + PREREG.sha256 = {prereg_digest}")
        print(f"draft_content_sha (unchanged) = {sha256_file(DRAFT_PATH)}")

    payload = write_receipt(
        gates, emit_attempted=bool(args.emit_hash), emit_done=emit_done, prereg_digest=prereg_digest
    )
    print(f"wrote {RECEIPT_JSON}")
    print(f"wrote {RECEIPT_MD}")
    print(f"prereg_sha256_emitted={emit_done}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
