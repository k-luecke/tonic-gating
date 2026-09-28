#!/usr/bin/env python3
"""One-seed T11 routing smoke across arms on sealed counting batch.

Usage:
  python3 smoke_run.py --seed 20260923

Does NOT freeze PREREG. Does NOT call specialist models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[4]  # /workspace/scp
T11_EXP = Path(__file__).resolve().parents[1]  # .../t11
sys.path.insert(0, str(T11_EXP))

from routing.hand_rule import HandRuleOrchestrator  # noqa: E402
from routing.classifier_stub import ClassifierStubOrchestrator  # noqa: E402
from routing.classifier_mlp import ClassifierMLPOrchestrator  # noqa: E402
from routing.hum_orchestrator import HumOrchestrator  # noqa: E402
from routing.hum_measure_arm import HumMeasureOrchestrator  # noqa: E402
from routing.observation_gate import sanitize_observation, assert_no_scoring_labels  # noqa: E402

SEALED_ALL = T11_EXP / "generators" / "sealed" / "ALL_SEQUENCES.jsonl"
SEALED_MANIFEST = T11_EXP / "generators" / "sealed" / "SEALED_MANIFEST.json"
HAND_RULE_MD = ROOT / "t11" / "HAND_RULE.md"
EXPECTED_ALL_SHA = "8f4d11bd961018c6fa39670271872e9a6fa4efebc36de7224138376addff6720"
EXPECTED_MANIFEST_SHA = "c9e8336e9da7db8302c0aeb9b247fed42bcb7416c09c102bad397fb0837b28f1"
EXPECTED_HAND_SHA = "28a3d07295ddf6558261b2f8c5065d6cf4934bc2c8457ef45cbbf081bc281334"

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DOCS_DIR = ROOT / "t11" / "docs"
RECEIPTS_DIR = DOCS_DIR / "receipts"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_sequences(path: Path, seed: Optional[int]) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if seed is None or row["seed"] == seed:
                rows.append(row)
    return rows


def thrash_count(preds: List[Optional[str]], oracle: List[str], spans: List[Optional[dict]]) -> int:
    """Unnecessary switch-away-and-back on distraction sequences.

    Simple segment-level proxy: if oracle stays same across consecutive segments
    but prediction leaves and returns, count 1 thrash event. Also if within a
    single distraction segment we only have one decision (segment-level smoke),
    thrash at segment granularity when pred != oracle while neighbors match.
    For smoke: count A→B→A patterns where oracle is A throughout the triple.
    """
    n = 0
    for i in range(len(preds) - 2):
        a, b, c = preds[i], preds[i + 1], preds[i + 2]
        oa, ob, oc = oracle[i], oracle[i + 1], oracle[i + 2]
        if a is None or b is None or c is None:
            continue
        if oa == oc == a == c and b != a and ob == oa:
            n += 1
        # classic thrash even if oracle mid differs? H2 is distraction spans —
        # prefer: oracle constant A,A,A and pred A,B,A
        if oa == ob == oc and a == c == oa and b != a:
            n += 1
    # dedupe double-count from both rules on same triple
    # recount cleanly:
    n = 0
    for i in range(len(preds) - 2):
        a, b, c = preds[i], preds[i + 1], preds[i + 2]
        oa, ob, oc = oracle[i], oracle[i + 1], oracle[i + 2]
        if None in (a, b, c):
            continue
        if oa == ob == oc and a == c and b != a:
            n += 1
    return n


def switch_latency(preds: List[Optional[str]], oracle: List[str]) -> List[int]:
    """Segments from regime change to first correct hot specialist."""
    latencies = []
    for i in range(1, len(oracle)):
        if oracle[i] != oracle[i - 1]:
            # find first j >= i where pred == oracle[i]
            found = None
            for j in range(i, len(preds)):
                if preds[j] == oracle[i]:
                    found = j - i
                    break
            if found is None:
                latencies.append(len(preds) - i)  # never recovered
            else:
                latencies.append(found)
    return latencies


def renewal_recall(preds: List[Optional[str]], oracle: List[str]) -> Optional[float]:
    """On returning regimes (oracle value reappears after absence), fraction correct."""
    seen = {}
    hits = 0
    total = 0
    for i, o in enumerate(oracle):
        if o in seen and seen[o] < i - 1:
            # returned after at least one other regime
            # check gap had different specialist
            gap = oracle[seen[o] + 1 : i]
            if gap and any(g != o for g in gap):
                total += 1
                if preds[i] == o:
                    hits += 1
        seen[o] = i
    if total == 0:
        return None
    return hits / total


def run_arm(
    orch: Any,
    sequences: List[Dict[str, Any]],
) -> Dict[str, Any]:
    orch.reset()
    per_seq = []
    # aggregate
    scored_correct = 0
    scored_total = 0  # excludes abstentions (M1)
    abstentions = 0
    by_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0, "abstain": 0})

    # H6
    m6_cells: Dict[str, Dict[str, Any]] = {}

    distraction_preds: List[Optional[str]] = []
    distraction_oracle: List[str] = []
    distraction_spans: List[Optional[dict]] = []

    all_latencies: List[int] = []
    renewal_scores: List[float] = []

    for seq in sequences:
        orch.reset()
        stype = seq["sequence_type"]
        segs = seq["segments"]
        wb = seq.get("window_break")
        break_at = wb["at"] if wb else None
        carry_mode = wb["carry_mode"] if wb else None
        placement = wb.get("placement") if wb else None

        preds: List[Optional[str]] = []
        oracles: List[str] = []
        details = []
        carry_info = None
        first_post_break = None

        for i, seg in enumerate(segs):
            text = seg["text"]
            oracle = seg["oracle_specialist"]
            oracles.append(oracle)

            post_break = False
            if break_at is not None and i == break_at:
                # window break BEFORE this segment (at boundary index)
                prior = segs[i - 1]["text"] if i > 0 else ""
                # For inside_task, break is still at segment index per generator meta
                carry_info = orch.on_window_break(carry_mode, prior_text=prior)
                post_break = True

            gate_obs = sanitize_observation(seg, text_fallback=text)
            assert_no_scoring_labels(gate_obs, where=f"{orch.name}.gate_obs")
            decision = orch.decide(text, post_break=post_break, observation=gate_obs)
            pred = decision.get("specialist")
            preds.append(pred)

            if decision.get("abstain"):
                abstentions += 1
                by_type[stype]["abstain"] += 1
                ok = None
            else:
                scored_total += 1
                by_type[stype]["total"] += 1
                ok = pred == oracle
                if ok:
                    scored_correct += 1
                    by_type[stype]["correct"] += 1

            if post_break:
                first_post_break = {
                    "pred": pred,
                    "oracle": oracle,
                    "correct": (pred == oracle) if pred is not None else False,
                    "abstain": bool(decision.get("abstain")),
                    "carry": carry_info,
                    "placement": placement,
                    "carry_mode": carry_mode,
                }

            details.append(
                {
                    "i": i,
                    "oracle": oracle,
                    "pred": pred,
                    "abstain": bool(decision.get("abstain")),
                    "abstain_reason": decision.get("abstain_reason"),
                    "post_break": post_break,
                    "used_carry": decision.get("used_carry"),
                    "family": seg.get("family"),
                }
            )

        # type-specific
        if stype == "Distraction":
            distraction_preds.extend(preds)
            distraction_oracle.extend(oracles)
            distraction_spans.extend([s.get("distraction_span") for s in segs])
        if stype in ("Clean", "Ambiguous", "Distraction", "Renewal"):
            all_latencies.extend(switch_latency(preds, oracles))
        if stype == "Renewal":
            rr = renewal_recall(preds, oracles)
            if rr is not None:
                renewal_scores.append(rr)

        if first_post_break is not None and carry_mode and placement:
            key = f"{carry_mode}|{placement}"
            cell = m6_cells.setdefault(
                key,
                {
                    "carry_mode": carry_mode,
                    "placement": placement,
                    "n": 0,
                    "correct": 0,
                    "bytes_list": [],
                },
            )
            cell["n"] += 1
            if first_post_break["correct"]:
                cell["correct"] += 1
            if carry_info:
                cell["bytes_list"].append(int(carry_info.get("bytes_carried", 0)))

        per_seq.append(
            {
                "seed": seq["seed"],
                "sequence_type": stype,
                "window_break": wb,
                "preds": preds,
                "oracles": oracles,
                "details": details,
                "first_post_break": first_post_break,
            }
        )

    m1 = (scored_correct / scored_total) if scored_total else None
    m1_by_type = {
        t: (v["correct"] / v["total"] if v["total"] else None) for t, v in sorted(by_type.items())
    }

    m2 = thrash_count(distraction_preds, distraction_oracle, distraction_spans)
    m3 = {
        "latencies": all_latencies,
        "mean": (sum(all_latencies) / len(all_latencies)) if all_latencies else None,
    }
    m4 = (sum(renewal_scores) / len(renewal_scores)) if renewal_scores else None

    m6 = {}
    for key, cell in sorted(m6_cells.items()):
        m6a = cell["correct"] / cell["n"] if cell["n"] else None
        bytes_mean = (
            sum(cell["bytes_list"]) / len(cell["bytes_list"]) if cell["bytes_list"] else 0.0
        )
        m6c = (m6a / bytes_mean) if (m6a is not None and bytes_mean > 0) else None
        m6[key] = {
            "carry_mode": cell["carry_mode"],
            "placement": cell["placement"],
            "M6a": m6a,
            "M6b_bytes_mean": bytes_mean,
            "M6c_csr_per_byte": m6c,
            "n": cell["n"],
            "correct": cell["correct"],
        }

    # M6d: stale-break penalty = M6a(at_task_boundary) - M6a(inside_task) per carry mode
    m6d = {}
    modes = sorted({c["carry_mode"] for c in m6_cells.values()})
    for mode in modes:
        atb = m6.get(f"{mode}|at_task_boundary", {}).get("M6a")
        ins = m6.get(f"{mode}|inside_task", {}).get("M6a")
        if atb is not None and ins is not None:
            m6d[mode] = {
                "M6a_at_task_boundary": atb,
                "M6a_inside_task": ins,
                "stale_penalty_boundary_minus_inside": atb - ins,
            }

    smoke_ok = scored_total > 0 and all(
        any(d["pred"] is not None or d["abstain"] for d in s["details"]) for s in per_seq
    )

    return {
        "arm": getattr(orch, "name", type(orch).__name__),
        "status": getattr(orch, "status", "unknown"),
        "disclosure": getattr(orch, "disclosure", None),
        "smoke_pass": bool(smoke_ok),
        "M1_CSR": m1,
        "M1_by_type": m1_by_type,
        "M1_counts": {"correct": scored_correct, "total": scored_total, "abstentions": abstentions},
        "M2_thrash_distraction": m2,
        "M3_switch_latency": m3,
        "M4_renewal_recall": m4,
        "M6": m6,
        "M6d_stale_break": m6d,
        "n_sequences": len(sequences),
        "per_sequence": per_seq,
        "by_type_raw": {k: dict(v) for k, v in by_type.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--all-seeds", action="store_true")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Verify seals
    all_sha = sha256_file(SEALED_ALL)
    man_sha = sha256_file(SEALED_MANIFEST)
    hand_sha = sha256_file(HAND_RULE_MD)
    seals = {
        "ALL_SEQUENCES.jsonl": {"path": str(SEALED_ALL), "sha256": all_sha, "expected": EXPECTED_ALL_SHA, "ok": all_sha == EXPECTED_ALL_SHA},
        "SEALED_MANIFEST.json": {"path": str(SEALED_MANIFEST), "sha256": man_sha, "expected": EXPECTED_MANIFEST_SHA, "ok": man_sha == EXPECTED_MANIFEST_SHA},
        "HAND_RULE.md": {"path": str(HAND_RULE_MD), "sha256": hand_sha, "expected": EXPECTED_HAND_SHA, "ok": hand_sha == EXPECTED_HAND_SHA},
    }
    if not all(v["ok"] for v in seals.values()):
        print("SEAL MISMATCH", json.dumps(seals, indent=2))
        return 2

    seed = None if args.all_seeds else args.seed
    sequences = load_sequences(SEALED_ALL, seed)
    if not sequences:
        print("No sequences for seed", seed)
        return 3

    arms = [
        HandRuleOrchestrator(),
        ClassifierStubOrchestrator(),
        ClassifierMLPOrchestrator(),
        HumOrchestrator(),  # retained: keyword-drive local baseline
        HumMeasureOrchestrator(),  # real hum arm (AffinityProvider + L1 + schema)
    ]

    arm_results = []
    for orch in arms:
        print(f"Running arm: {orch.name} ({orch.status}) …")
        arm_results.append(run_arm(orch, sequences))

    now = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S %Z")
    # Prefer PT label
    now_pt = datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + " PT"

    summary = {
        "batch_id": "t11-counting-sealed-v1",
        "smoke_id": "t11-routing-smoke-one-seed-v3-hum-measure",
        "created": now_pt,
        "seed": seed if seed is not None else "ALL",
        "n_sequences": len(sequences),
        "sequence_types": sorted({s["sequence_type"] for s in sequences}),
        "seals": seals,
        "prereg_frozen": False,
        "arms": [
            {
                "arm": r["arm"],
                "status": r["status"],
                "disclosure": r["disclosure"],
                "smoke_pass": r["smoke_pass"],
                "M1_CSR": r["M1_CSR"],
                "M1_by_type": r["M1_by_type"],
                "M1_counts": r["M1_counts"],
                "M2_thrash_distraction": r["M2_thrash_distraction"],
                "M3_mean_latency": r["M3_switch_latency"]["mean"],
                "M4_renewal_recall": r["M4_renewal_recall"],
                "M6": r["M6"],
                "M6d_stale_break": r["M6d_stale_break"],
            }
            for r in arm_results
        ],
        "blockers": [],
        "gnuradio_handoff": [
            "hum arm wired to t11/imports/hum_measure: L1 probe c3d70ed6…, AffinityProvider (Keyword fallback labeled), C_n3 40e3c0c9…, schema 4b55f40b….",
            "H5: typed_goal additive on sealed segments; typed_goal_family omitted (identity lock) → H5 inactive, L1 fallback.",
            "hum_local retained as disclosed keyword-drive baseline beside real hum arm.",
            "classifier_mlp calibrated + hashed (d482216c…); stub retained as disclosed baseline.",
            "Window-break carry modes are harness conditions; score M6a/M6c/M6d — hum has no special standing.",
            "T8 MoE .pt NOT loaded into T11 hum arm.",
        ],
        "ready_for_multi_seed": False,
        "ready_notes": (
            "One-seed smoke with calibrated classifier + real hum arm. Multi-seed still blocked on: "
            "(1) side-channel disclosure completed (labeled interim L2/stubs; see receipt), "
            "(2) PREREG still draft — do not freeze. "
            "GNURadio hum_measure package is landed and wired; KeywordAffinityProvider is labeled interim L2."
        ),
        "hum_measure_refs": {
            "probe_definition_sha256": "c3d70ed6778d6e6076add14144d0195e1f91888a8a50fb6f0b1c1e6657345fed",
            "coupling_n3_sha256": "40e3c0c95849ad67af6738e7842c1a4d24783dafa742ed247d03391db37a1493",
            "coupling_n8_sha256": "7dce0b7ee119add91859eb46a82fa2e94c821f7394468800a7f45a52e7ae2c8e",
            "schema_json_sha256": "4b55f40b9c87225926a7160974ad99f27ac5403cb7d99392c43172e87208f814",
            "classifier_mlp_v1_json_sha256": "d482216ce91423658851529cf78653cbebb034abe516d1a4d9c0402c0e490a59",
            "affinity_provider": "KeywordAffinityProvider (label=keyword_fallback)",
        },
    }

    # stub → calibrated delta (M1)
    stub_m1 = next((a["M1_CSR"] for a in summary["arms"] if a["arm"] == "classifier_stub"), None)
    mlp_m1 = next((a["M1_CSR"] for a in summary["arms"] if a["arm"] == "classifier_mlp"), None)
    summary["stub_to_calibrated_delta"] = {
        "classifier_stub_M1": stub_m1,
        "classifier_mlp_M1": mlp_m1,
        "delta_M1": (None if stub_m1 is None or mlp_m1 is None else mlp_m1 - stub_m1),
        "note": "Positive delta = calibrated MLP higher CSR than untrained stub on same smoke seed",
    }

    # blockers
    for r in arm_results:
        if not r["smoke_pass"]:
            summary["blockers"].append(f"arm {r['arm']} smoke_pass=false")
    if any(a["status"] == "stub" for a in summary["arms"]) and not any(
        a["arm"] == "classifier_mlp" and a["status"] == "calibrated" for a in summary["arms"]
    ):
        summary["blockers"].append("classifier_stub is untrained (disclosed) — not freeze-ready")
    elif any(a["status"] == "stub" for a in summary["arms"]):
        summary["blockers"].append(
            "classifier_stub retained as disclosed untrained baseline (calibrated arm present)"
        )
    if any(a["status"] == "local_dynamics_smoke" for a in summary["arms"]):
        summary["blockers"].append(
            "hum_local retained as disclosed keyword-drive baseline (real hum arm present)"
        )
    if any(a["arm"] == "hum" and a.get("status") == "gnuradio_hum_measure" for a in summary["arms"]):
        # Keyword L2 fallback is disclosed interim — not a freeze blocker by itself,
        # but side-channel disclosure remains open.
        pass
    summary["blockers"].append(
        "side-channel disclosure check COMPLETE — see t11/docs/T11_SIDE_CHANNEL_DISCLOSURE_RECEIPT.md (pass with labeled interim channels; PREREG still draft — do not freeze)"
    )

    # Full detail artifact
    detail = {
        **summary,
        "arm_details": arm_results,
    }
    detail_path = RESULTS_DIR / f"smoke_seed{seed if seed else 'ALL'}_detail.json"
    detail_bytes = json.dumps(detail, indent=2, sort_keys=True).encode("utf-8")
    detail_path.write_bytes(detail_bytes)
    detail_sha = sha256_bytes(detail_bytes)

    summary_path = RESULTS_DIR / f"smoke_seed{seed if seed else 'ALL'}_summary.json"
    summary_out = {**summary, "detail_path": str(detail_path), "detail_sha256": detail_sha}
    summary_bytes = json.dumps(summary_out, indent=2, sort_keys=True).encode("utf-8")
    summary_path.write_bytes(summary_bytes)
    summary_sha = sha256_bytes(summary_bytes)

    # Receipt markdown
    lines = [
        "# T11 routing smoke receipt — one-seed v3 (hum_measure wired)",
        "",
        "| Field | Value |",
        "|-------|--------|",
        f"| **Date** | {now_pt} |",
        f"| **Smoke id** | `t11-routing-smoke-one-seed-v3-hum-measure` |",
        f"| **Batch** | `t11-counting-sealed-v1` |",
        f"| **Seed** | `{seed}` |",
        f"| **Sequences** | {len(sequences)} (one seed of sealed 60) |",
        f"| **PREREG** | **not frozen** |",
        "",
        "## Seals verified",
        "",
        f"- `ALL_SEQUENCES.jsonl` sha256 `{all_sha}` OK",
        f"- `SEALED_MANIFEST.json` sha256 `{man_sha}` OK",
        f"- `HAND_RULE.md` sha256 `{hand_sha}` OK",
        "",
        "## Arms (pass = smoke completed with scored decisions)",
        "",
        "| Arm | Status | Smoke | M1 CSR | M2 thrash | M4 renewal |",
        "|-----|--------|-------|--------|-----------|------------|",
    ]
    for a in summary["arms"]:
        m1s = f"{a['M1_CSR']:.3f}" if a["M1_CSR"] is not None else "n/a"
        m4s = f"{a['M4_renewal_recall']:.3f}" if a["M4_renewal_recall"] is not None else "n/a"
        lines.append(
            f"| `{a['arm']}` | {a['status']} | "
            f"{'PASS' if a['smoke_pass'] else 'FAIL'} | "
            f"{m1s} | "
            f"{a['M2_thrash_distraction']} | "
            f"{m4s} |"
        )

    lines += [
        "",
        "### M1 by sequence type",
        "",
    ]
    for a in summary["arms"]:
        lines.append(f"**{a['arm']}:** " + ", ".join(
            f"{t}={v:.3f}" if v is not None else f"{t}=n/a" for t, v in a["M1_by_type"].items()
        ))
        lines.append("")

    lines += [
        "### M6 (window-break first-step CSR × carry mode)",
        "",
    ]
    for a in summary["arms"]:
        lines.append(f"**{a['arm']}:**")
        for key, cell in a["M6"].items():
            m6a = cell["M6a"]
            m6b = cell["M6b_bytes_mean"]
            m6c = cell["M6c_csr_per_byte"]
            lines.append(
                f"- `{key}`: M6a={m6a if m6a is None else round(m6a,3)}, "
                f"M6b_bytes={m6b:.1f}, M6c={m6c if m6c is None else round(m6c,6)}, n={cell['n']}"
            )
        lines.append("")

    lines += [
        "## Disclosures / stubs",
        "",
        "- **hand_rule**: full Kyle-signed rule (first-32, refuse-before-route, abstentions off M1).",
        "- **classifier_stub**: DISCLOSED untrained keyword-count argmax (alphabetical ties) + EMA buffer; baseline only.",
        "- **classifier_mlp**: calibrated 2-layer MLP; hashed weights; train 20260924–26, val 20260927, eval/smoke 20260923 held out.",
        "- **hum_local**: DISCLOSED local HumState n=3 + keyword projection baseline (retained).",
        "- **hum**: GNURadio hum_measure wired — L1 RegimeProbe + AffinityProvider "
        "(KeywordAffinityProvider **labeled fallback**) + schema serialize; H5 typed_goal "
        "field present but typed_goal_family omitted → L1 fallback; no T8 MoE .pt.",
        "- Sealed batch additive `typed_goal` (non-breaking); see T11_SEALED_TYPED_GOAL_ADDITIVE_RECEIPT.md.",
        "- No specialist model calls in this smoke (routing-only CSR vs `oracle_specialist`).",
        "- Oracle field used for scoring only; not fed to orchestrators.",
        "",
        "## Stub → calibrated delta",
        "",
        f"- classifier_stub M1: {summary['stub_to_calibrated_delta']['classifier_stub_M1']}",
        f"- classifier_mlp M1: {summary['stub_to_calibrated_delta']['classifier_mlp_M1']}",
        f"- delta (mlp − stub): {summary['stub_to_calibrated_delta']['delta_M1']}",
        "",
        "## Artifacts",
        "",
        f"- Summary: `{summary_path}` sha256 `{summary_sha}`",
        f"- Detail: `{detail_path}` sha256 `{detail_sha}`",
        "",
        "## Blockers / GNURadio handoff",
        "",
    ]
    for b in summary["blockers"]:
        lines.append(f"- {b}")
    lines.append("")
    for g in summary["gnuradio_handoff"]:
        lines.append(f"- {g}")
    lines += [
        "",
        "## Ready for multi-seed?",
        "",
        f"**{'Yes' if summary.get('ready_for_multi_seed') else 'No'}.** {summary['ready_notes']}",
        "",
        "Do not freeze PREREG.",
        "",
    ]

    receipt_md = "\n".join(lines)
    receipt_path = DOCS_DIR / "T11_ROUTING_SMOKE_RECEIPT.md"
    receipt_path.write_text(receipt_md, encoding="utf-8")
    receipt_sha = sha256_file(receipt_path)

    # Copy summary into receipts/
    receipt_json = RECEIPTS_DIR / "T11_ROUTING_SMOKE_RECEIPT.json"
    receipt_json_obj = {
        **summary_out,
        "receipt_md": str(receipt_path),
        "receipt_md_sha256": receipt_sha,
        "summary_sha256": summary_sha,
    }
    receipt_json.write_text(json.dumps(receipt_json_obj, indent=2, sort_keys=True), encoding="utf-8")

    # FILES.sha256 for smoke artifacts
    files_sha = RESULTS_DIR / "SMOKE_FILES.sha256"
    with open(files_sha, "w", encoding="utf-8") as f:
        for p, s in [
            (summary_path, summary_sha),
            (detail_path, detail_sha),
            (receipt_path, receipt_sha),
            (receipt_json, sha256_file(receipt_json)),
        ]:
            f.write(f"{s}  {p}\n")

    print(json.dumps({
        "ok": True,
        "seed": seed,
        "n_sequences": len(sequences),
        "arms": [(a["arm"], a["smoke_pass"], a["M1_CSR"]) for a in summary["arms"]],
        "receipt": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "summary": str(summary_path),
        "summary_sha256": summary_sha,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
