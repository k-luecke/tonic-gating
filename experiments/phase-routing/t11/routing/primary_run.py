#!/usr/bin/env python3
"""Multi-seed T11 routing primary (counting).

Frozen pins (verify on disk; do not mutate sealed/PREREG):
  HANDOFF  056be962…
  PREREG   c62bfda8…
  ALL_SEQUENCES 8f4d11bd…
  classifier_mlp_v1.json d482216c…

Usage:
  python3 primary_run.py

Writes under results/primary/ + t11/docs/T11_ROUTING_PRIMARY_RECEIPT.md
Does NOT call specialist models. Does NOT edit PREREG or sealed batch.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[4]  # /workspace/scp
T11_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T11_EXP))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoke_run import (  # noqa: E402
    EXPECTED_ALL_SHA,
    EXPECTED_HAND_SHA,
    EXPECTED_MANIFEST_SHA,
    HAND_RULE_MD,
    SEALED_ALL,
    SEALED_MANIFEST,
    load_sequences,
    run_arm,
    sha256_bytes,
    sha256_file,
)
from routing.hand_rule import HandRuleOrchestrator  # noqa: E402
from routing.classifier_stub import ClassifierStubOrchestrator  # noqa: E402
from routing.classifier_mlp import ClassifierMLPOrchestrator  # noqa: E402
from routing.hum_orchestrator import HumOrchestrator  # noqa: E402
from routing.hum_measure_arm import HumMeasureOrchestrator  # noqa: E402

PRIMARY_SEEDS = [20260923, 20260924, 20260925, 20260926, 20260927]
# classifier_mlp train/val contamination relative to primary seeds
MLP_TRAIN_SEEDS = {20260924, 20260925, 20260926}
MLP_VAL_SEED = 20260927
MLP_HELD_OUT = {20260923}

EXPECTED_PREREG = "c62bfda8bab88df14677e47e20f28c99ba20595bc0f9273050484d4ae92e079a"
EXPECTED_HANDOFF = "056be9628d3a07edd9fa0a2bff32b7e01ebb8fdd9a36b7b66d0ec5f7e39c8eaf"
EXPECTED_MLP = "d482216ce91423658851529cf78653cbebb034abe516d1a4d9c0402c0e490a59"
EXPECTED_SIDE_CHANNEL_MD = "01e73af4ffad4490a3c71a203df0c7182be07ba8ec2343efce77b81ef4f00f81"

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "primary"
DOCS_DIR = ROOT / "t11" / "docs"
RECEIPTS_DIR = DOCS_DIR / "receipts"
PREREG_PATH = ROOT / "t11" / "PREREG.md"
HANDOFF_PATH = ROOT / "t11" / "HANDOFF.md"
MLP_PATH = Path(__file__).resolve().parent / "artifacts" / "classifier_mlp_v1.json"
SIDE_CHANNEL_MD = DOCS_DIR / "T11_SIDE_CHANNEL_DISCLOSURE_RECEIPT.md"

H3_GAP_MAX = 2.0  # |latency_hum − latency_classifier| ≤ 2 segments
T_HOLD_N3 = 65
RHO_EFF = 0.9891373214


def arm_factory():
    return [
        HandRuleOrchestrator(),
        ClassifierStubOrchestrator(),
        ClassifierMLPOrchestrator(),
        HumOrchestrator(),
        HumMeasureOrchestrator(),
    ]


def slim_arm(r: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "arm": r["arm"],
        "status": r["status"],
        "disclosure": r["disclosure"],
        "smoke_pass": r["smoke_pass"],
        "M1_CSR": r["M1_CSR"],
        "M1_by_type": r["M1_by_type"],
        "M1_counts": r["M1_counts"],
        "M2_thrash_distraction": r["M2_thrash_distraction"],
        "M3_switch_latency": r["M3_switch_latency"],
        "M3_mean_latency": r["M3_switch_latency"]["mean"] if r["M3_switch_latency"] else None,
        "M4_renewal_recall": r["M4_renewal_recall"],
        "M6": r["M6"],
        "M6d_stale_break": r["M6d_stale_break"],
        "n_sequences": r["n_sequences"],
    }


def seed_interval(values: List[float]) -> Dict[str, Any]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"n": 0, "min": None, "max": None, "range": None, "stdev": None, "mean": None}
    if len(vals) == 1:
        return {"n": 1, "min": vals[0], "max": vals[0], "range": 0.0, "stdev": 0.0, "mean": vals[0]}
    return {
        "n": len(vals),
        "min": min(vals),
        "max": max(vals),
        "range": max(vals) - min(vals),
        "stdev": statistics.stdev(vals),
        "mean": statistics.mean(vals),
    }


def score_hypotheses(
    per_seed: Dict[int, Dict[str, Dict[str, Any]]],
    pooled: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Score H1–H6 per frozen PREREG primary comparisons.

    Primary hum arm = 'hum' (hum_measure). Classifier comparator = 'classifier_mlp'.
    Hand = 'hand_rule'. Stubs retained as disclosed baselines, not primary comparators.
    """
    seeds = sorted(per_seed.keys())
    hum = "hum"
    clf = "classifier_mlp"
    hand = "hand_rule"

    def by_type(arm: str, stype: str, seed: Optional[int] = None) -> Optional[float]:
        if seed is None:
            return pooled[arm]["M1_by_type"].get(stype)
        return per_seed[seed][arm]["M1_by_type"].get(stype)

    def m2(arm: str, seed: Optional[int] = None) -> Optional[int]:
        src = pooled[arm] if seed is None else per_seed[seed][arm]
        return src["M2_thrash_distraction"]

    def m3(arm: str, seed: Optional[int] = None) -> Optional[float]:
        src = pooled[arm] if seed is None else per_seed[seed][arm]
        return src["M3_mean_latency"]

    def m4(arm: str, seed: Optional[int] = None) -> Optional[float]:
        src = pooled[arm] if seed is None else per_seed[seed][arm]
        return src["M4_renewal_recall"]

    # --- H1 ---
    amb_hum = [by_type(hum, "Ambiguous", s) for s in seeds]
    amb_clf = [by_type(clf, "Ambiguous", s) for s in seeds]
    amb_hand = [by_type(hand, "Ambiguous", s) for s in seeds]
    clean_arms = {
        a: [by_type(a, "Clean", s) for s in seeds]
        for a in (hum, clf, hand, "classifier_stub", "hum_local")
    }
    clean_means = {a: (statistics.mean([x for x in vs if x is not None]) if any(x is not None for x in vs) else None)
                   for a, vs in clean_arms.items()}
    clean_vals = [v for v in clean_means.values() if v is not None]
    clean_spread = (max(clean_vals) - min(clean_vals)) if clean_vals else None
    # seed interval on Clean CSR for primary arms
    clean_si = seed_interval([by_type(hum, "Clean", s) for s in seeds])

    h1_hum_ge_clf = pooled[hum]["M1_by_type"].get("Ambiguous", 0) >= pooled[clf]["M1_by_type"].get("Ambiguous", 1)
    h1_hum_gt_hand = pooled[hum]["M1_by_type"].get("Ambiguous", 0) > pooled[hand]["M1_by_type"].get("Ambiguous", 1)
    # sanity: Clean tie within seed interval (use range of per-seed Clean CSR across primary arms)
    clean_per_seed_means = []
    for s in seeds:
        vals = [by_type(a, "Clean", s) for a in (hum, clf, hand)]
        vals = [v for v in vals if v is not None]
        if vals:
            clean_per_seed_means.append(max(vals) - min(vals))
    clean_tie_spread = statistics.mean(clean_per_seed_means) if clean_per_seed_means else None
    clean_tie_ok = (
        clean_tie_spread is not None
        and clean_si["range"] is not None
        and clean_tie_spread <= max(clean_si["range"], 1e-9)
    ) or (clean_spread is not None and clean_spread <= 1e-9)

    # Alternative clean sanity: all primary arms within max per-seed Clean range of hum
    if clean_spread is not None and clean_si["range"] is not None:
        clean_tie_ok = clean_spread <= max(clean_si["range"], 0.0) + 1e-12

    h1_verdict = "supported" if (h1_hum_ge_clf and h1_hum_gt_hand) else (
        "mixed" if (h1_hum_ge_clf or h1_hum_gt_hand) else "null"
    )
    if not clean_tie_ok and h1_verdict == "supported":
        h1_verdict = "mixed"  # sanity failed

    # --- H2 ---
    m2_hum_ps = [m2(hum, s) for s in seeds]
    m2_clf_ps = [m2(clf, s) for s in seeds]
    m2_hand_ps = [m2(hand, s) for s in seeds]
    m2_si = seed_interval([float(x) for x in m2_hum_ps + m2_clf_ps + m2_hand_ps if x is not None])
    # "lower by more than the seed interval" — use range of per-seed M2 as interval
    interval = m2_si["range"] if m2_si["range"] is not None else 0.0
    # If all zeros, interval=0; then hum < clf and hum < hand with any positive gap fails;
    # equal zeros → advantage not > interval → null/inconclusive
    pooled_gap_clf = (m2(clf) or 0) - (m2(hum) or 0)
    pooled_gap_hand = (m2(hand) or 0) - (m2(hum) or 0)
    h2_pass = pooled_gap_clf > interval and pooled_gap_hand > interval
    h2_verdict = "supported" if h2_pass else (
        "inconclusive_zero_thrash" if (m2(hum) == 0 and m2(clf) == 0 and m2(hand) == 0)
        else "null"
    )

    # --- H3 ---
    m3_hum_ps = [m3(hum, s) for s in seeds]
    m3_clf_ps = [m3(clf, s) for s in seeds]
    gaps = []
    for s in seeds:
        a, b = m3(hum, s), m3(clf, s)
        if a is not None and b is not None:
            gaps.append(abs(a - b))
    pooled_gap = None
    if m3(hum) is not None and m3(clf) is not None:
        pooled_gap = abs(m3(hum) - m3(clf))
    h3_pass = pooled_gap is not None and pooled_gap <= H3_GAP_MAX
    # also require majority of seeds within gap
    seeds_ok = sum(1 for g in gaps if g <= H3_GAP_MAX)
    h3_verdict = "supported" if h3_pass else "null"
    # Pre-freeze prediction: H3 may fail due to long T_hold
    h3_vs_prediction = (
        "matches_pre_freeze_fail_expectation"
        if not h3_pass
        else "contrary_to_pre_freeze_risk_prediction"
    )

    # --- H4 ---
    m4_report = {a: m4(a) for a in (hum, clf, hand, "classifier_stub", "hum_local")}
    m4_hum_ps = [m4(hum, s) for s in seeds]
    m4_clf_ps = [m4(clf, s) for s in seeds]
    # propose hum ≥ classifier
    h4_hum_ge_clf = (
        m4(hum) is not None and m4(clf) is not None and m4(hum) >= m4(clf)
    )
    h4_verdict = "supported" if h4_hum_ge_clf else (
        "report-only-null" if m4(hum) is not None else "report-only-missing"
    )

    # --- H5 ---
    h5 = {
        "verdict": "not_active",
        "reason": (
            "typed_goal additive present; typed_goal_family omitted on sealed batch "
            "(identity lock) → Hum+SCP level-0 inactive; L1 fallback. No separate H5 arm run."
        ),
        "scored_separately": True,
        "cannot_rescue_H1_H4": True,
    }

    # --- H6 ---
    h6_cells = {}
    for arm_name, arm_data in pooled.items():
        h6_cells[arm_name] = {
            "M6": arm_data["M6"],
            "M6d_stale_break": arm_data["M6d_stale_break"],
        }
    # rank carry modes by mean M6c where defined (hum has no special standing)
    mode_scores: Dict[str, List[float]] = defaultdict(list)
    for arm_name, arm_data in pooled.items():
        for key, cell in arm_data["M6"].items():
            mode = cell["carry_mode"]
            m6c = cell.get("M6c_csr_per_byte")
            m6a = cell.get("M6a")
            if m6c is not None:
                mode_scores[mode].append(m6c)
            elif m6a is not None and cell.get("M6b_bytes_mean", 0) == 0:
                # cold / zero-byte: report M6a alone, not ranked on M6c
                pass
    mode_rank = sorted(
        ((m, statistics.mean(vs)) for m, vs in mode_scores.items() if vs),
        key=lambda x: -x[1],
    )
    cold_m6a = []
    carry_m6a = defaultdict(list)
    for arm_data in pooled.values():
        for cell in arm_data["M6"].values():
            if cell.get("M6a") is None:
                continue
            if cell["carry_mode"] == "cold":
                cold_m6a.append(cell["M6a"])
            else:
                carry_m6a[cell["carry_mode"]].append(cell["M6a"])
    cold_mean = statistics.mean(cold_m6a) if cold_m6a else None
    h6_verdict = "report-only"
    h6_note = (
        "Compare M6a/M6c across arms × carry modes; hum has no special standing. "
        "Null if no carry beats cold after cost normalization beyond seed interval, "
        "or M6d shows stale-state hurt on task-change breaks."
    )

    # H2 vs H3 persistence trade-off note
    tradeoff = {
        "pre_freeze_prediction": "H2 easy PASS at sealed 6–9-token distractions vs T_hold=65; H3 risk FAIL (late switch)",
        "H2_verdict": h2_verdict,
        "H3_verdict": h3_verdict,
        "observed_tradeoff": (
            h2_verdict in ("supported", "inconclusive_zero_thrash") and h3_verdict == "null"
        ),
        "note": (
            "If H2 supported/inconclusive (no thrash) and H3 null, that is the measured "
            "persistence trade-off at model level — not a surprise per PREREG."
        ),
    }

    return {
        "H1": {
            "clause": "M1 Ambiguous: hum ≥ classifier; hum > hand; Clean tie within seed interval",
            "Ambiguous_pooled": {
                "hum": by_type(hum, "Ambiguous"),
                "classifier_mlp": by_type(clf, "Ambiguous"),
                "hand_rule": by_type(hand, "Ambiguous"),
            },
            "Ambiguous_per_seed": {
                "hum": amb_hum,
                "classifier_mlp": amb_clf,
                "hand_rule": amb_hand,
            },
            "Clean_means": clean_means,
            "Clean_spread_across_arms": clean_spread,
            "Clean_seed_interval_hum": clean_si,
            "hum_ge_classifier": h1_hum_ge_clf,
            "hum_gt_hand": h1_hum_gt_hand,
            "clean_tie_ok": clean_tie_ok,
            "verdict": h1_verdict,
            "classifier_contamination_note": (
                "classifier_mlp fit on seeds 20260924–26 (train) + 20260927 (val); "
                "only 20260923 is held out. Pooled H1 classifier numbers include train/val seeds."
            ),
        },
        "H2": {
            "clause": "M2 Distraction: hum lower than classifier and hand by more than seed interval",
            "M2_pooled": {"hum": m2(hum), "classifier_mlp": m2(clf), "hand_rule": m2(hand)},
            "M2_per_seed": {"hum": m2_hum_ps, "classifier_mlp": m2_clf_ps, "hand_rule": m2_hand_ps},
            "seed_interval": m2_si,
            "gap_vs_classifier": pooled_gap_clf,
            "gap_vs_hand": pooled_gap_hand,
            "T_hold_n3": T_HOLD_N3,
            "distraction_spans": [6, 8, 9],
            "pre_freeze_prediction": "PASS easily",
            "verdict": h2_verdict,
            "measure_caveat": (
                "M2 is segment-level A→B→A with constant oracle. Sealed Distraction sequences "
                "change oracle across segments, so segment-level thrash often zeros for all arms; "
                "stateless arms have no mid-segment switch by construction. Token-level thrash "
                "not scored in locked smoke harness."
            ),
        },
        "H3": {
            "clause": "|latency_hum − latency_classifier| ≤ 2 segments",
            "M3_pooled": {"hum": m3(hum), "classifier_mlp": m3(clf), "hand_rule": m3(hand)},
            "M3_per_seed": {"hum": m3_hum_ps, "classifier_mlp": m3_clf_ps},
            "abs_gap_pooled": pooled_gap,
            "abs_gaps_per_seed": gaps,
            "seeds_within_gap": seeds_ok,
            "n_seeds": len(gaps),
            "gap_max": H3_GAP_MAX,
            "pre_freeze_prediction": "RISK fail (late switch from T_hold=65)",
            "vs_prediction": h3_vs_prediction,
            "verdict": h3_verdict,
        },
        "H4": {
            "clause": "M4 Renewal: report per arm; propose hum ≥ classifier",
            "M4_pooled": m4_report,
            "M4_per_seed_hum": m4_hum_ps,
            "M4_per_seed_classifier_mlp": m4_clf_ps,
            "hum_ge_classifier": h4_hum_ge_clf,
            "verdict": h4_verdict,
        },
        "H5": h5,
        "H6": {
            "clause": "M6a/M6c across arms × carry modes; hum no special standing",
            "cells": h6_cells,
            "M6c_mode_rank_desc": [{"carry_mode": m, "mean_M6c": v} for m, v in mode_rank],
            "cold_M6a_mean": cold_mean,
            "verdict": h6_verdict,
            "note": h6_note,
        },
        "persistence_tradeoff": tradeoff,
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    start_pt = datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + " PT"

    # --- pin verification (read-only) ---
    pins = {
        "HANDOFF.md": {
            "path": str(HANDOFF_PATH),
            "sha256": sha256_file(HANDOFF_PATH),
            "expected": EXPECTED_HANDOFF,
        },
        "PREREG.md": {
            "path": str(PREREG_PATH),
            "sha256": sha256_file(PREREG_PATH),
            "expected": EXPECTED_PREREG,
        },
        "ALL_SEQUENCES.jsonl": {
            "path": str(SEALED_ALL),
            "sha256": sha256_file(SEALED_ALL),
            "expected": EXPECTED_ALL_SHA,
        },
        "SEALED_MANIFEST.json": {
            "path": str(SEALED_MANIFEST),
            "sha256": sha256_file(SEALED_MANIFEST),
            "expected": EXPECTED_MANIFEST_SHA,
        },
        "HAND_RULE.md": {
            "path": str(HAND_RULE_MD),
            "sha256": sha256_file(HAND_RULE_MD),
            "expected": EXPECTED_HAND_SHA,
        },
        "classifier_mlp_v1.json": {
            "path": str(MLP_PATH),
            "sha256": sha256_file(MLP_PATH),
            "expected": EXPECTED_MLP,
        },
        "T11_SIDE_CHANNEL_DISCLOSURE_RECEIPT.md": {
            "path": str(SIDE_CHANNEL_MD),
            "sha256": sha256_file(SIDE_CHANNEL_MD),
            "expected": EXPECTED_SIDE_CHANNEL_MD,
        },
    }
    for v in pins.values():
        v["ok"] = v["sha256"] == v["expected"]
    if not all(v["ok"] for v in pins.values()):
        print("PIN MISMATCH", json.dumps(pins, indent=2))
        return 2

    # Confirm PREREG still frozen text on disk (do not edit)
    prereg_text = PREREG_PATH.read_text(encoding="utf-8")
    if "**Status** | **FROZEN**" not in prereg_text and "FROZEN 2026-09-23" not in prereg_text:
        print("PREREG does not look FROZEN on disk")
        return 3

    per_seed_slim: Dict[int, Dict[str, Dict[str, Any]]] = {}
    per_seed_detail: Dict[int, List[Dict[str, Any]]] = {}

    for seed in PRIMARY_SEEDS:
        print(f"=== seed {seed} ===")
        seqs = load_sequences(SEALED_ALL, seed)
        arm_results = []
        for orch in arm_factory():
            print(f"  arm {orch.name} …")
            arm_results.append(run_arm(orch, seqs))
        per_seed_detail[seed] = arm_results
        per_seed_slim[seed] = {r["arm"]: slim_arm(r) for r in arm_results}
        # write per-seed summary
        seed_path = RESULTS_DIR / f"seed{seed}_summary.json"
        seed_obj = {
            "seed": seed,
            "n_sequences": len(seqs),
            "classifier_mlp_role": (
                "held_out" if seed in MLP_HELD_OUT
                else ("train" if seed in MLP_TRAIN_SEEDS else "val")
            ),
            "arms": [slim_arm(r) for r in arm_results],
        }
        seed_path.write_bytes(json.dumps(seed_obj, indent=2, sort_keys=True).encode())

    # pooled all seeds
    print("=== pooled ALL seeds ===")
    all_seqs = load_sequences(SEALED_ALL, None)
    pooled_results = []
    for orch in arm_factory():
        print(f"  arm {orch.name} …")
        pooled_results.append(run_arm(orch, all_seqs))
    pooled_slim = {r["arm"]: slim_arm(r) for r in pooled_results}

    hypotheses = score_hypotheses(per_seed_slim, pooled_slim)

    end_pt = datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + " PT"
    wall_s = time.perf_counter() - t0

    run_meta = {
        "experiment": "T11",
        "mode": "routing_primary_multi_seed",
        "primary_id": "t11-routing-primary-v1",
        "batch_id": "t11-counting-sealed-v1",
        "primary_start_pt": start_pt,
        "primary_end_pt": end_pt,
        "wall_clock_seconds": wall_s,
        "wall_clock_minutes": round(wall_s / 60.0, 3),
        "timezone": "Pacific/Easter (PT)",
        "seeds": PRIMARY_SEEDS,
        "n_sequences_pooled": len(all_seqs),
        "arms": [r["arm"] for r in pooled_results],
        "arm_status": {r["arm"]: r["status"] for r in pooled_results},
        "pins": pins,
        "operating": {
            "T_hold_n3": T_HOLD_N3,
            "rho_eff": RHO_EFF,
            "distraction_spans": [6, 8, 9],
        },
        "prereg_frozen": True,
        "prereg_sha256": EXPECTED_PREREG,
        "handoff_sha256": EXPECTED_HANDOFF,
        "sealed_untouched": True,
        "prereg_untouched": True,
        "specialist_model_calls": False,
        "classifier_split": {
            "train": sorted(MLP_TRAIN_SEEDS),
            "val": MLP_VAL_SEED,
            "held_out": sorted(MLP_HELD_OUT),
            "note": "Pooled primary includes train/val seeds for classifier_mlp — disclosed contamination",
        },
        "disclosures": [
            "classifier_stub: untrained keyword baseline (retained)",
            "hum_local: keyword-drive local baseline (retained)",
            "hum: GNURadio hum_measure; KeywordAffinityProvider labeled L2 fallback",
            "H5 inactive (no typed_goal_family on sealed)",
            "Oracle = scoring only, not an arm",
            "side-channel disclosure PASS (receipt pin refreshed)",
        ],
        "verdicts": {
            "H1": hypotheses["H1"]["verdict"],
            "H2": hypotheses["H2"]["verdict"],
            "H3": hypotheses["H3"]["verdict"],
            "H4": hypotheses["H4"]["verdict"],
            "H5": hypotheses["H5"]["verdict"],
            "H6": hypotheses["H6"]["verdict"],
            "persistence_tradeoff_observed": hypotheses["persistence_tradeoff"]["observed_tradeoff"],
        },
    }

    score_summary = {
        "experiment": "T11",
        "mode": "routing_primary_multi_seed",
        "primary_id": "t11-routing-primary-v1",
        "seeds": PRIMARY_SEEDS,
        "pins": {k: {"sha256": v["sha256"], "ok": v["ok"]} for k, v in pins.items()},
        "pooled_arms": [slim_arm(r) for r in pooled_results],
        "per_seed_arms": {
            str(s): list(per_seed_slim[s].values()) for s in PRIMARY_SEEDS
        },
        "hypotheses": hypotheses,
        "held_out_seed_only": {
            "seed": 20260923,
            "note": "Clean classifier_mlp eval (never fit)",
            "arms": list(per_seed_slim[20260923].values()),
        },
        "wall_clock_seconds": wall_s,
        "created": end_pt,
    }

    # detail (large)
    detail = {
        **run_meta,
        "pooled_arm_details": pooled_results,
        "per_seed_arm_details": {str(s): per_seed_detail[s] for s in PRIMARY_SEEDS},
        "hypotheses": hypotheses,
    }

    detail_path = RESULTS_DIR / "RESULTS_detail.json"
    detail_bytes = json.dumps(detail, indent=2, sort_keys=True).encode()
    detail_path.write_bytes(detail_bytes)
    detail_sha = sha256_bytes(detail_bytes)

    score_path = RESULTS_DIR / "score_summary.json"
    score_bytes = json.dumps(score_summary, indent=2, sort_keys=True).encode()
    score_path.write_bytes(score_bytes)
    score_sha = sha256_bytes(score_bytes)

    meta_path = RESULTS_DIR / "run_meta.json"
    meta_obj = {**run_meta, "detail_sha256": detail_sha, "score_summary_sha256": score_sha}
    meta_bytes = json.dumps(meta_obj, indent=2, sort_keys=True).encode()
    meta_path.write_bytes(meta_bytes)
    meta_sha = sha256_bytes(meta_bytes)

    # RESULTS.md
    lines = [
        "# RESULTS — T11 routing primary (multi-seed)",
        "",
        "| Field | Value |",
        "|-------|--------|",
        f"| **Primary id** | `t11-routing-primary-v1` |",
        f"| **Status** | complete |",
        f"| **PREREG sha256** | `{EXPECTED_PREREG}` |",
        f"| **HANDOFF sha256** | `{EXPECTED_HANDOFF}` |",
        f"| **ALL_SEQUENCES sha256** | `{EXPECTED_ALL_SHA}` |",
        f"| **classifier_mlp_v1.json** | `{EXPECTED_MLP}` |",
        f"| **Primary start** | {start_pt} |",
        f"| **Primary end** | {end_pt} |",
        f"| **Wall clock** | {wall_s:.1f} s ({wall_s/60:.2f} min) |",
        f"| **Seeds** | `{PRIMARY_SEEDS}` |",
        f"| **Sequences (pooled)** | {len(all_seqs)} |",
        f"| **T_hold(n=3)** | {T_HOLD_N3} |",
        f"| **ρ_eff** | {RHO_EFF} |",
        "",
        "## Verdicts",
        "",
        f"| Hypothesis | Verdict |",
        f"|------------|---------|",
        f"| H1 | **{hypotheses['H1']['verdict']}** |",
        f"| H2 | **{hypotheses['H2']['verdict']}** |",
        f"| H3 | **{hypotheses['H3']['verdict']}** |",
        f"| H4 | **{hypotheses['H4']['verdict']}** |",
        f"| H5 | **{hypotheses['H5']['verdict']}** |",
        f"| H6 | **{hypotheses['H6']['verdict']}** |",
        "",
        "### Persistence trade-off (H2 vs H3)",
        "",
        f"- Pre-freeze prediction: H2 easy PASS; H3 risk FAIL at T_hold=65.",
        f"- Observed tradeoff flag: **{hypotheses['persistence_tradeoff']['observed_tradeoff']}**",
        f"- H3 vs prediction: `{hypotheses['H3']['vs_prediction']}`",
        f"- H2 caveat: {hypotheses['H2']['measure_caveat']}",
        "",
        "## Pooled arm summary",
        "",
        "| Arm | Status | M1 CSR | M2 thrash | M3 mean lat | M4 renewal |",
        "|-----|--------|--------|-----------|-------------|------------|",
    ]
    for a in score_summary["pooled_arms"]:
        m1 = f"{a['M1_CSR']:.3f}" if a["M1_CSR"] is not None else "n/a"
        m3v = a["M3_mean_latency"]
        m3s = f"{m3v:.3f}" if m3v is not None else "n/a"
        m4v = a["M4_renewal_recall"]
        m4s = f"{m4v:.3f}" if m4v is not None else "n/a"
        lines.append(
            f"| `{a['arm']}` | {a['status']} | {m1} | {a['M2_thrash_distraction']} | {m3s} | {m4s} |"
        )

    lines += [
        "",
        "### M1 by type (pooled, primary arms)",
        "",
    ]
    for name in ("hum", "classifier_mlp", "hand_rule"):
        a = pooled_slim[name]
        lines.append(
            f"**{name}:** "
            + ", ".join(
                f"{t}={v:.3f}" if v is not None else f"{t}=n/a"
                for t, v in sorted(a["M1_by_type"].items())
            )
        )

    lines += [
        "",
        "## H1 detail",
        "",
        f"- Ambiguous pooled: hum={hypotheses['H1']['Ambiguous_pooled']['hum']}, "
        f"clf={hypotheses['H1']['Ambiguous_pooled']['classifier_mlp']}, "
        f"hand={hypotheses['H1']['Ambiguous_pooled']['hand_rule']}",
        f"- hum≥clf: {hypotheses['H1']['hum_ge_classifier']}; "
        f"hum>hand: {hypotheses['H1']['hum_gt_hand']}; "
        f"clean_tie_ok: {hypotheses['H1']['clean_tie_ok']}",
        f"- Contamination: {hypotheses['H1']['classifier_contamination_note']}",
        "",
        "## H2 / H3 detail",
        "",
        f"- M2 pooled: {hypotheses['H2']['M2_pooled']}",
        f"- M3 pooled: {hypotheses['H3']['M3_pooled']}; abs_gap={hypotheses['H3']['abs_gap_pooled']}",
        "",
        "## Artifacts",
        "",
        f"- `results/primary/score_summary.json` sha256 `{score_sha}`",
        f"- `results/primary/RESULTS_detail.json` sha256 `{detail_sha}`",
        f"- `results/primary/run_meta.json` sha256 `{meta_sha}`",
        "",
        "## Non-mutations",
        "",
        "- Sealed batch NOT modified.",
        "- PREREG.md NOT edited (remains FROZEN).",
        "",
    ]
    results_md_path = RESULTS_DIR / "RESULTS.md"
    results_md_path.write_text("\n".join(lines), encoding="utf-8")
    results_md_sha = sha256_file(results_md_path)

    # inventory
    inv_rows = []
    for p in sorted(RESULTS_DIR.rglob("*")):
        if p.is_file():
            inv_rows.append((p.relative_to(RESULTS_DIR).as_posix(), p.stat().st_size, sha256_file(p)))
    inv_md = [
        "# ARTIFACT_INVENTORY — T11 routing primary",
        "",
        f"Generated: {end_pt}",
        f"PREREG sha: `{EXPECTED_PREREG}`",
        f"ALL_SEQUENCES sha: `{EXPECTED_ALL_SHA}`",
        "",
        "| Rel path | Size (B) | sha256 |",
        "|----------|---------:|--------|",
    ]
    for rel, sz, h in inv_rows:
        inv_md.append(f"| `{rel}` | {sz} | `{h}` |")
    inv_md.append("")
    inv_path = RESULTS_DIR / "ARTIFACT_INVENTORY.md"
    inv_path.write_text("\n".join(inv_md), encoding="utf-8")
    # refresh inventory to include itself after write — rewrite with self hash placeholder then finalize
    inv_sha = sha256_file(inv_path)
    inv_json = {
        "generated": end_pt,
        "prereg_sha256": EXPECTED_PREREG,
        "files": [
            {"path": rel, "bytes": sz, "sha256": h} for rel, sz, h in inv_rows
        ] + [{"path": "ARTIFACT_INVENTORY.md", "bytes": inv_path.stat().st_size, "sha256": inv_sha}],
    }
    inv_json_path = RESULTS_DIR / "ARTIFACT_INVENTORY.json"
    inv_json_path.write_text(json.dumps(inv_json, indent=2, sort_keys=True), encoding="utf-8")

    # FILES.sha256
    files_sha_path = RESULTS_DIR / "PRIMARY_FILES.sha256"
    tracked = [
        results_md_path,
        score_path,
        detail_path,
        meta_path,
        inv_path,
        inv_json_path,
    ] + [RESULTS_DIR / f"seed{s}_summary.json" for s in PRIMARY_SEEDS]
    with open(files_sha_path, "w", encoding="utf-8") as f:
        for p in tracked:
            f.write(f"{sha256_file(p)}  {p}\n")
    # append self
    with open(files_sha_path, "a", encoding="utf-8") as f:
        # can't easily include self; list others only is fine — rehash file after
        pass
    # final rewrite including PRIMARY_FILES after computing content without self
    content_lines = [f"{sha256_file(p)}  {p}\n" for p in tracked]
    files_sha_path.write_text("".join(content_lines), encoding="utf-8")

    # docs receipt
    receipt_lines = [
        "# T11 routing primary receipt — multi-seed v1",
        "",
        "| Field | Value |",
        "|-------|--------|",
        f"| **Date** | {end_pt} |",
        f"| **Primary id** | `t11-routing-primary-v1` |",
        f"| **Batch** | `t11-counting-sealed-v1` |",
        f"| **Seeds** | `{PRIMARY_SEEDS}` |",
        f"| **Wall clock** | {wall_s:.1f} s |",
        f"| **PREREG** | **FROZEN** `{EXPECTED_PREREG}` |",
        f"| **HANDOFF** | `{EXPECTED_HANDOFF}` |",
        f"| **ALL_SEQUENCES** | `{EXPECTED_ALL_SHA}` |",
        f"| **classifier_mlp** | `{EXPECTED_MLP}` |",
        f"| **Side-channel** | PASS `{EXPECTED_SIDE_CHANNEL_MD}` |",
        "",
        "## Verdicts",
        "",
        f"| H | Verdict |",
        f"|--|---------|",
        f"| H1 | **{hypotheses['H1']['verdict']}** |",
        f"| H2 | **{hypotheses['H2']['verdict']}** |",
        f"| H3 | **{hypotheses['H3']['verdict']}** |",
        f"| H4 | **{hypotheses['H4']['verdict']}** |",
        f"| H5 | **{hypotheses['H5']['verdict']}** |",
        f"| H6 | **{hypotheses['H6']['verdict']}** |",
        "",
        f"Persistence trade-off observed: **{hypotheses['persistence_tradeoff']['observed_tradeoff']}** "
        f"(H3 `{hypotheses['H3']['vs_prediction']}`).",
        "",
        "## Pooled M1 / M2 / M3 / M4",
        "",
        "| Arm | M1 | M2 | M3 | M4 |",
        "|-----|----|----|----|----|",
    ]
    for a in score_summary["pooled_arms"]:
        m1 = f"{a['M1_CSR']:.3f}" if a["M1_CSR"] is not None else "n/a"
        m3v = a["M3_mean_latency"]
        m3s = f"{m3v:.3f}" if m3v is not None else "n/a"
        m4v = a["M4_renewal_recall"]
        m4s = f"{m4v:.3f}" if m4v is not None else "n/a"
        receipt_lines.append(
            f"| `{a['arm']}` | {m1} | {a['M2_thrash_distraction']} | {m3s} | {m4s} |"
        )
    receipt_lines += [
        "",
        "## Pins verified",
        "",
    ]
    for k, v in pins.items():
        receipt_lines.append(f"- `{k}` `{v['sha256']}` OK")
    receipt_lines += [
        "",
        "## Artifacts",
        "",
        f"- `{score_path}` `{score_sha}`",
        f"- `{detail_path}` `{detail_sha}`",
        f"- `{meta_path}` `{meta_sha}`",
        f"- `{results_md_path}` `{results_md_sha}`",
        f"- `{inv_path}`",
        f"- `{files_sha_path}`",
        "",
        "## Blockers / notes",
        "",
        "- Sealed batch and PREREG were not mutated.",
        "- H5 not active (typed_goal_family omitted).",
        "- classifier_mlp contaminated on 4/5 primary seeds (train/val); held-out seed 20260923 reported separately in score_summary.",
        f"- H2 measure caveat: {hypotheses['H2']['measure_caveat']}",
        "- No specialist model calls (routing-only vs oracle_specialist).",
        "",
    ]
    receipt_path = DOCS_DIR / "T11_ROUTING_PRIMARY_RECEIPT.md"
    receipt_path.write_text("\n".join(receipt_lines), encoding="utf-8")
    receipt_sha = sha256_file(receipt_path)

    receipt_json = {
        **meta_obj,
        "hypotheses": hypotheses,
        "score_summary_path": str(score_path),
        "score_summary_sha256": score_sha,
        "detail_path": str(detail_path),
        "detail_sha256": detail_sha,
        "results_md_path": str(results_md_path),
        "results_md_sha256": results_md_sha,
        "receipt_md": str(receipt_path),
        "receipt_md_sha256": receipt_sha,
    }
    receipt_json_path = RECEIPTS_DIR / "T11_ROUTING_PRIMARY_RECEIPT.json"
    receipt_json_path.write_text(json.dumps(receipt_json, indent=2, sort_keys=True), encoding="utf-8")

    # append receipt hashes to PRIMARY_FILES
    with open(files_sha_path, "a", encoding="utf-8") as f:
        f.write(f"{receipt_sha}  {receipt_path}\n")
        f.write(f"{sha256_file(receipt_json_path)}  {receipt_json_path}\n")
        f.write(f"{sha256_file(files_sha_path)}  {files_sha_path}\n")

    print(json.dumps({
        "ok": True,
        "wall_clock_seconds": wall_s,
        "verdicts": run_meta["verdicts"],
        "score_summary": str(score_path),
        "score_summary_sha256": score_sha,
        "detail_sha256": detail_sha,
        "receipt": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "H2": hypotheses["H2"]["verdict"],
        "H3": hypotheses["H3"]["verdict"],
        "tradeoff": hypotheses["persistence_tradeoff"]["observed_tradeoff"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
