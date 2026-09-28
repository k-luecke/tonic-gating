#!/usr/bin/env python3
"""Emit T11.2 *dev* candidate sequences (does NOT overwrite sealed-v1).

Writes under generators/t11_2_dev/. Rule 19 should still FAIL until:
  - classifier retrained on new Ambiguous/Clean geometry
  - typed_goal_family + specialist-exec marker land (gate 4)

Also runs a hand-rule first-32 Ambiguous self-check (expect ≤ 0.45).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # t11 exp root for routing if needed
sys.path.insert(0, str(HERE))

from ambiguous import generate_ambiguous  # noqa: E402
from clean import generate_clean  # noqa: E402
from common import CARRY_MODES, PLACEMENTS, SEEDS, sha256_file, write_jsonl  # noqa: E402
from distraction import generate_distraction  # noqa: E402
from renewal import generate_renewal  # noqa: E402
from window_break import generate_window_break  # noqa: E402

OUT = HERE / "t11_2_dev"


def hand_rule_amb_csr(sequences) -> float:
    sys.path.insert(0, str(HERE.parent))
    from routing.hand_rule import HandRuleOrchestrator  # type: ignore

    hr = HandRuleOrchestrator()
    correct = total = 0
    for seq in sequences:
        for seg in seq.segments:
            d = hr.decide(seg.text)
            if d.get("abstain"):
                continue
            total += 1
            if d.get("specialist") == seg.oracle_specialist:
                correct += 1
    return (correct / total) if total else float("nan")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    by_type = {
        "Clean": [generate_clean(s) for s in SEEDS],
        "Ambiguous": [generate_ambiguous(s) for s in SEEDS],
        "Distraction": [generate_distraction(s) for s in SEEDS],
        "Renewal": [generate_renewal(s) for s in SEEDS],
    }
    wb = []
    for s in SEEDS:
        for carry in CARRY_MODES:
            for placement in PLACEMENTS:
                wb.append(generate_window_break(s, carry_mode=carry, placement=placement))
    by_type["Window-break"] = wb

    all_seqs = []
    files = {}
    for name, seqs in by_type.items():
        fname = {
            "Clean": "clean.jsonl",
            "Ambiguous": "ambiguous.jsonl",
            "Distraction": "distraction.jsonl",
            "Renewal": "renewal.jsonl",
            "Window-break": "window_break.jsonl",
        }[name]
        path = OUT / fname
        write_jsonl(path, seqs)
        files[fname] = sha256_file(path)
        all_seqs.extend(seqs)

    all_path = OUT / "ALL_SEQUENCES.jsonl"
    write_jsonl(all_path, all_seqs)
    files["ALL_SEQUENCES.jsonl"] = sha256_file(all_path)

    amb_csr = hand_rule_amb_csr(by_type["Ambiguous"])
    # Distraction other-family flag check
    other_family_spans = 0
    total_spans = 0
    for seq in by_type["Distraction"]:
        for seg in seq.segments:
            sp = seg.distraction_span or {}
            total_spans += 1
            if sp.get("other_family_cues"):
                other_family_spans += 1

    manifest = {
        "batch_id": "t11-2-dev-v0",
        "status": "dev_candidate_not_sealed",
        "note": "Does not overwrite sealed-v1. Rule19 gate1 Ambiguous hand_rule self-check below.",
        "n_sequences": len(all_seqs),
        "files_sha256": files,
        "hand_rule_ambiguous_first32_csr": amb_csr,
        "hand_rule_ambiguous_gate1_target_max": 0.45,
        "hand_rule_ambiguous_self_check_pass": amb_csr <= 0.45,
        "distraction_other_family_spans": other_family_spans,
        "distraction_total_spans": total_spans,
    }
    man_path = OUT / "DEV_MANIFEST.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"wrote {OUT}")
    if amb_csr > 0.45:
        print("WARN: Ambiguous hand_rule first-32 CSR still > 0.45 — tighten preambles")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
