#!/usr/bin/env python3
"""Validate T11.2 generator geometry against Rule 19 gates 1–2 (hand-rule side).

Does not emit PREREG hash. Safe to run on sealed-v1 (expect FAIL) or t11_2_dev.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
T11_EXP = HERE.parent
sys.path.insert(0, str(T11_EXP))

from routing.hand_rule import HandRuleOrchestrator  # noqa: E402
from routing.lexicon import first_n_tokens, keyword_counts  # noqa: E402


def load(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()
    d = args.dir
    hr = HandRuleOrchestrator()

    amb = load(d / "ambiguous.jsonl")
    cln = load(d / "clean.jsonl")
    dist = load(d / "distraction.jsonl")

    def csr(seqs):
        c = t = 0
        for seq in seqs:
            for seg in seq["segments"]:
                dec = hr.decide(seg["text"])
                if dec.get("abstain"):
                    continue
                t += 1
                if dec["specialist"] == seg["oracle_specialist"]:
                    c += 1
        return (c / t if t else None), c, t

    amb_csr, _, _ = csr(amb)
    cln_csr, _, _ = csr(cln)
    print(f"hand_rule Ambiguous first-32 CSR={amb_csr} (want ≤0.45)")
    print(f"hand_rule Clean first-32 CSR={cln_csr} (want ≥0.90)")

    # Show first-32 keyword leaks on Ambiguous
    leaks = 0
    for seq in amb:
        for seg in seq["segments"]:
            toks = first_n_tokens(seg["text"], 32)
            counts = keyword_counts(toks)
            if any(v > 0 for v in counts.values()):
                leaks += 1
    print(f"Ambiguous segments with any first-32 family keyword: {leaks}")

    of = 0
    spans = 0
    for seq in dist:
        for seg in seq["segments"]:
            sp = seg.get("distraction_span") or {}
            spans += 1
            if sp.get("other_family_cues") or sp.get("cue_family"):
                of += 1
    print(f"Distraction spans with other-family meta: {of}/{spans}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
