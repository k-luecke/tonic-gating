#!/usr/bin/env python3
"""Emit one smoke JSONL example per sequence type (scaffold only; not full T2)."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from clean import generate_clean
from ambiguous import generate_ambiguous
from distraction import generate_distraction
from renewal import generate_renewal
from window_break import generate_window_break
from common import CARRY_MODES, SEEDS, write_jsonl

SMOKE = HERE / "smoke"
SMOKE_SEED = SEEDS[0]  # 20260923


def main() -> None:
    write_jsonl(SMOKE / "clean_smoke.jsonl", [generate_clean(SMOKE_SEED)])
    write_jsonl(SMOKE / "ambiguous_smoke.jsonl", [generate_ambiguous(SMOKE_SEED)])
    write_jsonl(SMOKE / "distraction_smoke.jsonl", [generate_distraction(SMOKE_SEED)])
    write_jsonl(SMOKE / "renewal_smoke.jsonl", [generate_renewal(SMOKE_SEED)])

    wb_seqs = [
        generate_window_break(SMOKE_SEED, carry_mode=m, placement="at_task_boundary")
        for m in CARRY_MODES
    ]
    write_jsonl(SMOKE / "window_break_smoke.jsonl", wb_seqs)
    write_jsonl(
        SMOKE / "window_break_inside_task_smoke.jsonl",
        [generate_window_break(SMOKE_SEED, carry_mode="cold", placement="inside_task")],
    )
    print(f"Wrote smoke JSONL under {SMOKE}")
    for p in sorted(SMOKE.glob("*.jsonl")):
        print(f"  {p.name}: {p.stat().st_size} bytes")


if __name__ == "__main__":
    main()
