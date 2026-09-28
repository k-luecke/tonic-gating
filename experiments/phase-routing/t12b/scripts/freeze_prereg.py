#!/usr/bin/env python3
"""Authorized T12b PREREG freeze entry — Rule 19 is mandatory.

Default: run Rule 19 gates only; write PASS/FAIL receipt.
With --authorize: emit PREREG.md + PREREG.sha256 iff all gates PASS.

Naive ``sha256sum PREREG.md > PREREG.sha256`` is FORBIDDEN.
Parent T11.2 / T11.3 PREREG hashes untouched. Affinity / C₃ / hum never retuned.
Draft file content sha must remain 408eaaac… (promote writes separate PREREG.md).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

T12B = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve().parent / "rule19_freeze_gates.py"
PREREG_SHA = T12B / "PREREG.sha256"
PREREG_MD = T12B / "PREREG.md"
DRAFT = T12B / "docs" / "T12B_PREREG_DRAFT.md"
EXPECTED_DRAFT_SHA = "408eaaaca59d0c210e1b1ef50d2494213662da1059b3542c38ce866651801b21"


def main() -> int:
    ap = argparse.ArgumentParser(description="Authorized T12b PREREG freeze (Rule 19 gated)")
    ap.add_argument(
        "--authorize",
        action="store_true",
        help="Emit PREREG.sha256 after gates PASS",
    )
    args = ap.parse_args()

    print("=== freeze_prereg.py ticket=t12b (Rule 19 required) ===")
    print("Naive sha256sum → PREREG.sha256 is blocked; this entry calls Rule 19 first.")

    if not SCRIPT.is_file():
        print(f"ERROR: missing {SCRIPT}", file=sys.stderr)
        return 2

    cmd = [sys.executable, str(SCRIPT)]
    if args.authorize:
        cmd.append("--emit-hash")
    print("running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(T12B))
    if proc.returncode != 0:
        print("FREEZE BLOCKED — Rule 19 (t12b) failed; hash not updated.")
        return proc.returncode

    if not args.authorize:
        print("Status-only: Rule 19 evaluated; PREREG.sha256 NOT written (re-run with --authorize).")
        if DRAFT.is_file():
            print(f"draft: {DRAFT}")
        return 0

    if not PREREG_SHA.is_file() or not PREREG_MD.is_file():
        print("ERROR: gates PASS but PREREG.md / PREREG.sha256 missing after emit", file=sys.stderr)
        return 3

    import hashlib
    h = hashlib.sha256()
    with open(DRAFT, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    draft_sha = h.hexdigest()
    if draft_sha != EXPECTED_DRAFT_SHA:
        print(f"ERROR: draft content sha drifted: {draft_sha}", file=sys.stderr)
        return 4

    print(f"FREEZE OK — PREREG.sha256 = {PREREG_SHA.read_text().split()[0]}")
    print(f"draft_content_sha confirmed = {draft_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
