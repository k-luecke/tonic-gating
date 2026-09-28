#!/usr/bin/env python3
"""Authorized T12 PREREG freeze entry — Rule 19 is mandatory (T5).

Naive ``sha256sum PREREG.md > PREREG.sha256`` is FORBIDDEN.

Calls ``scripts/rule19_freeze_gates.py`` first. Emits PREREG.md + PREREG.sha256
only when all gates PASS and ``--authorize`` is set.

Hum / AffinityProvider / C₃ are never retuned. Parent T11.x hashes untouched.
T12b is a separate track — this script does not freeze t12b.

Usage:
  python3 scripts/freeze_prereg.py                 # status / gates only
  python3 scripts/freeze_prereg.py --authorize     # emit iff gates PASS
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

T12 = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve().parent / "rule19_freeze_gates.py"
PREREG_SHA = T12 / "PREREG.sha256"


def main() -> int:
    ap = argparse.ArgumentParser(description="Authorized T12 PREREG freeze (Rule 19 gated)")
    ap.add_argument("--authorize", action="store_true", help="Emit PREREG.sha256 iff gates PASS")
    args = ap.parse_args()

    print("=== freeze_prereg.py ticket=t12 (Rule 19 required) ===")
    print("Naive sha256sum → PREREG.sha256 is blocked; this entry calls Rule 19 first.")

    if not SCRIPT.is_file():
        print(f"ERROR: missing {SCRIPT}", file=sys.stderr)
        return 2

    cmd = [sys.executable, str(SCRIPT)]
    if args.authorize:
        cmd.append("--emit-hash")
    print("running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(T12))
    if proc.returncode != 0:
        print("FREEZE BLOCKED — Rule 19 (t12) failed; hash not updated.")
        return proc.returncode

    if not args.authorize:
        print("Status-only: gates PASS but hash not written (re-run with --authorize).")
        return 0

    if not PREREG_SHA.is_file():
        print("ERROR: gates PASS but PREREG.sha256 missing after emit", file=sys.stderr)
        return 3
    digest = PREREG_SHA.read_text(encoding="utf-8").split()[0].strip()
    print(f"FREEZE OK — PREREG.sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
