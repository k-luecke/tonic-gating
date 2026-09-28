#!/usr/bin/env python3
"""T0 entry: delegates to scripts/run_t0_hum.py (one documented command)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

T12_ROOT = Path(__file__).resolve().parents[1]
RUNNER = T12_ROOT / "scripts" / "run_t0_hum.py"


def main() -> int:
    if not RUNNER.is_file():
        print(f"FAIL: missing {RUNNER}")
        return 1
    return subprocess.call([sys.executable, str(RUNNER)], cwd=str(T12_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
