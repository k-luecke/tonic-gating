#!/usr/bin/env python3
"""T0 stub: import hum_measure and print OK.

GNURadio owns confirming hum dynamics. This only checks the import path
from the T12 tree resolves.
"""
from __future__ import annotations

import sys
from pathlib import Path

T12_ROOT = Path(__file__).resolve().parents[1]
IMPORTS = T12_ROOT / "imports"
HUM = IMPORTS / "hum_measure"

def main() -> int:
    if not HUM.exists():
        print(f"FAIL: hum_measure missing at {HUM}")
        return 1
    # hum_measure is a package dir with __init__.py
    sys.path.insert(0, str(IMPORTS))
    import hum_measure  # noqa: F401

    print("import OK: hum_measure")
    print(f"  path: {HUM.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
