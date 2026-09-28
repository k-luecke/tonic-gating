#!/usr/bin/env python3
"""Thin wrapper: run frozen T8 HumState acceptance from the T12 tree.

Does not retune dynamics. Executes:
  /workspace/scp/experiments/phase-routing/t11/hum/acceptance_t8.py
and copies ACCEPTANCE_RESULT.* into t12/results/t0/.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

T12_ROOT = Path(__file__).resolve().parents[1]
HUM_DIR = (T12_ROOT / "imports" / "t8_hum").resolve()
OUT_DIR = T12_ROOT / "results" / "t0"
ACC = HUM_DIR / "acceptance_t8.py"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not ACC.is_file():
        print(f"FAIL: frozen acceptance missing at {ACC}")
        return 1
    print(f"Running frozen acceptance: {ACC}")
    print(f"  hum_state sha256: {sha256_file(HUM_DIR / 'hum_state.py')}")
    print(f"  acceptance sha256: {sha256_file(ACC)}")
    proc = subprocess.run(
        [sys.executable, str(ACC)],
        cwd=str(HUM_DIR),
        check=False,
    )
    src_json = HUM_DIR / "results" / "ACCEPTANCE_RESULT.json"
    src_md = HUM_DIR / "results" / "ACCEPTANCE_RESULT.md"
    if src_json.is_file():
        shutil.copy2(src_json, OUT_DIR / "ACCEPTANCE_RESULT.json")
    if src_md.is_file():
        shutil.copy2(src_md, OUT_DIR / "ACCEPTANCE_RESULT.md")
        print(src_md.read_text())
    print(f"Copied acceptance artifacts → {OUT_DIR}")
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
