"""Thin re-export of frozen hum_measure (do not retune)."""
from __future__ import annotations

import sys
from pathlib import Path

_HUM = Path(__file__).resolve().parent / "hum_measure"
if str(_HUM) not in sys.path:
    sys.path.insert(0, str(_HUM.parent))

# Prefer package import via symlink name
from hum_measure import *  # noqa: F401,F403
