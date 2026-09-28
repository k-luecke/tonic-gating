#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SEED="${SEED:-20260921}"
STEPS="${STEPS:-600}"
MODE="${1:-all}"
PYTHON="${PYTHON:-.venv/bin/python}"
[[ -x "$PYTHON" ]] || { echo "Missing $PYTHON" >&2; exit 1; }
data() { "$PYTHON" -m phase_routing.data --seed "$SEED"; sha256sum --check data/MANIFEST.sha256; }
test_all() { "$PYTHON" -m pytest -q; }
train_arm() { "$PYTHON" train.py --router "$1" --seed "$SEED" --steps "$STEPS"; }
case "$MODE" in
  data) data ;;
  test) test_all ;;
  lr|hr1|hr2|pg) train_arm "$MODE" ;;
  harness) "$PYTHON" harness.py --seed "$SEED" ;;
  all) data; test_all; train_arm lr; train_arm hr1; train_arm hr2; train_arm pg; "$PYTHON" harness.py --seed "$SEED" ;;
  *) echo "Usage: $0 [data|test|lr|hr1|hr2|pg|harness|all]" >&2; exit 2 ;;
esac
