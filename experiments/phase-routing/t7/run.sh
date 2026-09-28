#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SEED="${SEED:-20260921}"
STEPS="${STEPS:-1000}"
MODE="${1:-all}"
# Prefer parent venv if present, else local, else system python.
if [[ -x ../.venv/bin/python ]]; then
  PYTHON="${PYTHON:-../.venv/bin/python}"
elif [[ -x .venv/bin/python ]]; then
  PYTHON="${PYTHON:-.venv/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"

data() { "$PYTHON" -m t7_routing.data --seed "$SEED"; sha256sum --check data/MANIFEST.sha256; }
test_all() { "$PYTHON" -m pytest -q; }
train_arm() { "$PYTHON" train.py --router "$1" --seed "$SEED" --steps "$STEPS" --output-dir results/exploratory/smoke; }
case "$MODE" in
  data) data ;;
  test) test_all ;;
  lr|lr_goal|hr1|pg_inferred|pg) train_arm "$MODE" ;;
  harness) "$PYTHON" harness.py --seed "$SEED" --arms lr lr_goal hr1 pg_inferred ;;
  smoke)
    data; test_all
    for arm in lr lr_goal hr1 pg_inferred; do train_arm "$arm"; done
    "$PYTHON" harness.py --seed "$SEED" --arms lr lr_goal hr1 pg_inferred
    ;;
  *) echo "Usage: $0 [data|test|lr|lr_goal|hr1|pg_inferred|pg|harness|smoke]" >&2; exit 2 ;;
esac
