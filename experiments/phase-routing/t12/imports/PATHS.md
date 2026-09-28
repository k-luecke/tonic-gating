# T12 imports — documented paths

| Import | Path | Notes |
|--------|------|-------|
| **hum_measure** | `imports/hum_measure` → `/workspace/scp/t11/imports/hum_measure` | Symlink. Frozen dynamics — **do not retune** C₃ / hum / Affinity `__all__`. |
| **t8_hum** | `imports/t8_hum` → `/workspace/scp/experiments/phase-routing/t11/hum/` | Frozen HumState + `acceptance_t8.py` (T8 unit / dynamics persistence). |
| **t8_results_primary** | `imports/t8_results_primary` → `/workspace/scp/experiments/phase-routing/t8/results/primary/` | T8 rerun persistence artifacts (`RESULTS.md`, `SMOKE_HUM.md`, `score_summary.json`). |
| **t8_PREREG** | `imports/t8_PREREG.md` → `/workspace/scp/t11/imports/t8_PREREG.md` | Persistence / shift regression lock reference. |
| **t7_phase_routing_recovered** | `imports/t7_phase_routing_recovered` → `/workspace/phase-routing/t8/_recovered/phase_routing` | Recovered T7 package (`data.py` generator, model, routers). **Outside** `/workspace/scp` work tree; linked read-only for T1 difficulty-split adaptation. |
| **t7_routing_freeze_recovered** | `imports/t7_routing_freeze_recovered` → `/workspace/phase-routing/t8/_recovered/t7_freeze/t7_routing` | T7 freeze-tree routing package (order-3 / harder task). |
| T8 FilterStack (ref) | `/workspace/phase-routing/t8/t8_routing/filters.py` | Roofing high-pass source (HP_PERIOD=64). T12 smoke uses numpy reimpl of roofing stage only (`scripts/highpass_drive.py`). |

## T7 generator search (2026-09-23)

- **Inside `/workspace/scp`:** no `*t7*` generator source tree.
- **On box (outside scp):** FOUND at `/workspace/phase-routing/t8/_recovered/` (see above).
- T12 `generators/` remains a **placeholder** until T1 adapts T7 into easy/hard difficulty split.
- If recovered link breaks: **MISSING — need T7 generator path from Kyle/Drive.**

## Hum / T0 reproduce (GNURadio)

One command from `t12/`:

```bash
cd /workspace/scp/experiments/phase-routing/t12
python3 scripts/run_t0_hum.py
```

Stepwise:

```bash
python3 tests/test_hum_import.py                 # hum_measure import
python3 scripts/run_t8_acceptance.py             # T8 hum unit tests (dynamics)
python3 scripts/check_t8_rerun_persistence.py    # T8 rerun primary numbers + shift lock
python3 scripts/smoke_hum_n2.py                  # n=2 easy/hard + HP + amp readout
```

Receipt: `results/T0_HUM_REPRODUCE.md`. Artifacts: `results/t0/`.

**Locks confirmed by T0:** hum **n=2** (easy/hard); high-pass **every** drive; amplitude readout **never** phase-gated; Affinity/C₃/T11 n=3 package **untouched**.
