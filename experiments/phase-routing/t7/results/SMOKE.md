# T7 one-seed smoke (non-counting)

Seed `20260921`, steps **1000**, batch 32, device CPU. Artifacts under `results/exploratory/smoke/`.

## Clean + primary robustness cells

| Arm | clean L | dom≥0.25 (from train metrics) | T_sw | ΔL noise 0.3 | ΔL shift 0.3 | C_route |
|-----|--------:|------------------------------:|-----:|-------------:|-------------:|---------|
| lr | 1.255369 | 7/8 | 1.8145 | 0.004311 | 0.009991 | 1024 MACs |
| lr_goal | 1.260293 | 8/8 | 0.3438 | 0.001200 | 0.011955 | 2048 MACs |
| hr1 | 1.326834 | 0/8 | 1.0255 | 0.000000 | 0.003790 | 0 MACs |
| pg_inferred | 1.271557 | 8/8 | 4.9920 | 0.002602 | 0.006558 | 256 MACs + 8 cos |

## Band check

- LR clean L = **1.2554** (preferred 0.5–1; acceptance 0.3–1.5) → **in acceptance**.
- LR specialization = **7/8** (need ≥4/8) → **pass**.
- |L_PG−L_LR| = **0.0162** (H1 gate 0.1) → pass on this seed.

## Knobs changed from initial defaults (single-arm only)

- `p_flip`: 0.15 → **0.0** (0.15 unlearnable; see PREREG §7).
- Coeffs: large multipliers → **A=1, B=(1,1,1,1,2,2,2,2), C=(1,1,2,2,1,1,2,2), D=offsets**.
- Steps: 1200 → **1000**.

## Qualitative notes (non-counting)

- PG-inferred **noise ΔL ≠ 0** — not by-construction (φ_ref from hidden).
- T_sw(PG-inferred) **>** T_sw(LR) on this seed; LR-goal fastest (goal channel).
- HR1 noise ΔL = 0 (hash ignores hidden) — expected.

## Status

Smoke looks sane (no NaNs; LR in band with specialization; PG-inferred learns).
**STOP before multi-seed primary.** See `READY_TO_FREEZE.md`.
