# model/ — recurrent-depth toy (T1)

**Status:** smoke toy for fixed-depth eval (no train).

- Prelude → recurrent residual (`ρ(A)<1`) → coda mixture
- Width ≤128 (smoke uses 64); numpy CPU
- Belief over frozen T7 rule-bank with depth-budgeted updates + OT dilution
- Sweep: `scripts/run_t1_smoke.py` → `results/t1/`

Do **not** treat as trained coordinator. No PREREG freeze here.
