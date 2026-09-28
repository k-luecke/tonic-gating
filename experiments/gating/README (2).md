# Oscillatory Gating vs Learned Router

Drive-backed experiment home for Kyle's preregistered comparison of **oscillatory gating** (stencil + phase pilot) against a **learned top-k router**. This directory is the working copy on the agent box; it is **not a git repo yet** — provenance and freeze happen later (T6).

**Handoff:** [Handoff: Oscillatory Gating vs Learned Router](https://docs.google.com/document/d/1O6JYHOctJB_QomccStfsmV4COwNznZhY4YJiWZU35dU) (also condensed in `HANDOFF_EXCERPT.md`).

## Mission (one line)

Compare a sheet of fast oscillatory units gated by a slow top-down stencil + pilot against the same patches selected by a learned router; report selectivity, leakage, switching latency, and cost under mismatch, jitter, and shared drift.

## Tickets

| Ticket | Summary |
|--------|---------|
| **T0** | Repo scaffold + reproduce `lockin.py` figure (this directory) |
| **T1** | Receiver/sender timing-jitter sweep → slots/cycle vs σ_j |
| **T2** | `sheet.py` — 16×16 phase-oscillator substrate |
| **T3** | `stencil.py` — inhibition map + per-patch pilot PLL |
| **T4** | `router.py` — learned top-k baseline |
| **T5** | `run_comparison.py` — full harness + measures |
| **T6** | `PREREG.md` freeze + external hash anchor |
| **T7** | Run and `RESULTS.md` (no tuning) |
| **T8** | Hardware notes (SpiNNaker / BrainScaleS-2) stretch |

T1–T4 can run in parallel after T0.

## How to run T0 (lock-in reproduction)

```bash
cd /workspace/oscillatory-gating
python3 -m venv .venv          # once
source .venv/bin/activate
pip install -r requirements.txt
python lockin.py               # writes results/lockin_results.png
# equivalent without activating:  .venv/bin/python lockin.py
```

Writes `results/lockin_results.png` (and a copy as `./lockin_results.png`). Prints a short console summary of the four preregistered findings.

## How to run T2/T3 tests

```bash
cd /workspace/oscillatory-gating
source .venv/bin/activate
pip install -r requirements.txt pytest   # pytest once if needed
pytest -q tests/test_sheet.py tests/test_stencil.py
# or:  .venv/bin/python -m pytest -q tests/
```

## Patch layout (T2)

**Choice recorded:** K=8 patches of 32 units as **4×8 blocks** (4 rows × 8 cols).

On the 16×16 grid: 4 patch-rows × 2 patch-cols. Patch `k = p_r * 2 + p_c` owns
rows `[4*p_r .. 4*p_r+3]`, cols `[8*p_c .. 8*p_c+7]`.

## Modules (T2/T3)

| File | Role |
|------|------|
| `sheet.py` | 16×16 Kuramoto sheet, leaky integrators, `step(input_vector, control)` |
| `stencil.py` | Inhibition map + per-patch pilots; `control_for_target(d_star)` |
| `content.py` | Eight phase-multiplexed pulsed streams `θ_d = 2π d / 8` |

Selectivity: `S = A_{k*} / Σ_k A_k` with `A_k = mean(s_i)` over patch k.

## Environment

Python 3, numpy, scipy, matplotlib, pytest. Brian2 is optional (commented in `requirements.txt`) for a later spiking variant.

## Notes

- Deterministic seeds; re-runs should match qualitatively (and bit-for-bit given the same env).
- Design constraints from the lock-in findings: positive-only volleys, PLL locks to pilot not content, fixed refs need resync under drift.
- Do not break T0: `python lockin.py` must keep working.
