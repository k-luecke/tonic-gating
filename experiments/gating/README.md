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

## Environment

Python 3, numpy, scipy, matplotlib. Brian2 is optional (commented in `requirements.txt`) for a later spiking variant.

## Notes

- Deterministic seeds; re-runs should match qualitatively (and bit-for-bit given the same env).
- Design constraints from the lock-in findings: positive-only volleys, PLL locks to pilot not content, fixed refs need resync under drift.
