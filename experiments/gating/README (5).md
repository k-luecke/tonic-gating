# Oscillatory Gating vs Learned Router

Drive-backed experiment home for Kyle's preregistered comparison of **oscillatory gating** (stencil + phase pilot) against a **learned top-k router**. This directory is the working copy on the agent box; it is **not a git repo yet** — `PREREG.md` freezes the analysis plan (T6); external hash/commit anchor still pending.

**Handoff:** [Handoff: Oscillatory Gating vs Learned Router](https://docs.google.com/document/d/1O6JYHOctJB_QomccStfsmV4COwNznZhY4YJiWZU35dU) (also condensed in `HANDOFF_EXCERPT.md`).

**Preregistration (T6 freeze):** [`PREREG.md`](PREREG.md) — hypotheses, system/router spec, disturbance grid, measures, and H1 criteria frozen **before** comparison runs. Do not retune from results.

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

## Results (primary)

See **[RESULTS.md](RESULTS.md)** for the registered verdict (**mixed**), what counts vs what does not, and the secondary router+pilot arm note. Artifacts: `results/comparison_primary/` · PREREG hash in `results/PREREG.sha256`.

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


## How to run T4 (learned router)

```bash
cd /workspace/oscillatory-gating
source .venv/bin/activate
python train_router.py            # trains 2k symbols, writes results/
# or:  .venv/bin/python train_router.py
pytest -q tests/test_router.py    # loads frozen weights; clean ACC ≥ 0.95
```

Artifacts:
- `results/router_weights.npz` — frozen MLP (+ sha256 in stats)
- `results/router_readouts.npz` — frozen per-patch bit readouts
- `results/T4_router_stats.txt` — param count, MACs/decision, ACC, hashes

Router path sheet: `Sheet(..., gain_mode="fixed")` (g=1, no rhythm). Default
`gain_mode="oscillatory"` is unchanged for T2/T3.

### T5 call pattern

```python
from router import load_frozen_router, load_frozen_readouts, router_features
from sheet import Sheet

router = load_frozen_router()          # results/router_weights.npz
readouts = load_frozen_readouts()
sheet = Sheet(gain_mode="fixed", sigma_f=..., seed=...)

# per symbol window (length SAMPLES_PER_SYMBOL), trial goal d_star:
feat = router_features(window, d_star)           # 24-bin hist || one-hot(d*)
k = int(router.select_topk(feat, k=1)[0])        # or top_k=m
ctrl = router.make_control(k)                    # inhibition only, no pilot
for u_t in window:
    sheet.step(u_t, ctrl)
bit_hat = readouts.predict_bit(window, k)
# equivalently: ctrl = router.control_for_target(d_star, window=window)
```



## How to run T5 (gate vs router comparison)

```bash
cd /workspace/oscillatory-gating
source .venv/bin/activate
python run_comparison.py            # smoke-fast default (minutes)
python run_comparison.py --primary  # registered primary → results/comparison_primary/
python run_comparison.py --full     # full 4×4×4 if wall-clock allows
python run_comparison.py --switch   # add switching-latency protocol to smoke
# or:  .venv/bin/python run_comparison.py
```

Default smoke: 40 symbols × 3 seeds on a **subset** of the disturbance grid
(clean, each-axis high, mid corner, wobble). Explicitly non-preregistered —
for wiring / green runs only. Smoke outputs live under
`results/comparison_smoke_exploratory/` (do not treat as primary).

`--primary` writes `results/comparison_primary/` after `results/PREREG.sha256`
exists (prereg trial counts on a denser-than-smoke grid). `--full` is the
complete 4×4×4 PREREG grid when wall-clock allows.

**Demodulation:** router bit head = frozen 24-bin phase histogram (unnormalized
amplitudes) → per-patch linear head. Gate bit head = gated drive resampled onto
the open-pilot-phase grid (not absolute-time FIR).

Loads frozen router from `results/router_weights.npz` (does **not** retrain).
Trains gate pilot-phase readouts once into `results/gate_readouts.npz` if missing
or legacy.

Primary outputs under `results/comparison_primary/`:
- one JSON per condition + `all_conditions.json` + `summary.csv`
- figures: `fig_ACC.png`, `fig_S_L.png`, `fig_T_sw.png`, `fig_cost.png`
- `H1_verdict.txt` (H1 / H0 / mixed per PREREG §6)
- `run_meta.json` with `mode`, `prereg_sha256`, `start_time_iso`

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
| `router.py` | Learned top-k MLP router + frozen patch bit-readouts (T4) |
| `train_router.py` | Train/freeze router; write `results/T4_router_stats.txt` |
| `run_comparison.py` | T5 harness: gate vs router under disturbance grid |

Selectivity: `S = A_{k*} / Σ_k A_k` with `A_k = mean(s_i)` over patch k.

## Environment

Python 3, numpy, scipy, matplotlib, pytest. Brian2 is optional (commented in `requirements.txt`) for a later spiking variant.

## Notes

- Deterministic seeds; re-runs should match qualitatively (and bit-for-bit given the same env).
- Design constraints from the lock-in findings: positive-only volleys, PLL locks to pilot not content, fixed refs need resync under drift.
- Do not break T0: `python lockin.py` must keep working.
