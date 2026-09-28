# Gate readout vs pilot — diagnostic (pre-PREREG hash)

Date: 2026-09-21 (PT). Gate PLL parameters (K_p, τ, τ_m, K_c) **not** retuned.

## 1. T3 lock test

### T3-native σ_c (shared fractional frequency offset δ ~ N(0, σ_c))
This is what `tests/test_stencil.py` / T3 acceptance uses.

| σ_c | max |φ−ψ| (deg) | mean (deg) | pass ±20°? |
|-----|------------------|------------|------------|
| 0.00 | 7.49 | 5.75 | True |
| 0.05 | 7.32 | 5.46 | True |

**Result:** T3 acceptance still passes at σ_c=0.05 (fractional freq). Loop is fine under the T3 disturbance model.

### Harness σ_c (PREREG: i.i.d. N(0, σ_c) rad/sample random walk on shared phase)
| σ_c | mean max-err (deg) | p95 (deg) | frac ≤20° |
|-----|--------------------|-----------|-----------|
| 0.00 | 8.7 | 10.3 | 1.00 |
| 0.05 | 63.7 | 94.2 | 0.00 |

**Result:** Under the harness RW at σ_c=0.05, open-patch phase error often exceeds ±20°. That is a harder disturbance than T3-native δ. Still **no PLL retune** (gate free params frozen from T0–T3).

## 2. What the gate readout was referenced to

**Before (bug):** `GateReadouts` applied a ridge FIR to the **absolute-time** series `mean(g·u)` over the 500-sample symbol window. Features were indexed by sample index, not by pilot phase — an absolute carrier/time reference.

**After (fix):** Features = the same gated drive **resampled onto a uniform open-pilot-phase grid** (`pilot_phase_resample`). Bit head is linear on that pilot-referenced waveform. Mode stored as `pilot_phase` in `results/gate_readouts.npz`. Legacy `time_fir` kept only as exploratory.

## 3. Single-system bit ACC (gate only; clean train → eval)

| readout mode | ACC clean (σ_c=0) | ACC σ_c=0.05 RW |
|--------------|-------------------|-----------------|
| time_fir (legacy absolute) | 0.975 | 0.535 |
| pilot_phase (fixed) | 0.990 | 0.660 |

Absolute-time FIR collapses toward chance under shared RW even though T3-native lock passes — consistent with a readout reference bug on top of a harder harness drift. Pilot-phase resampling is the correctness fix (analogous to router phase-binned demod); K_p/τ untouched.

## 4. Router demod (parallel fairness fix)

Router primary bit head = 24-bin phase histogram → per-patch linear head (`phase_hist`), frozen in `results/router_readouts.npz`. Demod ACC (patch k → bit[k] from mux window alone) logged in T4 stats.

## 5. σ_c unit unification (post-diagnostic)

PREREG now defines σ_c in **rad/cycle**. Harness RW step per sample = σ_c / samples_per_cycle (100 at 4 kHz/40 Hz).
The ACC numbers above used the **legacy rad/sample** grid (smoke-era). Under unified units, σ_c=0.05 rad/cycle ≡ 0.0005 rad/sample (100× milder than the legacy smoke cell of the same numeric label).
