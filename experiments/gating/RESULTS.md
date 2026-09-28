# RESULTS — Oscillatory Gating vs Learned Router

| Field | Value |
|-------|--------|
| **Date** | 2026-09-21 (PT) |
| **Owner** | Kyle (ZetaZero) |
| **Primary verdict** | **mixed** |
| **PREREG** | `PREREG.md` · sha256 `c81851b03cf5920c5f34a5fe6ea81411b6cef73012d2fc99561ad155df1fef10` |
| **Hash created** | 2026-09-21T19:16:41-05:00 |
| **Primary start** | 2026-09-21T19:16:46-05:00 (after hash) |
| **Artifacts** | `results/comparison_primary/` |

This document reports the **primary registered run** against the frozen criteria in `PREREG.md`. It does not move thresholds or retune gate PLL parameters.

---

## 1. Verdict (primary)

**mixed.**

Against PREREG §6:

1. **Clean ACC** — `ACC_gate = 0.9915` ≥ `ACC_router − 0.02` (`0.9935 − 0.02`) → **pass**.
2. **Clean C_act** — `C_act_gate = 119.71` ≤ `0.5 × C_act_router = 128` → **pass**.
3. **Shared drift** at `σ_c = 0.05` **rad/cycle** — drop_gate = `0.0145`, drop_router = `−0.0035`, CI width ≈ `0.0055` → relative gracefulness **fails** (`drift_ok = False`).

H1 required all three. The cost and clean-ACC clauses hold; the mechanism clause (“degrades more gracefully under shared drift”) does not, on its own stated relative-drop definition. Report stays **mixed**, not reframed to a weaker positive claim.

“More gracefully” means exactly: at `σ_c = 0.05` rad/cycle, the gate’s ACC drop is smaller than the router’s by more than the bootstrap CI width. Absolute ACC need not stay near clean. Here the router does not drop (slightly rises within noise); the gate drops a little. Relative grace fails.

---

## 2. What counts

| Counts as registered evidence | Notes |
|-------------------------------|--------|
| Primary gate vs **no-pilot** router on ACC, S, L, T_sw, C_act, C_energy | H1 decision uses ACC + C_act clauses above |
| `σ_c` grid in **rad/cycle** | Step per sample = `σ_c / N_spc` with `N_spc = 100` at 4 kHz / 40 Hz |
| Bootstrap 95% CIs over **seeds** (not symbols) | As in `summary.csv` `*_lo` / `*_hi` |
| Frozen router weights / readouts and gate pilot-phase readouts | Hashes in `results/T4_router_stats.txt` and gate readout artifact |
| Run start after `results/PREREG.sha256` | See `run_meta.json` |

Cost units stay native and **not cross-comparable**:

- Gate `C_ctl` ≈ **34.7 control bits**
- Router `C_ctl` = **2560 MACs / decision**
- Cross-system cost comparison that means something: **C_act** and **C_energy** on the same trials

Clean **selection / routing** ACC is a **sanity check**, not an H1 outcome (both systems see goal one-hot). Wiring floor was fine (routing ACC ≈ 1.0 in T4).

---

## 3. What does not count

| Does not count toward the primary verdict | Why |
|-------------------------------------------|-----|
| `results/comparison_smoke_exploratory/` | Pre-hash; wrong `σ_c` units (rad/sample ≡ 100×); absolute-time gate FIR |
| Premature hash `PREREG.sha256.superseded_20260921T191201` | Superseded before definitional re-freeze |
| Retuning K_p, τ, τ_m, or K_c after seeing smoke or primary | Forbidden; params remain T0–T3 |
| Moving H1 thresholds after peeking | Would void preregistration |
| Secondary `ACC_router_pilot_ref` | Registered for interpretation; **not** in §6 H1 rule |
| Clean selection ACC alone | Sanity only |

Smoke was **seen and disclosed** in PREREG §10 (exploratory mixed under the legacy grid). Seen-and-quiet would not have been acceptable.

---

## 4. Primary numbers (selected cells)

200 symbols × 10 seeds. Full table: `results/comparison_primary/summary.csv`.

| Condition | ACC_gate | ACC_router | ACC_router_pilot_ref | C_act_g / C_act_r |
|-----------|----------|------------|----------------------|-------------------|
| Clean (`σ_f=σ_j=σ_c=0`) | 0.9915 | 0.9935 | 0.9935 | 119.7 / 256 |
| `σ_c = 0.05` rad/cycle | 0.9770 | 0.9970 | 0.9970 | 119.8 / 256 |
| `σ_f = 0.10` | 0.8915 | 0.9935 | 0.9935 | 119.6 / 256 |
| `σ_j = 0.10` cycles | 0.5585 | 0.8220 | **0.9950** | 119.9 / 256 |
| Combined harsh (`σ_f=0.10`, `σ_j=0.10`, `σ_c=0.05`) | 0.5775 | 0.8140 | **0.9965** | 119.8 / 256 |
| 5 Hz wobble (amp 0.3 rad) | 0.7495 | 0.9935 | 0.9935 | 119.6 / 256 |

Selectivity stays high and leakage near zero on reported cells (`S_gate ≈ 0.99`, `L_gate ≈ 0`). Switching latency is ~1.3 cycles for both arms on the clean cell.

**C_energy** (clean): gate ~4.09×10⁴ vs router ~1.29×10⁵ — same direction as C_act (stencil cheaper on the activity/energy proxies).

Figures: `fig_ACC.png`, `fig_S_L.png`, `fig_T_sw.png`, `fig_cost.png` in `results/comparison_primary/`.

---

## 5. Secondary arm — note for reviewers

**Question the secondary closes:** if the gate wins (or loses) under drift, is that the **stencil / spatial gating**, or merely having a **pilot / alignment reference**?

**Design (PREREG §2.6):**

- **Primary router:** no pilot; bit demod = frozen **24-bin phase histogram** head on the multiplex window (absolute carrier phase binning).
- **Secondary router:** same frozen MLP selection, but bit demod uses the **same pilot-phase resampling** as the gate.

**What we see:**

- At clean and at `σ_c = 0.05` rad/cycle, `ACC_router_pilot_ref` **equals** primary `ACC_router` (both ≈ 0.99). Giving the router the pilot does not change the H1 drift comparison on this grid.
- Under **timing jitter** (`σ_j = 0.10`), the split is large: primary router ACC ≈ 0.82 while pilot-ref secondary ≈ 0.995 (gate ≈ 0.56). The pilot reference rescues **router demodulation** under jitter; it does not rescue the oscillatory gate’s bit path in the same cell.

So for the H1 drift clause at rad/cycle scale, the null is not “router lacked a pilot.” For jitter-heavy cells, alignment reference matters a lot for the router’s phase-hist head — exploratory relative to H1, but useful for interpreting mechanism.

---

## 6. Correctness fixes that were allowed (pre-hash)

Documented single-system / fairness fixes; **not** comparison tuning:

1. **Router bit demod** — primary head is phase-binned (24-bin hist → linear), not flat integrated state and not raw time-FIR as the sole path.
2. **Gate readout** — was absolute-time FIR on gated drive; fixed to **open-pilot phase** resampling. Diagnostic: T3-native lock still passes at σ_c=0.05 (fractional-freq test); harness RW at 0.05 rad/sample (legacy) was much harder. Single-system bit ACC under RW: time_fir 0.535 → pilot_phase 0.660. **K_p / τ untouched.** See `results/diagnostics/gate_readout_vs_pilot.md`.
3. **`σ_c` units** — unified to **rad/cycle** so T3 framing, gate, and router see one disturbance definition.

---

## 7. How to read this against H0 / H1

- **Supported (partial):** under clean conditions the stencil matches router ACC within 0.02 and uses about half the active units (C_act) and lower C_energy.
- **Not supported:** greater relative robustness under shared drift at `σ_c = 0.05` rad/cycle. On that clause H1 fails; the combined pattern is **mixed**.
- **H0-relevant note:** once mismatch/jitter are large (`σ_f` / `σ_j` = 0.10), gate ACC falls hard while the no-pilot router holds better on several cells; the pilot-ref secondary router holds near ceiling under jitter. That is consistent with “advantage closes under realistic mismatch/jitter” for the oscillatory sheet path as implemented — exploratory relative to the exact H0 wording, but directionally aligned.

A null / mixed result with cause localization (unit mismatch in early smoke; readout reference bug; then a clean relative-drift failure after repairs) is still a first measured comparison of this composition. Do not upgrade it to H1 support.

---

## 8. Exploratory (explicitly not primary)

Anything in `results/comparison_smoke_exploratory/`, pre-demod router ACC interpretations, absolute-time gate FIR curves, and narrative “almost H1 on cost alone” claims without the drift clause. New measures belong only in a clearly labeled exploratory addendum, never back-edited into `PREREG.md`.

---

## 9. Reproduce

```bash
cd /path/to/oscillatory-gating
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# confirm hash matches PREREG.md
sha256sum PREREG.md   # expect c81851b0…fef10
.venv/bin/python run_comparison.py --full   # writes results/comparison_primary/
```

Primary metadata: `results/comparison_primary/run_meta.json` · verdict: `H1_verdict.txt`.
