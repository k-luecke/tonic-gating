# PREREG — Oscillatory Gating vs Learned Router

| Field | Value |
|-------|--------|
| **Title** | Oscillatory Gating vs Learned Router |
| **Date** | 2026-09-21 |
| **Owner** | Kyle (ZetaZero) |
| **Status** | Frozen before T5 / T7 comparison runs |
| **Source** | Handoff Sections 3–4 (system spec + measures), Mission hypotheses; free parameters fixed from code as implemented in T0–T4 |

This document freezes the preregistration **as decided before seeing any gate-vs-router comparison results**. It must not be edited to match outcomes. Exploratory additions after results belong only in a clearly labeled exploratory section of `RESULTS.md` (or equivalent), never here.

---

## 1. Hypotheses (Mission, exact)

**H1.** Oscillatory gating matches the router on selectivity and leakage under clean conditions at lower cost, and degrades more gracefully under shared drift, because alignment is carried by a pilot that drifts with the senders rather than by per-input computation.

**H0.** The router matches or beats the gate on cost, or the gate's advantage closes once mismatch and jitter reach realistic levels (unit frequency spread of 5 to 10 percent, jitter of 5 to 10 percent of a cycle).

What counts as a result: either outcome, reported against the preregistered numbers below. Report **supported**, **H0**, or **mixed** in those words (see §8).

---

## 2. System specification (frozen)

### 2.1 Substrate (`sheet.py`)

| Parameter | Frozen value |
|-----------|----------------|
| Grid | 16 × 16 (256 units) |
| Patches \(K\) | 8 contiguous patches of 32 units |
| **Patch geometry** | **4×8 blocks** (4 rows × 8 cols). On the grid: 4 patch-rows × 2 patch-cols; patch \(k = p_r\cdot 2 + p_c\) owns rows \([4p_r\,..\,4p_r+3]\), cols \([8p_c\,..\,8p_c+7]\). *(Choice recorded in code/README; handoff allowed 2×16 or 4×8.)* |
| Carrier \(f_0\) | 40 Hz |
| Sample rate \(f_s\) | **4000 Hz** (`lockin.FS`; `DT = 1/4000`) |
| Symbol | 5 carrier cycles → \(T_\mathrm{sym} = 0.125\,\mathrm{s}\), **500 samples/symbol** |
| Unit frequency | \(f_i \sim f_0\,(1 + \mathcal{N}(0,\sigma_f))\) |
| Gain | \(g_i(t) = \max(0, \cos\phi_i(t))\) (oscillatory path); router path uses `gain_mode="fixed"` → \(g=1\) |
| Membrane | Leaky integrator \(s_i\), \(\tau_m = 2\) cycles \(\Rightarrow \tau_m = 2/f_0\) seconds |
| Coupling | Within-patch 4-neighbor Kuramoto; **\(K_c = 0.1 \times 2\pi \times 40\) rad/s** per edge; cross-patch edges omitted |
| PLL (when pilot present) | First-order loop from `lockin.py`: **\(K_p = 800\)**, **\(\tau = 3\) cycles** (\(\tau = 3/f_0\) s) |

### 2.2 Controller / stencil (`stencil.py`)

Per patch \(k\), slow control (updated per trial / target change):

- **Inhibition** \(I_k \in \{0,1\}\): 1 = closed, 0 = open. Base task: exactly one patch open.
- **Pilot** at 40 Hz, amplitude **`PILOT_AMP = 0.5`**, phase \(\psi_k\):
  - open patch: \(\psi = \theta_{d^*}\)
  - closed patches: \(\psi = \theta_{d^*} + \pi\)
- Units PLL-lock to the pilot (never to summed content).
- Controller is set from the trial goal \(d^*\), not from the input.

### 2.3 Content (`content.py`)

- **\(D = 8\)** phase-multiplexed streams, \(\theta_d = 2\pi d / 8\).
- **Positive-only volleys:** \(\max(0,\cos(\cdot))^2\) — never signed carriers.
- On/off symbol every 5 cycles; streams summed and delivered to every unit.
- Task: report the bit of target stream \(d^*\). Categories map 1:1 to patches.

### 2.4 Readouts

Per-patch linear readout on the symbol window / integrated state, trained once on clean data with the correct patch open, then **frozen**.

### 2.5 Router baseline (`router.py`) — as actually implemented

Same 8 patches; sheet with **`gain_mode="fixed"`** (no rhythm). Stencil replaced by a small MLP:

| Item | Frozen implementation |
|------|------------------------|
| Architecture | MLP, hidden **64**, softmax over 8 patches |
| **Features (honest)** | **32-dim = 24-bin amplitude-weighted carrier phase histogram ‖ 8-dim goal one-hot \(d^*\)** — *not* a bare 32-bin histogram. Handoff T4 text said “e.g. 32-bin phase histogram”; the built system concatenates a **24-bin** hist with goal one-hot so the router has the same goal access as the stencil while still seeing content features. |
| Training | **2000 clean symbols**, then **freeze** weights |
| Selection | top-\(k\) (base task \(k=1\)); Control = inhibition mask only (**no pilot**) |
| Cost accounting | report **parameter count** and **MACs per decision** (dense matvecs: \(F\cdot H + H\cdot K\)) |

Frozen artifacts live under `results/router_weights.npz` and `results/router_readouts.npz` (with content hashes in `results/T4_router_stats.txt`).

---

## 3. Disturbance model (applied identically to both systems)

| Factor | Grid |
|--------|------|
| **Mismatch** \(\sigma_f\) | \(\{0,\,0.02,\,0.05,\,0.10\}\) (fractional unit frequency spread) |
| **Jitter** \(\sigma_j\) | \(\{0,\,0.02,\,0.05,\,0.10\}\) cycles; each volley time-shifted by \(\mathcal{N}(0,\sigma_j)\); receiver gain phase gets the same per-cycle jitter |
| **Shared drift** \(\sigma_c\) | \(\{0,\,0.01,\,0.03,\,0.05\}\) — common phase random walk step **\(\sigma_c\) rad per sample**, added to **all content streams and to the pilot** |
| Full factorial | All \(4\times 4\times 4\) disturbance cells |

**Stretch (correlated disturbance):** 5 Hz sinusoidal phase wobble of amplitude **0.3 rad** hitting all streams and the pilot together (common-mode / photonics-protocol analogue). Report alongside the grid; does not replace the primary grid.

**Critical prior:** the pilot **must share** disturbances with senders for the drift advantage. If a run breaks shared drift, it is not the preregistered H1 test.

---

## 4. Trial counts

| Protocol | Count | Status |
|----------|-------|--------|
| **Full (preregistered)** | **200 symbols × 10 seeds** per condition; all disturbance cells + wobble stretch | **This is the comparison** |
| Switching latency run | **50 switches** per condition (dedicated switching protocol) | Preregistered for \(T_\mathrm{sw}\) |
| **Smoke / dev defaults** | Any smaller \(N_\mathrm{symbols}\), fewer seeds, or subset of the grid used for wiring checks | **Explicitly non-preregistered** — must not be reported as the primary result or used to tune |

Same trial streams (seeds, symbols, disturbance realizations) are generated once and fed to both systems.

---

## 5. Measures (exact formulas)

Let \(A_k\) = mean integrated activity of patch \(k\) over a symbol, \(k^*\) = target patch, \(A_0\) = resting activity of a closed patch with no input.

### 5.1 Selectivity
\[
S = \frac{A_{k^*}}{\sum_k A_k}
\]
Range \([0,1]\); chance \(= 1/K\). If \(\sum_k A_k \approx 0\), define \(S = 0\).

### 5.2 Leakage
\[
L = \operatorname{mean}_{k \neq k^*}\;\frac{\max(0,\, A_k - A_0)}{A_{k^*}}
\]
\(L = 0\) means nothing above rest outside the target.

### 5.3 Task accuracy (primary endpoint)
\[
\mathrm{ACC} = \text{fraction of symbols where the target patch's readout reports the correct bit.}
\]
\(S\) and \(L\) explain ACC; they are not substitutes for it.

### 5.4 Switching latency
After the controller changes target from \(k_1\) to \(k_2\) at \(t_0\),
\[
T_\mathrm{sw} = \text{first time (in cycles) at which } A_{k_2} > A_{k_1} \text{ and stays so for one full cycle.}
\]
Measured on the dedicated switching run (50 switches / condition).

### 5.5 Cost (three numbers, never combined)

**\(C_\mathrm{act}\)** — mean number of units with gain \(> 0.1\) per time step (active fraction × 256).

**\(C_\mathrm{ctl}\)** — bits / compute to specify control:

- Gate: \(\log_2(K) + K \times \log_2(\textit{slots})\) (open-patch index + phase-slot bits per patch; *slots* from the lock-in prior at the relevant jitter).
- Router: parameter count × 1 per symbol evaluation, **reported as multiply-accumulates (MACs) per decision**.

**\(C_\mathrm{energy}\)** — proxy for synaptic events:
\[
C_\mathrm{energy} = \sum_{\text{units},\,t} \bigl(g_i(t) \times |u_i(t)|\bigr)
\]
Report for both systems on the **same** trials.

Do **not** combine \(C_\mathrm{act}\), \(C_\mathrm{ctl}\), \(C_\mathrm{energy}\) into one score.

---

## 6. Primary comparison rule and H1 support criteria

**Primary comparison:** ACC and \(C_\mathrm{act}\) at each disturbance cell, gate vs router, with **bootstrap 95% CIs over seeds** (resample **seeds**, not symbols within a seed).

**H1 is supported if all of the following hold:**

1. **Clean cell:** \(\mathrm{ACC}_\mathrm{gate} \ge \mathrm{ACC}_\mathrm{router} - 0.02\) **and** \(C_{\mathrm{act},\,\mathrm{gate}} \le 0.5 \times C_{\mathrm{act},\,\mathrm{router}}\).
2. **Shared drift** \(\sigma_c = 0.05\): the drop in \(\mathrm{ACC}_\mathrm{gate}\) is **smaller** than the drop in \(\mathrm{ACC}_\mathrm{router}\) by **more than the CI width**.

Any other pattern is reported as **H0** or **mixed**, in those words. No silent reframing to a weaker claim that happens to hold.

---

## 7. Analysis hygiene (frozen rules)

1. **Do not add measures after seeing results.** New quantities go only in a clearly labeled **exploratory** section of the report.
2. Fix gate parameters (\(K_p\), \(\tau\), \(K_c\), \(\tau_m\), pilot amp, positive-only, patch layout) from single-system lock-in / sheet tests — **never from the comparison**.
3. Router may be tuned on **clean data only**, then frozen before the comparison.
4. Bootstrap over seeds, not symbols.
5. Cost channels stay separate.
6. No biological claims in the result write-up beyond motivation/citation.

---

## 8. Prior constraints from lock-in (T0 / T1)

These are **inputs** to the preregistration, not comparison outcomes:

| Finding | Constraint on the sheet comparison |
|---------|-------------------------------------|
| T0 slots | Pulsed senders: usable separation → ~9–18 slots/cycle (no receiver jitter) |
| T1 jitter | At \(\sigma_j=0.05\), slots fall toward ~4–8 (biology-scale); use for \(C_\mathrm{ctl}\) slot bits where relevant |
| Positive-only is load-bearing | Content = positive volleys only; signed carriers forbidden |
| Fixed reference fails under drift | Do not use a fixed phase reference for the gate path in the preregistered runs |
| PLL must lock to pilot, not content | Content-locked PLL collapses selectivity; pilot amp 0.5 with **shared drift** |
| Pilot shares disturbance | Required for H1’s drift clause; otherwise the gate has no preregistered drift advantage |

---

## 9. Out of scope (this sprint)

- Consolidation / compilation
- Learning in the sheet (sheet fixed; only the router is trained)
- Hardware ports (SpiNNaker / BrainScaleS-2 notes are stretch T8, not part of this freeze)

---

## 10. Provenance pointers

- Full handoff: [Handoff: Oscillatory Gating vs Learned Router](https://docs.google.com/document/d/1O6JYHOctJB_QomccStfsmV4COwNznZhY4YJiWZU35dU)
- Local condensation: `HANDOFF_EXCERPT.md`
- Constants: `lockin.py` (`F0`, `FS`, `PLL_K`, `PLL_TAU`, `PILOT_AMP`)
- Implementation: `sheet.py`, `stencil.py`, `content.py`, `router.py`

**Freeze statement:** This file records decisions made for T6 before T7. Comparison runs must cite this document (and later an external hash/commit anchor). Editing measures or H1 thresholds after peeking at T5/T7 outputs voids the preregistration for that run.
