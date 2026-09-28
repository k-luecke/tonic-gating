# PREREG — Phase-Gated Expert Routing vs Learned Router / Hash Baselines

| Field | Value |
|-------|--------|
| **Title** | Phase-Gated Expert Routing vs Learned Router (LR) and Hash Router (HR1) |
| **Date** | 2026-09-21 (PT) |
| **Owner** | Kyle (ZetaZero) |
| **Status** | Candidate freeze for parent review → counting after `results/PREREG.sha256` |
| **Source** | Handoff Sections 1, 3, 4, 6; free parameters frozen from single-arm T1–T4 |

This document freezes the preregistration **before** any multi-seed primary comparison. It must not be edited to match outcomes. Exploratory one-seed numbers below are **disclosed and non-counting**. New measures after seeing primary results are forbidden except in a clearly labeled exploratory section of `RESULTS.md`.

---

## 1. Hypotheses (exact)

**H1 (routing cost + robustness).** On the shared modular-recurrence language task, the Phase Gate (PG) matches clean predictive quality of the learned top-2 router (LR) at **strictly lower native routing-decision cost**, and **degrades less** under the primary disturbance cells than LR:

1. **Clean sanity / quality tie:** all primary arms achieve heldout loss well below per-task unigram baselines; LR specialization acceptance (≥4/8 tasks with top-1 dominant share ≥0.25) holds on each primary seed (the shared task-embedding interface is part of the frozen model). PG clean heldout loss is within **0.1 nats** of LR on the seed-mean (implementation-parity band used in development for hash checks; applied here as the clean PG↔LR quality gate).
2. **Routing-decision cost (native units, never collapsed into one score):**  
   - LR: \(C_\mathrm{route}=1024\) MACs/token (= \(128\times 8\)).  
   - PG: \(C_\mathrm{route\_ops}=8\) primitive cosine comparisons/token (not MACs).  
   - H1 cost clause: \(8 < 0.1\times 1024\).  
   HR1 reports \(0\) MACs (hash/table lookup); it is a **primary baseline arm** for quality/cost context, not the H1 robustness comparator.
3. **Robustness (primary cells):** at router-input noise \(\sigma_n=0.3\) and at MoE-layer shift \(\delta=0.3\) (units of each arm’s own clean heldout MoE hidden std), let \(\Delta = L_\mathrm{cell}-L_\mathrm{clean}\). H1 requires, for each of these two cells separately, that the paired seed difference \(D=\Delta_\mathrm{LR}-\Delta_\mathrm{PG}\) satisfies:
   - the percentile bootstrap 95% CI over the 5 seeds (10 000 resamples of the paired differences) **excludes 0**, and
   - the point estimate \(D\) **exceeds the width** of that 95% CI  
   (“by more than interval width,” with the explicit bootstrap method above; both clauses required — do not silently drop either).

**H2 (switching; PG vs LR only).** On the switching split, \(T_\mathrm{sw}\) (definition in §4) is smaller for PG than LR under the same paired-seed bootstrap rule as H1 clause 3 (CI on \(T_{\mathrm{sw},LR}-T_{\mathrm{sw},PG}\) excludes 0 **and** the gap exceeds CI width). **HR1/HR2 crossings are reported but are not task-aware and must not be interpreted as an H2 baseline.**

**H0.** Any failure of the H1 clauses (clean tie, cost, or either primary robustness cell) is reported as **H0** or **mixed** in those words. H2 is scored separately and does not rescue a failed H1.

**What counts as a result.** Either outcome, reported against these exact criteria. T7 (follow-ups) runs **only if H1** is supported.

**Primary arms for counting:** **LR, HR1, PG** (5 seeds × 3 arms = 15 trainings). **HR2** is a registered **secondary / implementation-parity** variant (two-hash top-2), reported from development and optionally evaluated if budget permits; it **does not enter H1/H0 primary criteria**.

---

## 2. System specification (frozen)

### 2.1 Data

| Item | Frozen value |
|------|----------------|
| Data seed | `20260921` |
| RNG | `numpy.random.PCG64` |
| Vocab | markers `0..7`; content `8..63` (56 symbols) |
| Sequence length | 128 |
| Rule | \(x_{next}=(m[t]\,x+b[t]) \bmod 56\) with fixed multipliers/offsets in `data/metadata.json` |
| Train | 256 sequences/task → shape `[2048,128]` |
| Heldout | 64 sequences/task → `[512,128]` |
| Switching | 512 sequences with a mid-sequence distinct marker restarting the new recurrence |
| Manifest | `data/MANIFEST.sha256` (bytes frozen; regenerate only if deliberately re-frozen) |

### 2.2 Shared model

| Item | Frozen value |
|------|----------------|
| Architecture | 4-layer decoder, width 128, 4 heads, FFN width 256 |
| MoE layer | Layer index 2 (third layer); 8 experts `128→256→128` GELU |
| Task interface | Learned **8×128 most-recent-marker task embedding**, broadcast to every token, **identical for all arms**; updates at switching markers |
| Specialization repair | Disclosed pre-freeze: original marker-only LR was 0/8 dominant tasks; repaired shared embedding yielded 8/8 on one-seed exploratory. Routing-arm semantics unchanged. |

### 2.3 Routing arms

| Arm | Role | Frozen definition |
|-----|------|-------------------|
| **LR** | Primary | Linear hidden→8 router; softmax; **top-2** renormalized gates; Switch-style aux \(E\sum_e(\mathrm{load}_e\cdot\mathrm{importance}_e)\) with weight **0.01** |
| **HR1** | Primary baseline | Parameter-free hash `(5·token+1) mod 8`; top-1; aux=0 |
| **HR2** | Secondary only | HR1 + `(3·token+2) mod 8`; equal gates; aux=0; **not in H1/H0** |
| **PG** | Primary | Fixed expert phases \(2\pi e/8\); learnable task phases (init aligned); gain \(\max(0,\cos(\phi_e-\phi_\mathrm{task}))\), normalize over positive gains only; marker-controlled |

### 2.4 Training (every primary seed/arm)

| Item | Frozen value |
|------|----------------|
| Optimizer | AdamW, lr **0.002**, weight decay 0.01 |
| Schedule | Linear warmup 50 steps, then cosine down to 0.1× base |
| Steps | **600** |
| Batch | **32**, shuffle with `Generator.manual_seed(seed)`, drop_last |
| Grad clip | 1.0 |
| Init / data order | `seed_everything(seed)` + DataLoader generator seed = run seed |
| Seeds | **`[20260921, 20260922, 20260923, 20260924, 20260925]`** |
| Device | CPU or CUDA allowed; deterministic algorithms + same step count; device recorded in metrics |

---

## 3. Disturbance model (T4 harness)

Grid for **router-input noise** and **MoE-layer shift** multipliers:

\[
\{0,\ 0.1,\ 0.3,\ 1.0\}
\]

**Units / calibration (frozen choice):** each arm measures its **own** clean heldout MoE-layer hidden standard deviation once; that arm’s std is reused for all of its cells. Disturbances use a **shared seeded base realization** per seed/cell (same unit-RMS shift direction seed `seed+101`; same noise RNG seed `seed+10000+round(1000·units)` across arms), scaled by each arm’s own std:

\[
\text{noise/shift} = (\text{shared unit tensor}) \times (\text{units}) \times \sigma_{\mathrm{hidden},\,\mathrm{arm}}.
\]

**Placement (exact):**

- **Noise** enters the **routing-decision hidden only** (experts do not see the noise).
- **Shift** is added to **every** MoE-layer hidden state (**routing and expert input**).

Also evaluated (not primary H1 cells): single-expert dropout for each of 8 experts; switching split for \(T_\mathrm{sw}\).

---

## 4. Measures (exact)

| Symbol | Definition |
|--------|------------|
| \(L\) | Mean content-token NLL on heldout (markers excluded from loss mask) |
| \(C_\mathrm{route}\) | **Routing decision** cost in **native units**: LR = **1024 MACs/token**; HR1/HR2 = **0 MACs** (optional exploratory hash-op counts); PG = **`C_route_ops=8`** primitive cosine comparisons (not MACs). Never combine with \(C_\mathrm{exp}\) into one score. |
| \(C_\mathrm{exp}\) | \(\texttt{mean\_active\_experts\_per\_token} \times 65536\), where \(65536=128\times256+256\times128\) (activation ignored consistently). Report `mean_active_experts_per_token`, `expert_MACs_per_expert`, `C_exp_MACs_per_token`. |
| \(B\) | Load-balance CV = population std/mean of aggregate expert dispatch mass on heldout |
| \(\Delta L\) | **`delta_L_from_clean` = \(L_\mathrm{cell}-L_\mathrm{clean}\)** (primary robustness). Exp transform may be stored as exploratory only. |
| \(\Delta L_\mathrm{drop}\) | Mean over 8 single-expert dropouts of \(L_\mathrm{drop}-L_\mathrm{clean}\) |
| \(T_\mathrm{sw}\) | Per-task dominant expert set = smallest expert prefix with cumulative clean-heldout dispatch mass ≥0.5. After the switching marker, first lag with >0.5 mass on the **new-task** set; **index 0 = immediate at the marker**. |

**Primary robustness cells:** \(\sigma_n=0.3\) and \(\delta=0.3\).

**Bootstrap (frozen):** over the 5 seeds, resample **paired seed differences**, 10 000 replicates, percentile 95% CI.

---

## 5. Primary comparison rule

1. Train LR, HR1, PG on seeds `20260921..20260925` into `results/primary/` (15 runs).
2. Run T4 harness per seed for arms `lr hr1 pg`; write **one JSON per arm/seed/cell** under `results/primary/harness/` (or equivalent `results/primary/` tree documented in `RESULTS.md`).
3. Score H1 and H2 exactly as §1; report **supported**, **H0**, or **mixed** in those words.
4. Budget target: **under 4 GPU-hours** (CPU wall-clock acceptable if deterministic settings match).
5. **No new measures after results.** **T7 only if H1.**

---

## 6. Rules of the road

1. No multi-seed primary before `results/PREREG.sha256` exists.
2. Fix parameters from single-arm T1–T4 only; never from the comparison.
3. Same data splits and seeded disturbance realizations across arms within a seed/cell.
4. Cost channels stay in native units; no hidden composite score.
5. HR2 does not enter H1/H0; H2 is PG vs LR only.
6. Provenance on every JSON: source-tree SHA256 manifest, seed, data manifest hash, checkpoint hash, definitions.
7. External anchor is parent’s Drive upload of `results/PREREG.sha256` (no public gist/pastebin).

---

## 7. Mandatory disclosure — one-seed exploratory seen before freeze

All of the following were seen **before** this freeze and are **non-counting**:

### 7.1 Specialization repair

- Original marker-only LR (600 steps): language learned (~0.0015 nats) but **0/8** tasks at ≥0.25 top-1 dominant share.
- Shared latest-marker task embedding repair: LR **8/8**, heldout **0.001323**. Retained aux weight 0.01; no arm comparison used to tune it.

### 7.2 One-seed metrics (seed `20260921`)

| Arm | heldout \(L\) | mean active experts/token |
|-----|--------------:|--------------------------:|
| LR | 0.001323 | 2.000 |
| HR1 | 0.001298 | 1.000 |
| HR2 | 0.001431 | 2.000 |
| PG | 0.001377 | 3.875 |

### 7.3 T4 pre-fix bugs (fixed before this document)

1. Disturbance grid wrongly used `{0.5, 1.0}`; corrected to `{0, 0.1, 0.3, 1.0}`.
2. `C_route` wrongly = active dispatch count; corrected to routing-decision MACs/ops as in §4.
3. `C_exp` wrongly = experts used anywhere; corrected to mean-active × 65536.
4. Robustness primary field is `delta_L_from_clean` (not exp); exp retained exploratory only.
5. Noise must hit **routing hidden only**; shift hits **routing + expert input** — implementation corrected and tested.

### 7.4 Post-fix exploratory harness snapshot (seed `20260921`, non-counting)

| Arm | \(C_\mathrm{route}\) | \(C_\mathrm{exp}\) MACs/tok | \(T_\mathrm{sw}\) | mean \(\Delta L_\mathrm{drop}\) | \(\Delta L\) noise 0.3 | \(\Delta L\) shift 0.3 |
|-----|----------------------|----------------------------:|------------------:|-------------------------------:|-----------------------:|-----------------------:|
| LR | 1024 MACs | 131072 | 1.488 | 0.243 | 0.000669 | 0.000296 |
| HR1 | 0 MACs | 65536 | 0.507† | 0.075 | 0.0 | 0.000144 |
| HR2 | 0 MACs | 131072 | 0.738† | 0.0003 | 0.0 | 0.000152 |
| PG | 8 ops | 253952 | 0.000 | 0.003 | 0.0 | 0.000237 |

† HR \(T_\mathrm{sw}\) not task-aware. Noise \(\Delta L=0\) for HR/PG is expected under routing-only noise because those routers ignore the hidden state.

**No multi-seed comparison was seen before writing this file.**

---

## 8. Primary registered run procedure

1. Write `results/PREREG.sha256` = sha256(`PREREG.md`) + `created_iso` (local PT).
2. Any run started before that file exists is exploratory only.
3. Primary must log `mode: primary`, `prereg_sha256`, and `start_time_iso` with start ≥ hash timestamp.
4. Outputs under `results/primary/`; one JSON per arm/seed/cell for harness cells.

---

## 9. Freeze statement

This file records decisions for the phase-routing comparison before the counting multi-seed run. Editing H1/H0/H2 thresholds or primary measures after peeking at primary outputs voids the preregistration for that run. Parent must upload `results/PREREG.sha256` to Drive as the external anchor; **do not** create a public gist or pastebin.
