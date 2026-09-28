# PREREG — T10: Coherence as an Agreement Classifier (v3 FROZEN)

| Field | Value |
|-------|--------|
| **Title** | T10 — Reconciliation: coherence classifies agreement under matched plasticity |
| **Date** | 2026-09-22 (PT); **v3 frozen** 2026-09-22T22:58:39 CDT |
| **Frozen at (v1)** | 2026-09-22T20:37:41 CDT |
| **Frozen at (v2)** | 2026-09-22T21:21:10 CDT |
| **Frozen at (v3)** | 2026-09-22T22:58:39 CDT |
| **Owner** | Kyle (ZetaZero) |
| **Status** | **FROZEN (v3)** — `results/PREREG.sha256` written 2026-09-22T22:58:39 CDT. Archives: v1 `docs/PREREG_v1_ae2f212a.md`, v2 `docs/PREREG_v2_64bf9f73.md`. Supersedes v2 `64bf9f73…`. Multi-seed primary **not** started (awaiting Kyle authorize). |
| **Relation** | Separately preregistered follow-on to closed T7, T8, T9. Those tickets remain **FROZEN**. |
| **T9 closing citation** | RESULTS.md sha256 `d1c70486c40e637e22e1e6d034379d566fd365306b72bb30fd34dedb500c4fd1`; closing: **Phase selects; it does not amplify; and it does not lower loss.** Survivors: **H2 / H3 only.** |
| **Drive folder** | `1BAfXrMq_YOgUfjVAHKaTrLb0UdDfyXsp` |

**Supersedes v2** (`docs/PREREG_v2_64bf9f73.md`, sha `64bf9f73ad4c6d47d96d3ba4e73d518329de1b087e95bc62693965d7dfdf2e88`). One-seed / plumbing numbers remain **non-counting**.

---

## 0. Why v3 (flaws found in n16 smoke)

n16 plumbing (`results/exploratory/plumbing_seed20260921_n16/`) exposed three scientific flaws. v3 fixes them **before** any multi-seed primary:

1. **Plastic update path differed by arm** — `coh_*` / `random_tag` installed PhaseMediumFFN so plastic grads were demodulated / coherence-weighted; `loss_delta` / `ungated` used plain_sum. Equal AdamW budget ≠ equal update. **Fix (chosen):** identical plain_sum update on every arm when admit; admission rule is the sole arm difference for Forget/Acquire attribution.
2. **Oracle was circular for phase Agree** — `write_phases` were planted from `oracle_case`, so coh Agree_acc=1.0 by reading the label. **Fix:** oracle is **content-defined** (expert rule vs conflict remap); phase/tag are follow-on side-channels for those gates only.
3. **loss_delta Agree ≈ 0.25 on 3 classes** — 1-step both-probe S_L was noise; ŷ was 2-class. **Fix:** S_L := d_old after K=5 preview steps; three-class S_L_LOW/S_L_HIGH + TAU_L fitted on hold-out (Agree > chance on cal split).
4. **Admit-conditioned Forget/Acquire** now **primary** matched comparison (unconditional retained as secondary).

Diagnosis with file:line cites: `docs/DIAGNOSIS_v3.md`.

---

## 1. Mission

Treat **coherence as a classifier of agreement** — whether a candidate expert update is **consistent with**, **in conflict with**, or **partially overlapping** that expert's existing **content rule** — rather than as a **source of working power** (lower L / higher G / better Rec).

---

## 2. Protocol lock: H_match (fact — not scored)

T9 matched-train exploratory established that unequal non-expert train budgets invent ordinary-loss gaps.

**Every T10 arm, full stop:**

1. Same **training budget** (steps, batch, optimizer, schedule, data mix for plastic updates).
2. Same **plastic surface** (§4) — including `plain_sum` / loss-baseline arms.
3. Same **forward path for plastic updates** (v3): plain_sum MoE FFN only — **no** medium install on the update path.
4. Same **active-compute reporting** discipline.

H_match is **not** a T10 primary hypothesis. Scoreboard cites T9.

---

## 3. Hypotheses (exact scoring)

### 3.1 Arms (primary grid) — unchanged set

| Arm | Role |
|-----|------|
| **`coh_hum`** | Primary coherence policy: phase side-channel + HumRef geometry for the **gate**; admit iff ŷ ≠ conflicting. |
| **`coh_fixed`** | FixedRef coherence gate (geometry control). |
| **`random_tag`** | Amplitude / orthogonal-tag mirror gate (no phase). |
| **`loss_delta`** | Null baseline: identical plastic surface; admit iff S_L > τ_L (§5). |
| **`ungated`** | Always-admit plastic updates (capacity / damage control). |

All arms start from the same T9 T1 frozen-experts checkpoint (`2ef3bfa53b4209497e6c9efd4c25f6cd3c7e1fae709a151b78a764226888d9eb`). Seeds: **`[20260921, 20260922, 20260923, 20260924, 20260925]`**.

### 3.2 H_agree (primary) — coherence classifies content-defined `oracle_case`

Episodes carry generator-emitted `oracle_case ∈ {consistent, conflicting, partial}` defined by **content rule** (§6). Gate emits ŷ. **Agree_acc** = macro accuracy of ŷ vs oracle_case.

**H_agree supported** if all hold (same dual-bootstrap rules as v2):

1. Seed-mean macro Agree_acc(`coh_hum`) **> 1/3** (chance).
2. Seed-mean macro Agree_acc(`coh_hum`) **>** Agree_acc(`loss_delta`).
3. **Phase attribution:** Agree_acc(`coh_hum`) **>** Agree_acc(`random_tag`).

**Disclosure:** coh_* Agree still uses the phase side-channel that *follows* the content label; interpret H_agree phase clause with that caveat (oracle itself is content-primary).

### 3.3 Forget_old / Acquire_new — **admit-conditioned PRIMARY**

| Measure | Definition |
|---------|------------|
| **Forget_old** | `L_after_old − L_before_old`. **Positive = forgetting**. |
| **Acquire_new** | `L_before_new − L_after_new`. **Positive = acquisition**. |

**Primary reporting (v3):** seed-mean ± bootstrap CI of Forget_old / Acquire_new **conditioned on admit=True** (matched comparison across arms when an update actually runs). Also report unconditional means as secondary.

Interpretive expectations unchanged from v2 (conflicting rejects should spare Forget relative to ungated; consistent admits should Acquire without catastrophic Forget).

### 3.4 H_select / 3.5 H_persist — regression locks (real; can fail)

Unchanged from v2: real T9 H2/H3 and T8 λ locks via import-only APIs. Must remain runnable and able to fail.

### 3.6 Null (live)

Coherence classifies **no better than** `loss_delta` under matched budget **and identical update path**, or any win is amplitude-tag rather than phase.

---

## 4. Plastic surface + identical update path (v3)

### 4.1 Named parameters (unchanged)

Plastic set = output Linear of expert `e*` only (`experts[e*].net[2].weight/bias`) ≈ **32896** params. All else frozen.

### 4.2 Plastic budget (matched; unchanged)

steps=200, batch=32, AdamW lr=2e-3, wd=0.01, warmup=10, cosine, grad clip=1.0. Episodes/seed counting: 48 = 16/case.

### 4.3 Identical update path (v3 choice)

**When admit=True, every arm runs the same plain_sum MoE forward + same AdamW recipe on the same plastic tensors.** Medium FFN is **not** installed for the plastic window or for Forget/Acquire probe NLL. Gate scoring may still use phase/tag geometry; that geometry does **not** scale plastic gradients.

Rationale: n16 budget audit showed equal optimizer budget but medium arms had L_before≈2.4 vs plain_sum≈0.64 and ~25× larger |Forget|/|Acquire| — a path confound, not a science result. v3 removes that confound so Forget/Acquire attribute to the **admission rule**.

Report per arm: admit_rate, mean steps_run, mean steps_run|admit, n_params, tokens_seen, Forget/Acquire **|admit** (primary) and overall (secondary).

---

## 5. Gates

### 5.1 Coherence / tag score S

Unchanged geometry formulas (ReLU half-wave cos / |tag cos|). Inputs are **follow-on** phase/tag side-channels (§6), not the oracle definition.

ŷ map: S ≥ S_high=**0.85** → consistent; S ≤ S_low=**0.25** → conflicting; else partial. Tag-partial √1/2 stays partial.

### 5.2 Loss score S_L (v3)

On a held probe set fixed before the plastic window:

- Run **K = 5** virtual AdamW steps on a detached copy (same plastic surface, plain_sum).
- Train on new-regime batch (e* forced as task marker).
- **S_L := L_before_old − L_after_old** (d_old compatibility: does training on new content improve old-regime NLL?).

Also log S_L_both for disclosure.

Admit (`loss_delta`): **S_L > τ_L**.

ŷ (`loss_delta`): S_L ≥ S_L_high → consistent; S_L ≤ S_L_low → conflicting; else partial.

### 5.3 Thresholds (calibrated hold-out; see docs/CALIBRATION.md)

| Symbol | Value | Role |
|--------|------:|------|
| S_high | **0.85** | phase/tag → consistent |
| S_low | **0.25** | phase/tag → conflicting |
| τ_L | **0.00064452** | loss_delta admit |
| S_L_high | **0.00064452** | loss_delta ŷ consistent |
| S_L_low | **−0.00186276** | loss_delta ŷ conflicting |

Calibration data_seed = `20260921 + 1_000_003` (disjoint from smoke/primary). Hold-out loss_delta Agree_acc ≈ **0.417 > 1/3** chance (see calibration disclosure).

**Admit rule:**

| Arm | Admit iff |
|-----|-----------|
| coh_hum / coh_fixed / random_tag | ŷ ≠ conflicting |
| loss_delta | S_L > τ_L |
| ungated | always |

---

## 6. Agreement oracle — **content-defined** (v3)

### 6.1 One-sentence conflict build (no phase)

**A conflicting episode is built by generating new/train sequences under the expert's next_old rule plus Δ=28 (mod 56), so every next-symbol disagrees with the expert's learned content rule.**

### 6.2 Construction

| oracle_case | Content construction (DEFINES the label) |
|-------------|------------------------------------------|
| **consistent** | new + train via `next_old` (same affine rule as expert e*) |
| **conflicting** | new + train via `next_conflict = (next_old + 28) mod 56` |
| **partial** | 50/50 mix of next_old / next_conflict (seeded) |

`oracle_case` is locked at creation; never rewritten from model/gate/loss. Unit tests attempt forbidden rewrite. Helper `infer_oracle_case_from_content` recovers the label from sequences alone (no phase).

### 6.3 Follow-on side-channels (NOT the oracle)

After the content label is chosen, write_phases / tags are filled for coh_*/random_tag **gates**:

| case | write phases | tag |
|------|--------------|-----|
| consistent | φ_e* | r_e* |
| conflicting | φ_e*+π | r_(e*+4) |
| partial | half/half | normalize(r_e*+r_partner) |

These side-channels let phase/tag gates operate; they **follow** content rather than defining it.

---

## 7–13. System imports / cost / bootstrap / disclosure / out-of-scope / deliverables

Unchanged from v2 except: biology = motivation only; disclose peeks; no counting primary until Kyle authorizes multi-seed after this freeze.

### Freeze statement (v3)

Parent freeze authorized by Kyle ("Freeze", 2026-09-22 evening PT) on the **v3** PREREG after: n16 smoke exposed path/oracle/loss_delta flaws; identical plain_sum update path; content-defined oracle; calibrated loss_delta (Agree≈0.417); admit-conditioned Forget/Acquire primary. Frozen at 2026-09-22T22:58:39 CDT. v2 sha `64bf9f73ad4c6d47d96d3ba4e73d518329de1b087e95bc62693965d7dfdf2e88` and v1 sha `ae2f212ae73aba9a3a7bbd98a82f4c64e2d4c05441f6783401d8020e57b9819e` archived under `docs/`. Multi-seed primary must **not** start until Kyle explicitly authorizes; this freeze alone does not start primary. Disclosed caveats remain in-doc: phase side-channel for coh_* Agree; loss_delta cal Agree≈0.417.
