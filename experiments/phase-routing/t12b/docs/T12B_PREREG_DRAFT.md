# PREREG DRAFT — T12b Persistence under evidence gaps (routing-only)

| Field | Value |
|-------|--------|
| **Status** | **DRAFT** — do **not** freeze; do **not** emit `PREREG.sha256` |
| **Date** | 2026-09-23 PT |
| **Owner** | Kyle (ZetaZero) |
| **Parent** | T11.3 **CLOSED null (window)** — PREREG `27ab4a9573ec8c01ed043d72ef6c83ba47fdd4a764a373acb7453381bb9bd442` |
| **T11.2 (read-only)** | PREREG `593762c4c19e85d1720f6b46edbb93869258c4047ad023e42e144416aaed20d1` — **untouched** |
| **Artifact home** | `experiments/phase-routing/t12b/` |
| **Run location** | Linux `/workspace/scp` (scaffold + draft this turn; **no primary**) |
| **Sibling** | T12 loop budget — `../t12/` (parallel; not blocked by T12b) |
| **Scope** | **Routing-only** — no specialist execution |

> **DRAFT.** Numbers below lock **intent** for freeze review. Emitting a hash before Rule-19 gates + Kyle freeze greenlight voids the draft for that run.
>
> **Do not** retune C₃ / hum / Affinity `__all__`. **Do not** touch T11.2 / T11.3 PREREG hashes.
>
> **Amendment 2026-09-23 PT (Kyle + GNURadio):** Hold freeze until **in-gap perturbation** is locked (not only until HumState wire). Primary perturbation = single-frame spurious ScoreAffinity flip mid-gap. Document `one_step_memory` failure mode. G9 unchanged.
>
> **Amendment 2026-09-23 PT (Kyle + GNURadio, pre-freeze):** Add **`majority_k3`** as a **labeled secondary** arm (stateless majority over a short window of ScoreAffinity argmax, k=3). **Primary contrast stays `hum_measure` vs `one_step_memory`.** Without this secondary, a positive primary would force **T12b.2**; with it, a positive is interpretable as **smoothing vs held state**.

---

## 0. Why T12b exists

T11.3 closed model-level orchestration as **null (window)**: `hum_measure` beat the classifier on Ambiguous step-CSR, but **stateless ScoreAffinity argmax** scored higher — the lift was the observation window, not integrator state.

T12b asks the complementary routing-only question: when the affinity signal is **temporarily zeroed** (evidence gaps) or **delayed** (evidence lag), does **held hum state** keep correct-specialist routing where a stateless reader and a one-step memory baseline cannot?

Legitimate as routing-only; does **not** reopen T11.3 by retune.

---

## 1. Frozen inheritance (do not touch)

| Item | Rule |
|------|------|
| Hum dynamics (λ, a₀, coupling, thresholds) | **Frozen** — no retune |
| C₃ / ρ_eff / **T_hold(n=3)=65** | **Frozen** (cite `t11/imports/hum_measure/T11_N3_THOLD_RECEIPT.md`) |
| Affinity package `__all__` | **Untouched** |
| T11.2 `PREREG.sha256` | **Do not re-emit / do not edit** (`593762c4…`) |
| T11.3 `PREREG.sha256` | **Do not re-emit / do not edit** (`27ab4a95…`) |
| Sealed generators | **Reuse** via symlink `generators/sealed` → `../../t11_3/generators/sealed` — **no copy, no rewrite** |
| L2 provider | **ScoreAffinityProvider** (same as T11.3 primary / addendum) |

---

## 2. Arms

| Arm | Role | Definition |
|-----|------|------------|
| **`hum_measure`** | **Primary (counted)** | AffinityProvider + frozen **HumState / HumMeasureOrchestrator** (ScoreAffinity L2; amplitude gate; high-pass on drive if applicable). **Scaffold `HumMeasureArmStub` is dry-run only (G9) — must not be the counted arm** |
| **`stateless_argmax`** | Primary control | Per-step argmax of ScoreAffinity magnitudes over SPEC_ORDER — **no hum state** (see `t11_3/addendum_stateless_argmax.py`) |
| **`one_step_memory`** | **Primary null baseline** | Repeat last non-null decision; on first step / after reset with no prior, emit default (None / no specialist) |
| **`majority_k3`** | **Labeled secondary** (not primary contrast) | Stateless majority over a short window (**k=3**) of ScoreAffinity argmax labels. Zero-affinity frames do **not** vote. Ties → most recent winner. |

**Primary contrast (unchanged):** `hum_measure` vs `one_step_memory` (seed-mean CSR_gap; §4.2 null). `stateless_argmax` remains the instantaneous control.

**Why `majority_k3` before freeze:** Without a smoothing secondary, a **positive** primary (hum beats one-step) would be ambiguous between **held integrator state** and **any short-window smoothing of argmax**, forcing a **T12b.2** follow-on to separate those. With `majority_k3` preregistered as labeled secondary, a hum > one-step result that also beats (or ties) majority is interpretable as **held state vs smoothing**; hum ≤ majority with hum > one-step would read as smoothing-class, not integrator persistence.

Oracle remains report-only. Classifier / hand_rule are **out of primary arm set** for T12b.

---

## 3. Design — evidence gaps and lags (the one change)

### 3.1 Evidence gaps

- At **random start indices** inside each scored segment (after `evidence_point` when present; else full segment), **zero the per-step affinity** for a contiguous span of length **G ∈ {3, 6, 12, 24}** tokens.
- Gap cells are factorial over G (separate cells; same sealed text).
- **Assert (Rule 19):** during gap steps, affinity vector fed to every arm is **exactly zero** (L1/L∞ = 0) **except** at the locked in-gap perturbation frame (§3.1.1).

### 3.1.1 In-gap perturbation (locked — hold freeze until wired)

**Hold freeze** until this perturbation is locked and harness-wired — **not** only until HumState / HumMeasureOrchestrator is wired (G9 remains necessary but insufficient).

Bare silence with **G ≤ 24 ≪ T_hold(n=3)=65** makes hum survive **by construction** and **ties `one_step_memory`** whenever the correct label should not change across the gap. Lag alone does not separate them. Without a mid-gap lure that fools hold-last, the null is uninformative.

| Item | Lock |
|------|------|
| **Primary** | **Single-frame spurious ScoreAffinity flip mid-gap** — for exactly **one** frame inside the gap (prefer midpoint of the zero span), inject a **wrong specialist peak** (nonzero affinity peaked on an incorrect specialist), then **zeros resume** for the remainder of the gap. Outside that one frame, gap zeros still assert (G1). |
| **Alt (document only)** | Mid-gap **regime change in the scores** (brief non-zero regime vector inconsistent with pre-gap label) — same spirit; use only if freeze review demotes the single-frame flip. |
| **Applies to** | All primary gap cells G ∈ {3,6,12,24}; same sealed text / seeds as gap randomization. |
| **Freeze gate** | Do **not** emit `PREREG.sha256` until primary perturbation is implemented in `gap_inject` (or successor) and dry-run asserts one-frame non-zero + surrounding zeros. |

### 3.1.2 Expected `one_step_memory` failure mode

| Arm | Expected behavior under primary perturbation |
|-----|-----------------------------------------------|
| **`one_step_memory`** | **Hold-last** (or last non-null) **latches the spurious label / wrong regime** through the **rest of the gap** after the flipped frame — that is the informative null failure. |
| **`hum_measure`** | Should **resist** the single-frame lure via **amplitude / high-pass + integrator state** (frozen HumState / HumMeasureOrchestrator — G9). Must **not** permanently adopt the spurious specialist for the remaining gap steps (seed-mean rule locked at freeze). |
| **`stateless_argmax`** | Follows the flipped frame on that step; otherwise defaults on zeros — not the persistence null. |

Without §3.1.1–3.1.2, bare silence + lag cannot separate hum from one-step memory when the ground-truth label is constant across the gap.

### 3.2 Evidence lag

- Separately, delay the affinity stream by **L ∈ {2, 4, 8}** steps: at step t the arm sees affinity from step max(t−L, 0) (or zeros for t < L — lock exact pad rule at freeze; draft = **zero pad** for t < L).
- Lag cells do **not** combine with gap cells in the primary grid (orthogonal factors; report separately).

### 3.3 Randomization

- Gap start positions drawn from seed-derived RNG per (seed, sequence_id, seg_index, G).
- Draft seeds: same T11.3 family **`[20260923, 20260924, 20260925, 20260926, 20260927]`** unless freeze review overrides.

---

## 4. Hypotheses / null / stopping

### 4.1 Hypothesis

1. **Inside gaps with G < T_hold:** `hum_measure` CSR **stays near pre-gap** CSR; `stateless_argmax` falls toward default (near chance / None); thrash may spike at gap boundaries for stateless more than hum.
2. **Crossover:** as G approaches measured hold, hum advantage shrinks — crossover near **T_hold(n=3)=65** (see `docs/T12B_CROSSOVER_PREDICTION.md`, written **before** run).
3. **Lags:** hum CSR after lag L remains above stateless argmax of the lagged signal for L in the grid (qualitative; numeric delta locked at freeze if needed).
4. **Secondary (labeled):** `majority_k3` is reported alongside primary — **not** part of the §4.2 null. Interpretation: if hum beats one-step **and** beats majority → held state; if hum beats one-step but not majority → smoothing-class (would have forced T12b.2 without this arm).

### 4.2 Null (must include one-step baseline)

**Null:** Inside gaps **with the locked in-gap perturbation (§3.1.1)**, `hum_measure` CSR is **no better** than **`one_step_memory`** (seed-mean; interval rule locked at freeze — draft propose same absolute thrash-style **0.05** CSR points unless review changes).

If null holds, persistence under silence is explained by **repeating the last decision** (including latching the mid-gap spurious flip), not by hum dynamics beyond one step.

**Informative null requires** the §3.1.2 failure mode: `one_step_memory` latches the spurious label; hum should not. Bare silence alone is **not** an informative null when G ≪ T_hold and the true label is constant.

### 4.3 Stopping rule

- **One** preregistered primary run × **five** seeds after freeze.
- **No T12b.2** retune series for missing smoothing control — `majority_k3` is preregistered as labeled secondary so a positive primary is interpretable without a follow-on. Gate defect only → fix harness / re-freeze.
- Failures reported as **null / mixed / support** in this vocabulary.

---

## 5. Measures

| Symbol | Meaning |
|--------|---------|
| **CSR_gap** | Correct-specialist rate on steps **inside** gap spans |
| **CSR_pre** | CSR on matched pre-gap window (same length, immediately before gap; or locked alternative at freeze) |
| **CSR_lag** | CSR on steps after lag applied (full post-evidence set under lag transform) |
| **Thrash_gap** | Token-level decision changes **across** gap boundaries / inside gap (lock exact definition at freeze; draft: swap rate vs prior step) |
| **Default rate** | Fraction of gap steps with null / default prediction (expected high for stateless) |

Primary support comparisons (draft):

1. CSR_gap(hum) ≈ CSR_pre(hum) for all G ∈ {3,6,12,24} (all ≪ 65) — see crossover doc.
2. CSR_gap(hum) − CSR_gap(stateless) > seed interval.
3. CSR_gap(hum) − CSR_gap(one_step_memory) > seed interval (**null fail ⇒ support**).

**Labeled secondary (report; not null):** CSR_gap(hum) vs CSR_gap(`majority_k3`) — smoothing vs held-state interpretation when primary is positive.

---

## 6. Seeds / batch / stopping disclosure

| Item | Draft lock |
|------|------------|
| Seeds | `[20260923, 20260924, 20260925, 20260926, 20260927]` |
| Batch | `generators/sealed` (t11_3 sealed v0) |
| Grain | Step-level routing decisions |
| Specialist exec | **OFF** (routing-only) |
| Seed interval (CSR) | **0.05** absolute (draft; confirm at freeze) |

---

## 7. Rule 19 gates (own — listed)

See `docs/T12B_RULE19_GATES.md`. Summary:

1. **Gaps zero affinity** — assert / measured on injected traces.
2. **`one_step_memory` arm present** in ARMS / primary config.
2b. **`majority_k3` labeled secondary present** (k=3 ScoreAffinity argmax majority) — not primary contrast; documents smoothing vs held-state without forcing T12b.2.
3. **Crossover prediction written before run** — `docs/T12B_CROSSOVER_PREDICTION.md` hashed into freeze inventory.
4. **No C₃ / hum / Affinity retune.**
5. **T11.2 / T11.3 PREREG hashes untouched.**
6. **G9 — Stub ≠ counted arm** (GNURadio 2026-09-23): freeze/primary refuse if counted `hum_measure` is still `HumMeasureArmStub` (hold-last). Wire frozen HumState / HumMeasureOrchestrator (amplitude gate; high-pass on drive if applicable). Dry-run stub OK; counted stub makes gap null uninformative by construction. **G9 stays.**
7. **G10 — In-gap perturbation locked** (2026-09-23): freeze held until primary **single-frame spurious ScoreAffinity flip mid-gap** is wired + asserted; `one_step_memory` expected to latch spurious label (§3.1.1–3.1.2). HumState wire alone does **not** unlock freeze.

---

## 8. Explicit non-goals this draft

- Do **not** run full primary while status is DRAFT.
- Do **not** emit `PREREG.sha256` for T12b while DRAFT.
- Do **not** retune C₃, hum, or Affinity exports.
- Do **not** execute specialists or claim H5.
- Do **not** edit T11.3 sealed files or PREREG.

---

## 9. Hold-time citation (crossover)

| Source | Value | Use in T12b |
|--------|-------|-------------|
| T11 C₃ **T_hold(n=3)** coupled | **65** | **Primary crossover reference** (frozen inheritance) |
| T8 MoE top-1 silence hold | ~25 | Different measure / n — **not** T12b operating hold |
| T8/T11 homogeneous 1/e | ~39–41 | Diagnostic only |

**Hold status:** Documented (65). Gap grid max G=24 ≪ 65 → pre-run prediction: **all gap cells remain below crossover** (hum should hold under bare silence). Extending G toward 65 is **out of this draft grid** unless Kyle adds a follow-on cell at freeze.

**Freeze hold (amendment):** Do **not** freeze on HumState wire alone. With G≪65, bare silence + hold-last **ties** hum and `one_step_memory` when the label should not change; **in-gap perturbation (§3.1.1)** is required to make the null informative. Lag cells alone do not substitute.

---

## 10. Artifact pointers

| Doc | Path |
|-----|------|
| Rule 19 | `docs/T12B_RULE19_GATES.md` |
| Crossover (pre-run) | `docs/T12B_CROSSOVER_PREDICTION.md` |
| Scripts | `scripts/gap_inject.py`, `scripts/arms.py` (incl. `majority_k3` secondary), `scripts/primary_run.py`, `scripts/rule19_freeze_gates.py`, `scripts/freeze_prereg.py` |
| Sealed pointer | `generators/SEALED_POINTER.md` |
