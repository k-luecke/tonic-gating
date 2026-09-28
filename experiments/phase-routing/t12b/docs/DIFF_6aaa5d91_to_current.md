# T12b PREREG draft diff — `6aaa5d91…` → `408eaaac…`

| Field | Value |
|-------|--------|
| **PT** | 2026-09-23 19:47 PT |
| **Prior draft content sha256** | `6aaa5d9171dff4534650dbcc997096c205efc4480b2bd202e2e5d152e92cee4a` |
| **New draft content sha256** | `408eaaaca59d0c210e1b1ef50d2494213662da1059b3542c38ce866651801b21` |
| **Status** | Still **DRAFT** — no `PREREG.sha256` emitted |
| **Rule 19** | PASS — `results/T12B_RULE19_GATES.{json,md}` |
| **Affinity / C₃ / hum** | UNTOUCHED (`7dae877a…`) |
| **Parent T11.2 / T11.3 PREREG** | untouched |

## Why this amend (Kyle + GNURadio, pre-freeze)

Add **`majority_k3`** as a **labeled secondary** arm before freeze:

- Stateless majority over a short window (**k=3**) of ScoreAffinity argmax.
- **Primary contrast unchanged:** `hum_measure` vs `one_step_memory`.
- **Without** this arm, a positive primary would force **T12b.2** to separate smoothing vs held state.
- **With** it, a positive is interpretable as **smoothing vs held state**.

Also: implement Rule 19 runner (G1–G11 incl. G9 HumState counted path + G10 mid-gap lure + G11 majority_k3) and freeze entry that **refuses emit** while DRAFT.

## Material draft changes

1. **Arms table** — add `majority_k3` row (labeled secondary; not primary contrast).
2. **Rationale block** — document T12b.2-avoidance / smoothing-vs-held interpretation.
3. **§4.1 hypothesis** — secondary interpretation clause (report-only; not part of §4.2 null).
4. **§4.3 stopping** — no T12b.2 retune series for missing smoothing control.
5. **§5 measures** — labeled secondary CSR_gap(hum) vs CSR_gap(majority_k3).
6. **Rule 19 list** — G11 / 2b majority_k3 presence.
7. **Artifact pointers** — `rule19_freeze_gates.py`, `freeze_prereg.py`, `majority_k3` in arms.

## Non-changes

- Primary contrast still hum vs one_step_memory.
- Gap grid / lag grid / seeds / T_hold=65 / G10 mid-gap lure lock unchanged in intent.
- Affinity / C₃ / hum / T11.2 / T11.3 PREREG untouched.
- No primary run; no freeze emit.

## Unified diff (content)

```diff
--- T12B_PREREG_DRAFT.md@6aaa5d91
+++ T12B_PREREG_DRAFT.md@408eaaac
@@ -17,6 +17,8 @@
 > **Do not** retune C₃ / hum / Affinity `__all__`. **Do not** touch T11.2 / T11.3 PREREG hashes.
 >
 > **Amendment 2026-09-23 PT (Kyle + GNURadio):** Hold freeze until **in-gap perturbation** is locked (not only until HumState wire). Primary perturbation = single-frame spurious ScoreAffinity flip mid-gap. Document `one_step_memory` failure mode. G9 unchanged.
+>
+> **Amendment 2026-09-23 PT (Kyle + GNURadio, pre-freeze):** Add **`majority_k3`** as a **labeled secondary** arm (stateless majority over a short window of ScoreAffinity argmax, k=3). **Primary contrast stays `hum_measure` vs `one_step_memory`.** Without this secondary, a positive primary would force **T12b.2**; with it, a positive is interpretable as **smoothing vs held state**.
 
 ---
 
@@ -46,13 +48,18 @@
 
 ## 2. Arms
 
-| Arm | Definition |
-|-----|------------|
-| **`hum_measure`** | AffinityProvider + frozen **HumState / HumMeasureOrchestrator** (ScoreAffinity L2; amplitude gate; high-pass on drive if applicable). **Scaffold `HumMeasureArmStub` is dry-run only (G9) — must not be the counted arm** |
-| **`stateless_argmax`** | Per-step argmax of ScoreAffinity magnitudes over SPEC_ORDER — **no hum state** (see `t11_3/addendum_stateless_argmax.py`) |
-| **`one_step_memory`** | **Required null baseline** — repeat last non-null decision; on first step / after reset with no prior, emit default (None / no specialist) |
-
-Oracle remains report-only. Classifier / hand_rule are **out of primary arm set** for T12b (question is persistence vs instantaneous / one-step controls under gaps).
+| Arm | Role | Definition |
+|-----|------|------------|
+| **`hum_measure`** | **Primary (counted)** | AffinityProvider + frozen **HumState / HumMeasureOrchestrator** (ScoreAffinity L2; amplitude gate; high-pass on drive if applicable). **Scaffold `HumMeasureArmStub` is dry-run only (G9) — must not be the counted arm** |
+| **`stateless_argmax`** | Primary control | Per-step argmax of ScoreAffinity magnitudes over SPEC_ORDER — **no hum state** (see `t11_3/addendum_stateless_argmax.py`) |
+| **`one_step_memory`** | **Primary null baseline** | Repeat last non-null decision; on first step / after reset with no prior, emit default (None / no specialist) |
+| **`majority_k3`** | **Labeled secondary** (not primary contrast) | Stateless majority over a short window (**k=3**) of ScoreAffinity argmax labels. Zero-affinity frames do **not** vote. Ties → most recent winner. |
+
+**Primary contrast (unchanged):** `hum_measure` vs `one_step_memory` (seed-mean CSR_gap; §4.2 null). `stateless_argmax` remains the instantaneous control.
+
+**Why `majority_k3` before freeze:** Without a smoothing secondary, a **positive** primary (hum beats one-step) would be ambiguous between **held integrator state** and **any short-window smoothing of argmax**, forcing a **T12b.2** follow-on to separate those. With `majority_k3` preregistered as labeled secondary, a hum > one-step result that also beats (or ties) majority is interpretable as **held state vs smoothing**; hum ≤ majority with hum > one-step would read as smoothing-class, not integrator persistence.
+
+Oracle remains report-only. Classifier / hand_rule are **out of primary arm set** for T12b.
 
 ---
 
@@ -106,6 +113,7 @@
 1. **Inside gaps with G < T_hold:** `hum_measure` CSR **stays near pre-gap** CSR; `stateless_argmax` falls toward default (near chance / None); thrash may spike at gap boundaries for stateless more than hum.
 2. **Crossover:** as G approaches measured hold, hum advantage shrinks — crossover near **T_hold(n=3)=65** (see `docs/T12B_CROSSOVER_PREDICTION.md`, written **before** run).
 3. **Lags:** hum CSR after lag L remains above stateless argmax of the lagged signal for L in the grid (qualitative; numeric delta locked at freeze if needed).
+4. **Secondary (labeled):** `majority_k3` is reported alongside primary — **not** part of the §4.2 null. Interpretation: if hum beats one-step **and** beats majority → held state; if hum beats one-step but not majority → smoothing-class (would have forced T12b.2 without this arm).
 
 ### 4.2 Null (must include one-step baseline)
 
@@ -118,7 +126,7 @@
 ### 4.3 Stopping rule
 
 - **One** preregistered primary run × **five** seeds after freeze.
-- **No T12b.2** retune series. Gate defect only → fix harness / re-freeze.
+- **No T12b.2** retune series for missing smoothing control — `majority_k3` is preregistered as labeled secondary so a positive primary is interpretable without a follow-on. Gate defect only → fix harness / re-freeze.
 - Failures reported as **null / mixed / support** in this vocabulary.
 
 ---
@@ -139,6 +147,8 @@
 2. CSR_gap(hum) − CSR_gap(stateless) > seed interval.
 3. CSR_gap(hum) − CSR_gap(one_step_memory) > seed interval (**null fail ⇒ support**).
 
+**Labeled secondary (report; not null):** CSR_gap(hum) vs CSR_gap(`majority_k3`) — smoothing vs held-state interpretation when primary is positive.
+
 ---
 
 ## 6. Seeds / batch / stopping disclosure
@@ -159,6 +169,7 @@
 
 1. **Gaps zero affinity** — assert / measured on injected traces.
 2. **`one_step_memory` arm present** in ARMS / primary config.
+2b. **`majority_k3` labeled secondary present** (k=3 ScoreAffinity argmax majority) — not primary contrast; documents smoothing vs held-state without forcing T12b.2.
 3. **Crossover prediction written before run** — `docs/T12B_CROSSOVER_PREDICTION.md` hashed into freeze inventory.
 4. **No C₃ / hum / Affinity retune.**
 5. **T11.2 / T11.3 PREREG hashes untouched.**
@@ -197,5 +208,5 @@
 |-----|------|
 | Rule 19 | `docs/T12B_RULE19_GATES.md` |
 | Crossover (pre-run) | `docs/T12B_CROSSOVER_PREDICTION.md` |
-| Scripts | `scripts/gap_inject.py`, `scripts/arms.py`, `scripts/primary_run.py` |
+| Scripts | `scripts/gap_inject.py`, `scripts/arms.py` (incl. `majority_k3` secondary), `scripts/primary_run.py`, `scripts/rule19_freeze_gates.py`, `scripts/freeze_prereg.py` |
 | Sealed pointer | `generators/SEALED_POINTER.md` |
```
