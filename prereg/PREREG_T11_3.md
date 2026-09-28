# PREREG — T11.3 Last Test of Model-Level Orchestration Claim

| Field | Value |
|-------|--------|
| **Status** | **FROZEN** 2026-09-23 17:13 PT — Rule-19 T11.3 PASS; emit only via freeze_prereg |
| **Date** | 2026-09-23 PT |
| **Parent** | T11.2 CLOSED — PREREG.sha256 `593762c4c19e85d1720f6b46edbb93869258c4047ad023e42e144416aaed20d1` frozen |
| **Scope** | **Last test** of the model-level orchestration claim — **not** an open series |
| **Run location** | Linux `/workspace/scp` (docs only this turn; no primary/generators yet) |
| **Artifact home (proposed)** | `experiments/phase-routing/t11_3/` |

> **FROZEN.** Rule-19 T11.3 gates (incl §5.1 affinity measurement) PASS. Hum/C₃/Affinity `__all__` untouched. T11.2 PREREG `593762c4…` not re-emitted.

---

## 0. Why T11.3 exists (one ticket)

T11.2 measured **persistence costs** at segment grain where the regime is readable (H3 supported with the bill; H4 null). It did **not** informatively test late integration (H1 untested — harness scores before evidence), token-level thrash (H2 uninformative), or H5. Benefits if any live **strictly below the segment**.

T11.3 is the **last** orchestration test: move decisions and scoring to the grain where the coordinator could theoretically win. If it still does not beat a classifier under the stopping rule, the model-level orchestration claim **closes as null** and the line moves to **T12** (looped-model seats). No further T11.x reopen by retune.

---

## 1. Frozen inheritance (do not touch)

| Item | Rule |
|------|------|
| Hum dynamics (λ, a₀, coupling, thresholds) | **Frozen** — no retune |
| C₃ / ρ_eff / T_hold(n=3)=65 | **Frozen** (same rebuilt C₃) |
| Affinity package `__all__` | **Untouched** |
| T11.2 `PREREG.sha256` | **Do not re-emit** |
| ScoreAffinityProvider | **Locked L2** wiring (§5.0); seal gate is §5.1 **measured** ≥90% argmax on sealed Ambiguous — **not** a retune |

---

## 2. Arms

| Arm | Status |
|-----|--------|
| `hand_rule` | keep |
| `classifier_mlp` (context classifier) | keep |
| `hum` | keep |
| `hum_measure` (AffinityProvider HumMeasureOrchestrator) | keep if needed for provider-parity disclosure; not H5 |
| **`classifier_stub`** | **DROP** — bit-identical to `hand_rule` on T11.2 batch |
| H5 path | typed-goal hum vs inferred hum (see §4) — not a fifth “mystery” arm |

Oracle ceiling remains report-only / not compared for support.

---

## 3. Design changes vs T11.2 (required)

### 3.1 Decisions and scoring — per step within a segment

- Orchestrator **decides at every action step** inside a segment (not once at segment open).
- **CSR (M1)** scored at every action step **at or after the late-evidence point**.
  - After evidence: an integrator **can** be right; a first-32 / first-action reader **cannot** (by construction of Ambiguous).
- Segment-open-only scoring (T11.2 failure mode for H1) is **forbidden** as the primary H1 measure.

### 3.2 Token-level thrash (M2)

- Log **token-level thrash** for **every arm** on the Distraction split.
- Segment-grain thrash alone is **not** informative (T11.2: swaps present, thrash 0 at segment grain; spans inside segments).

### 3.3 H5 run properly

- **Typed-goal hum** vs **inferred hum** on **Ambiguous**, with **specialists executing**.
- **Invalid-action rate (M5)** at **matched coverage**.
- Scored **separately** from H1–H4; cannot rescue an H1–H4 null.
- Note: T11.2 `hum_measure` ≠ H5.

### 3.4 Primary receipts

- Primary receipts carry **tokens + raw outputs** (as gate-4 live path already does / must do).
- Enough to audit step-grain decisions and post-evidence scoring.

### 3.5 Drop `classifier_stub`

- Removed from the arm set going forward (duplicate of `hand_rule` on T11.2).

---

## 4. Hypotheses (draft — same claim, finer grain)

| H | Primary measure | Support sketch (lock at freeze) |
|---|-----------------|----------------------------------|
| **H1** | Ambiguous CSR at **step grain**, steps ≥ evidence point | hum ≥ classifier; hum > hand |
| **H2** | **Token-level** thrash on Distraction | hum lower than classifier and hand by **more than the seed interval** |
| **H3** | Switch latency on genuine regime change (step or segment units as locked) | \|lat_hum − lat_classifier\| ≤ preregistered bound (carry T11.2 “within two segments” spirit at the new grain) |
| **H4** | Renewal recall (regime-A returns; report trailing unrelated) | report; support rule locked before freeze |
| **H5** | Typed-goal vs inferred on Ambiguous + M5 at matched coverage | separate combo; cannot rescue H1–H4 |

Clean remains the readable-regime sanity / cost bill (lookups expected to win when first tokens suffice).

---

## 5. Observation window (wiring) + Rule-19 affinity gate (measured)

Same frozen C₃ / hum; wider observation window only — **not a retune**. State beside the §6 stopping rule at freeze.

### 5.0 Observation wiring (disclosure — not the seal gate)

| Lock | Value |
|------|--------|
| **L2 provider** | **`ScoreAffinityProvider`** (or equivalent **full-segment / per-step scores**) |
| **Not allowed as primary L2** | `KeywordAffinityProvider` **first-32** (late tokens past 32 invisible) |
| **C₃ / hum** | **Same frozen C₃**; hum dynamics untouched |
| **What changes** | Observation window only — **not a retune** |

A property of the **code** (which provider is wired) is a promise. It is **not** the Rule-19 gate.

### 5.1 Rule 19 — affinity seal gate (measured on sealed Ambiguous)

**Gate = a property of the data the freeze script can measure**, not a description of the provider.

The freeze script **runs this test on the sealed Ambiguous split** before emitting `PREREG.sha256` (same spirit as T11.2 Rule 19 refusing a hollow Ambiguous):

1. For each Ambiguous segment, take the **per-step affinity trace actually fed to the hum** (the scores/vectors the coordinator consumed — not a parallel gate-1 classifier).
2. Count a segment as PASS if at some step **≥ the evidence point**, **argmax(trace) = oracle specialist**.
3. **Pass bar:** PASS on **≥ 90%** of Ambiguous segments.
4. **Fail → do not seal.** If the affinity signal never reaches the oracle post-evidence on enough segments, no orchestrator downstream can, and the run must not seal.

| Field | Value |
|-------|--------|
| Input | Sealed Ambiguous segments + recorded per-step affinity traces |
| Predicate | ∃ step ≥ evidence_point: argmax(affinity_trace[step]) = oracle |
| Threshold | ≥ 90% of Ambiguous segments |
| Fail action | Freeze script prints FAIL; blocks `PREREG.sha256` |

Keyword first-32 wiring may still be refused as a separate harness check; the **≥90% argmax test is the affinity gate**.

---

## 6. Stopping rule (prereg — non-negotiable)

If, on the frozen T11.3 primary:

1. **hum does not beat classifier on Ambiguous CSR at step grain**, **OR**
2. **hum does not beat classifier on token-level thrash by more than the seed interval**,

then the **model-level orchestration claim closes as null** and the line moves to **T12**.

No post-hoc retune of hum / C₃ / Affinity `__all__` to reopen. H5 remains a separate combo note either way. Publishable persistence-cost finding from T11.2 stands.

---

## 7. Thresholds / seeds (LOCKED at freeze)

- Seeds: `[20260923, 20260924, 20260925, 20260926, 20260927]` (eval=20260923; train=[20260924..26]; val=20260927 — same T11.2 split).
- Seed interval for thrash (H2): **0.05** absolute rate (lock).
- Evidence-point: **token index** per Ambiguous segment field `evidence_point` (see `generators/sealed/EVIDENCE_POINTS.json`); first family-keyword at/after token 32 by construction.
- Affinity seal bar (§5.1): **≥ 0.90** Ambiguous segments.
- λ / ρ_eff / T_hold: **inherit T11.2 frozen** — cite only, do not edit.
- L2 provider: **ScoreAffinityProvider** (observation window only; not a retune).

---

## 8. Explicit non-goals this draft

- Primary / generators authorized only after this freeze hash.
- `PREREG.sha256` emitted only via `freeze_prereg.py --ticket t11.3` after Rule-19 PASS.
- Do **not** retune C₃, hum, or Affinity package exports.
- Do **not** treat T11.3 as the start of an open T11.x series — it is the **last** orchestration claim test.

---

## 9. T11.2 close pointer

See `t11/docs/T11_2_CLOSING.md` / `experiments/phase-routing/t11_2/T11_2_CLOSING.md`. Honest sentence and publishable persistence-cost claim carry forward; T11.3 only asks whether benefits appear below the segment under a locked ScoreAffinity observation window with the §5.1 affinity-argmax seal gate (≥90% Ambiguous) so the input is verified whichever way the stopping rule lands.

---

## 10. LoopGNU wire note (2026-09-23 PT)

Rule-19 §5.1 mechanical check wired at `t11/scripts/rule19_t11_3_gates.py` (doc: `t11/docs/T11_3_RULE19_GATES.md`). Smoke: `--smoke-hollow` / `--smoke-keyword` refuse hollow affinity. T11.2 gates remain in `t11/scripts/rule19_freeze_gates.py`. C₃/hum untouched; T11.2 `PREREG` `593762c4…` not re-emitted.

