# T12b crossover prediction (written BEFORE primary run)

| Field | Value |
|-------|--------|
| **Status** | **PRE-RUN LOCK** — Rule 19 G3 |
| **Written** | 2026-09-23 PT (scaffold; before any T12b primary) |
| **Hold source** | T11 C₃ **T_hold(n=3)=65** (coupled) — `t11/imports/hum_measure/T11_N3_THOLD_RECEIPT.md` |
| **Gap grid** | G ∈ {3, 6, 12, 24} |
| **Lag grid** | L ∈ {2, 4, 8} (orthogonal; not used for crossover vs hold) |

## Prediction (plain)

1. **Operating hold** for hum state under affinity silence is **T_hold(n=3) = 65** tokens (frozen C₃; not T8 MoE ~25).
2. Every preregistered gap length **G ≤ 24 ≪ 65**, so **all gap cells sit below the expected crossover**.
3. Therefore, **before seeing data**, we predict:
   - **CSR_gap(hum_measure) ≈ CSR_pre(hum_measure)** for each G ∈ {3,6,12,24} (near pre-gap; within draft seed interval 0.05).
   - **CSR_gap(stateless_argmax)** collapses toward default / chance inside gaps (affinity zero → no instantaneous specialist).
   - **CSR_gap(hum_measure) > CSR_gap(one_step_memory)** if hum dynamics beat mere repeat-last; if not, **null**.
4. **Crossover** (hum advantage vs stateless vanishing as G grows) is predicted near **G ≈ 65**, **outside this draft grid**. Observing a crossover inside {3,6,12,24} would **contradict** this pre-run prediction and must be reported as such (not retuned away).

## What would falsify the pre-run story

| Observation | Interpretation |
|-------------|----------------|
| Hum CSR_gap falls to default already at G=3–12 | Effective hold ≪ 65 under ScoreAffinity-routing silence (dynamics or readout mismatch) |
| Hum ≤ one_step_memory inside gaps | **Null** — no credit beyond one-step memory |
| Stateless stays high inside gaps | **G1 fail** or gap injection not actually zeroing affinity |

## Non-claims

- Does not claim T8 MoE T_hold (~25) is the crossover; that is a different instrument.
- Does not authorize extending G to 65 in this ticket without freeze amendment.
- Does not authorize C₃ retune if crossover appears early.

## Hash note

This file must be present and unchanged in intent before primary. Freeze inventory shall record its sha256 beside the DRAFT→FROZEN transition.
