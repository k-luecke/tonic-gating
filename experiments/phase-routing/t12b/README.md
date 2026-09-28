# T12b — Persistence under evidence gaps (routing-only)

| Field | Value |
|-------|--------|
| **Status** | **FROZEN + primary complete** — PREREG emitted; Rule-19 PASS; primary × 5 seeds under frozen PREREG |
| **Date** | 2026-09-23 PT |
| **Owner** | Kyle (ZetaZero) / Drift Bound Labs |
| **Folder** | `/workspace/scp/experiments/phase-routing/t12b/` |
| **Parent** | T11.3 **CLOSED null (window)** — PREREG `27ab4a9573ec8c01ed043d72ef6c83ba47fdd4a764a373acb7453381bb9bd442` |
| **Sibling** | T12 loop budget — [`../t12/`](../t12/) (parallel; leave alone) |
| **Greenlit** | Kyle greenlit T12b in parallel with T12 (handoff §7) |

## Mission

Ask whether **hum integrator state** persists through **evidence gaps** and **evidence lag** when the per-step affinity signal is dropped or delayed — using the **same** T11.3 ScoreAffinity harness and sealed batch, **routing-only** (no specialist execution).

## Hypothesis / null

| Claim | Statement |
|-------|-----------|
| **H (gaps)** | Inside gaps shorter than measured hold, **hum_measure CSR stays near pre-gap**; **stateless_argmax** falls to default; crossover near measured hold. |
| **H (lags)** | After affinity lag of {2, 4, 8} steps, hum retains useful state vs instantaneous argmax of lagged signal. |
| **Null** | Hum CSR **inside gaps with in-gap perturbation** is **no better** than **one-step memory** (which should latch the mid-gap spurious flip). Bare silence alone is uninformative when G ≪ T_hold. |

## Relation to T11.3 + T12

| Line | Role |
|------|------|
| **T11.3** | Closed model-level orchestration as **null (window)** — ScoreAffinity window, not hum state, drove Ambiguous CSR. Sealed generators + `addendum_stateless_argmax.py` reused read-only. |
| **T12** | Loop-budget seat (recurrent depth). Parallel; T12b does **not** block T12. |
| **T12b** | Routing-only persistence under **silenced / lagged affinity** — closes the evidence-gap question T11.3 left open. |

## Design (one change vs T11.3)

- Same harness spirit as T11.3 / addendum (ScoreAffinity L2).
- Same sealed generators: `generators/sealed` → symlink to `../../t11_3/generators/sealed` (**do not copy**).
- Same frozen hum and C₃ (`imports/hum_measure` → `/workspace/scp/t11/imports/hum_measure`).
- Same **stateless_argmax** control (see `../t11_3/addendum_stateless_argmax.py`).
- **ONE change:** per-step affinity **dropped** for spans of length **{3, 6, 12, 24}** at random points inside segments (**evidence gaps**), and separately **delayed** by **{2, 4, 8}** steps (**evidence lag**).
- **In-gap perturbation (freeze lock):** **single-frame spurious ScoreAffinity flip mid-gap** (wrong specialist peak one frame, then zeros). Alt: mid-gap regime change in scores. Makes `one_step_memory` latch the lure; hum should resist. **Hold freeze** until this is wired — not only until HumState wire.
- Arms: `hum_measure`, `stateless_argmax`, **`one_step_memory`** (required null baseline); **`majority_k3`** labeled secondary (stateless k=3 ScoreAffinity argmax majority — smoothing vs held-state; avoids forcing T12b.2).

## Measures

- **CSR inside gaps** (stateless has nothing; hum has held state)
- **CSR after lags**
- **Thrash across gaps**

## Out of scope

- Specialist execution / SCP receipts / H5 typed-goal
- Retuning C₃ / hum / Affinity `__all__`
- Editing T11.2 PREREG `593762c4…` or T11.3 PREREG `27ab4a95…`
- Loop-budget model work (that is T12)

## Hard locks

1. Do **not** retune C₃ / hum / Affinity `__all__`.
2. Do **not** touch T11.2 / T11.3 PREREG hashes (read-only reuse).
3. Own Rule 19: gaps zero affinity except G10 perturbation frame; one-step-memory present; crossover prediction written **before** run; **G10 in-gap perturbation locked before freeze** (G9 stays).
4. No full primary until freeze + greenlight path.

## Tree

```text
experiments/phase-routing/t12b/
  README.md
  ARTIFACT_INVENTORY.md
  docs/T12B_PREREG_DRAFT.md
  docs/T12B_RULE19_GATES.md
  docs/T12B_CROSSOVER_PREDICTION.md   ← written pre-run (Rule 19)
  generators/sealed → ../../t11_3/generators/sealed
  generators/SEALED_POINTER.md
  imports/hum_measure → /workspace/scp/t11/imports/hum_measure
  imports/PATHS.md
  scripts/gap_inject.py
  scripts/arms.py
  scripts/primary_run.py              ← --dry-run / importable; primary stubbed
  scripts/freeze_prereg.py            ← Rule 19 entry; do not emit hash until Kyle authorize
  scripts/rule19_freeze_gates.py     ← G1–G11 (G9 HumState, G10 lure, G11 majority_k3)
  results/
```

## Sealed batch (reuse, no copy)

See `generators/SEALED_POINTER.md`. Symlink target:

`experiments/phase-routing/t11_3/generators/sealed/`

Manifest hash (ALL_SEQUENCES): `646df30bbeacdc77432e9cb1daa366640ca0cfeb2e01216d4cb5f03a9df404c6`

## Quick smoke

```bash
cd /workspace/scp
python3 experiments/phase-routing/t12b/scripts/gap_inject.py --self-test
python3 experiments/phase-routing/t12b/scripts/primary_run.py --dry-run
# expect ok=true; hum_measure_class=HumMeasureArm; G10 flip_nonzero; HP every drive
```

## Docs

- PREREG draft: [`docs/T12B_PREREG_DRAFT.md`](docs/T12B_PREREG_DRAFT.md) — **Status DRAFT**
- Rule 19 gates: [`docs/T12B_RULE19_GATES.md`](docs/T12B_RULE19_GATES.md)
- Crossover prediction (pre-run): [`docs/T12B_CROSSOVER_PREDICTION.md`](docs/T12B_CROSSOVER_PREDICTION.md)

## Gate G9 (before freeze / primary) — stays

`HumMeasureArmStub` in `scripts/arms.py` is **dry-run only**. Inside gaps it hold-lasts like `one_step_memory`, so a counted stub makes the null uninformative. Primary must wire frozen `HumState` / `HumMeasureOrchestrator` (amplitude gate; high-pass on drive if applicable). Affinity/C₃ stay frozen.

## Gate G10 + freeze hold (2026-09-23 amendment)

**Hold freeze** until **in-gap perturbation** is locked — not only until HumState wire.

- **Primary:** single-frame spurious ScoreAffinity flip mid-gap (wrong specialist peak for one frame, then zeros resume).
- **Alt:** mid-gap regime change in the scores.
- **Expected `one_step_memory` failure:** hold-last (or last non-null) latches the spurious label through the rest of the gap; `hum_measure` should not (amplitude/HP + state).
- Without G10, G≤24 ≪ T_hold(n=3)=65 makes bare silence let hum survive by construction and tie one_step when the label shouldn’t change; lag alone doesn’t separate them.

See [`docs/T12B_PREREG_DRAFT.md`](docs/T12B_PREREG_DRAFT.md) §3.1.1–3.1.2 and [`docs/T12B_RULE19_GATES.md`](docs/T12B_RULE19_GATES.md) G10.

## G9 HumState wire status (2026-09-23 PT)

| Item | Status |
|------|--------|
| Primary counted arm | `HumMeasureArm` — frozen `HumState` (t11/hum) + `ScoreAffinityProvider` + **HP every drive** |
| Stub | `HumMeasureArmStub` dry-run only; `arm_factory(mode='primary'|'freeze')` **refuses** stub (G9) |
| Orchestrator | `HumMeasureOrchestrator` import-checked from `t11/routing` (segment API); per-step path uses HumState directly |
| Linux | Pure numpy dynamics — **no Windows-only blocker** on step/HP path |
| G10 | `inject_gap_with_g10` mid-gap single-frame spurious flip; dry-run asserts |
| Freeze | **Still held** — G10+G9 scaffolded but PREREG remains DRAFT (no hash emit) |
| Affinity/C₃ | **Untouched** |

Dry-run receipt: `results/DRY_RUN.json`.
