# CLOSING — T12 The Loop Budget (T7–T8)

| Field | Value |
|-------|--------|
| **Date** | 2026-09-23 19:38 PT (amend 2026-09-23 19:43 PT; H3-T wording 2026-09-23 19:59 PT) |
| **Status** | **CLOSED** — T0–T8 complete; one primary × 5 seeds; **no T12.2** |
| **Overall verdict** | **mixed** — H3 **support**; H1/H2/H4 **null**; §1.3 null clauses **(1)** and **(3)** fire (mission claim not supported) |
| **PREREG.sha256** | `bf757ee4fc01e5c40b45f035150e7b716d12f7d8b30d745b9a7669f79bbf307c` |
| **T6 receipts** | `results/t6/T6_PRIMARY_AGGREGATE.{json,md}` + `T6_SEED_20260921..20260925.json` |
| **Parent** | T11.3 CLOSED null (window) — `27ab4a9573ec8c01ed043d72ef6c83ba47fdd4a764a373acb7453381bb9bd442` |
| **Affinity / C₃ / hum** | **UNTOUCHED** (Affinity sha `7dae877a3975c1ba4ca264f03f572d6749cce39a114ea3adbcffc010c7852c50`) |
| **Gate defect** | **false** → no T12.2 |
| **t12b** | untouched |

## Per-H verdicts (PREREG §1.2–1.3)

| H | Verdict | Evidence (T6 aggregate) |
|---|---------|-------------------------|
| **H1** Regime budgeting | **null** | **ACT = oracle** (ΔACC 0; ACC̄=LPC̄ identical): coordinator matched oracle; ACT matched coordinator → per-item convergence already at oracle ceiling, so regime budgeting had no ACC/LPC headroom. vs Fixed ΔACC=+0.0008 ΔLPC=+0.0001 (noise). §1.3 clause **(1)** fires. BP separates easy=4 / hard=10 gap=6 (5/5) but does not rescue primary claim. |
| **H2** Trajectory budgeting | **null** | `coordinator_regime_traj` vs ACT ΔACC=+0.0000 ΔLPC=+0.0000; vs CONV identical. Traj does **not** beat ACT/Convergence under same T_max. |
| **H3** Overthinking regime-dependent | **support** | OT(T) Fixed sweep: easy **degrades past T\*** **5/5** seeds; hard **0/5**. Effect size (seed-mean): easy **ΔACC(T=12 vs T\*=4)=−0.3418**, **ΔNLL=+1.5982** — i.e. easy degrades **−0.3418 at T=12 vs T\*=4**. Between **T\*=4** and cost-matched Fixed **T=7**, easy loss is **~0** (seed-mean ΔACC(T=7−T\*=4)≈−0.0016; Fixed arm within **0.0008** ACC of coordinator). hard ΔACC=0 / ΔNLL=0 (T\*=12). Support is the directional count + nonzero easy magnitude at T=12 — not a large mission lift. T\*_easy=4 ≠ T\*_hard=12 (§1.3 clause (2) does **not** fire). |
| **H4** Persistence under silencing | **null** | SD ΔACC mean=+0.0000; `coord_beats_stateless_acc_n`=**0/5**. Under operational silencing, coordinator is **no better** than Stateless regime. §1.3 clause **(3)** fires. |

## Null clauses applied

1. **Fired** — Fixed and/or ACT match coordinator on primary ACC/LPC comparisons (**ACT = oracle**, ΔACC 0; Fixed Δ≈0).
2. **Not fired** — Optimal T\* differs across regimes (4 vs 12 on Fixed sweep OT).
3. **Fired** — SD shows coordinator no better than Stateless (0/5).

Any fired clause → mission claim (persistent coordinator loop-budget job that lookup cannot do) **not supported**. H3 OT pattern remains a clean descriptive support within that null mission outcome → series close vocabulary = **mixed**.

## Aggregate snapshot (cite T6)

| Arm | ACC̄ | C̄ | LPC̄ | T_b easy/hard |
|-----|-----:|---:|-----:|--------------:|
| coordinator_regime | 0.9718 | 7.0 | 0.1388 | 4.0/10.0 |
| coordinator_regime_traj | 0.9718 | 7.0 | 0.1388 | 4.0/10.0 |
| act / convergence / oracle / stateless | 0.9718 | 7.0 | 0.1388 | 4.0/10.0 |
| fixed_t7 | 0.9710 | 7.0 | 0.1387 | 7.0/7.0 |

- Seeds: `[20260921, 20260922, 20260923, 20260924, 20260925]`
- Option-1 locks held: C̄=7±0.25; ACT/BP 4/10; Fixed T=7
- Regression locks (T8 shift + hum): **PASS**
- Stopping: one primary, five seeds; **no retune**; Affinity/C₃/hum frozen

## Honest sentence

On this toy under frozen PREREG `bf757ee4…`, **ACT = oracle** (ΔACC 0) and both coordinator arms sit on that same ceiling with Convergence / Stateless — so H1 is null by construction once per-item convergence is already optimal. Fixed is within +0.0008 ACC. Trajectory budgeting adds nothing. Silencing SD is flat. The **overthinking** signature is real but modest-to-seed-variable in magnitude: easy degrades **5/5** with seed-mean **ΔACC(T=12 vs T\*=4)=−0.3418** / **ΔNLL=+1.5982**; between T\*=4 and cost-matched Fixed T=7 the easy loss is ~0 (Fixed within 0.0008 of coordinator). hard 0/5; T\* 4≠12. Regime-dependent depth exists as a measurement; a persistent coordinator still does **not** beat lookup / ACT / Fixed on the preregistered primary comparisons.

## What this does **not** authorize

- No Affinity / C₃ / hum retune
- No new PREREG / no T12.2 (gate defect=false)
- No touch of `t12b/`
- Parent T11 / T11.2 / T11.3 PREREG hashes untouched

## Paths

- Close: `docs/T12_CLOSING.md` (this file)
- Results: `results/T12_RESULTS.md`
- Primary: `results/t6/T6_PRIMARY_AGGREGATE.md`
- Inventory: `ARTIFACT_INVENTORY.md`

## Closing amend (2026-09-23 19:43 PT; Kyle + GNURadio)

- **H3:** report effect size (ΔACC / ΔNLL of OT degradation from T6 `measures.OT_T`) **alongside** the 5/5 easy-degrades count — support stays H3 support but does not read bigger than the magnitude.
- **H1:** state **ACT = oracle** (ΔACC 0) as the null reason (coordinator = ACT = oracle ceiling).
- Overall **mixed** / per-H verdicts **unchanged**. Primary not re-run. Affinity untouched.

## Closing amend (2026-09-23 19:59 PT; Kyle + GNURadio — H3 T wording)

- **H3:** state the **T** at which easy ΔACC −0.3418 is measured: Fixed OT sweep **T=12 vs T\*=4** (not an unspecified depth).
- **Reconcile:** between **T\*=4** and cost-matched Fixed **T=7**, easy loss is **~0** (seed-mean ΔACC≈−0.0016); Fixed within **0.0008** ACC of coordinator.
- Mixed / per-H verdicts **unchanged**. Primary not re-run. Affinity untouched.

