# T12 Primary RESULTS (T7–T8 close)

| Field | Value |
|-------|--------|
| **Status** | **CLOSED** 2026-09-23 19:38 PT (amend 2026-09-23 19:43 PT; H3-T wording 2026-09-23 19:59 PT) |
| **Overall verdict** | **mixed** (H3 support; H1/H2/H4 null) |
| **PREREG.sha256** | `bf757ee4fc01e5c40b45f035150e7b716d12f7d8b30d745b9a7669f79bbf307c` |
| **T6 aggregate** | `results/t6/T6_PRIMARY_AGGREGATE.{json,md}` |
| **Per-seed** | `results/t6/T6_SEED_{20260921..20260925}.json` |
| **Seeds** | 20260921, 20260922, 20260923, 20260924, 20260925 |
| **Width / n_per_task / seq_len** | 128 / 8 / 48 |
| **Primary arm** | `coordinator_regime` |
| **Affinity sha** | `7dae877a…` untouched=true |
| **Gate defect** | false (no T12.2) |
| **Closing** | `docs/T12_CLOSING.md` |

## Per-H table

| H | Claim (short) | Verdict | Key numbers |
|---|----------------|---------|-------------|
| **H1** | Regime budget beats Fixed on ACC/LPC; BP separates | **null** | **ACT = oracle** (ΔACC 0): coord matched oracle; ACT matched coord → oracle ceiling. vs Fixed ΔACC=+0.0008 ΔLPC=+0.0001; BP easy=4 hard=10 gap=6 (5/5) |
| **H2** | regime+traj beats ACT / CONV | **null** | traj vs ACT ΔACC=0 ΔLPC=0; vs CONV ΔACC=0 ΔLPC=0 |
| **H3** | OT regime-dependent (easy hurt / hard helped) | **support** | easy degrades_past **5/5**; seed-mean **ΔACC(T=12 vs T\*=4)=−0.3418** / **ΔNLL=+1.5982**; between T\*=4 and Fixed T=7 easy loss ~0 (ΔACC≈−0.0016; Fixed within 0.0008 of coord); hard 0/5 ΔACC=0 ΔNLL=0 (T\*=12) |
| **H4** | Coordinator beats Stateless under silence (SD) | **null** | SD ΔACC=+0.0000; coord_beats 0/5 |

## Null rule (§1.3)

| Clause | Fired? | Detail |
|--------|:------:|--------|
| (1) Fixed and/or ACT match coordinator | **yes** | **ACT = oracle** (ΔACC 0); Fixed ≈ match |
| (2) Optimal T\* same across regimes | **no** | T\* 4 vs 12 |
| (3) SD coordinator ≤ Stateless | **yes** | ΔACC=0; 0/5 beats |

## Arm aggregates (from T6)

| Arm | ACC̄ | C̄ | LPC̄ |
|-----|-----:|---:|-----:|
| act | 0.9718 | 7.0 | 0.1388 |
| convergence | 0.9718 | 7.0 | 0.1388 |
| coordinator_regime | 0.9718 | 7.0 | 0.1388 |
| coordinator_regime_traj | 0.9718 | 7.0 | 0.1388 |
| fixed_t7 | 0.9710 | 7.0 | 0.1387 |
| oracle | 0.9718 | 7.0 | 0.1388 |
| stateless_regime | 0.9718 | 7.0 | 0.1388 |

## Stopping / locks

- One preregistered primary × five seeds — **done**
- Affinity / C₃ / hum — **not** retuned
- t12b — **not** touched
- No new PREREG emitted
- T8 regression lock — **PASS** (see T6 aggregate `regression_locks`)

## Re-run primary (do not for close)

```bash
cd /workspace/scp/experiments/phase-routing/t12
python3 scripts/run_t6_primary.py   # already COMPLETE; close does not re-run
```

## Closing amend (2026-09-23 19:43 PT)

H3 effect sizes from `results/t6/T6_SEED_*.json` → `measures.OT_T.by_regime` (Fixed sweep). H1 null reason: ACT=oracle. Verdicts unchanged; primary not re-run.

## Closing amend (2026-09-23 19:59 PT; H3 T wording)

H3 easy ΔACC **−0.3418** is measured at **T=12 vs T\*=4** (Fixed OT sweep). Between T\*=4 and cost-matched Fixed T=7, easy loss ~0; Fixed within 0.0008 of coordinator. Verdicts unchanged; primary not re-run.

