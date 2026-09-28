# T12 T6 Primary Receipt

- **Ticket:** T6 (one primary × 5 seeds)
- **PT:** 2026-09-23 19:34:57 PT
- **PREREG.sha256:** `bf757ee4fc01e5c40b45f035150e7b716d12f7d8b30d745b9a7669f79bbf307c`
- **Seeds:** [20260921, 20260922, 20260923, 20260924, 20260925] (T8/T11 family via `harness.parity.DRAFT_SEEDS`; FREEZE_META had no seed list)
- **Width / n_per_task / seq_len:** 128 / 8 / 48
- **Primary arm:** `coordinator_regime`
- **Affinity / C₃ / hum:** UNTOUCHED (Affinity sha `7dae877a…` match=True)
- **Gate defect:** False (→ T12.2 only if true)
- **Regression lock pass:** True

## Aggregate ACC / C̄ / LPC by arm

| Arm | ACC̄ | C̄ | LPC̄ | T_b easy/hard |
|-----|-----:|---:|-----:|--------------:|
| `act` | 0.9718 | 7.0000 | 0.1388 | 4.0/10.0 |
| `convergence` | 0.9718 | 7.0000 | 0.1388 | 4.0/10.0 |
| `coordinator_regime` | 0.9718 | 7.0000 | 0.1388 | 4.0/10.0 |
| `coordinator_regime_traj` | 0.9718 | 7.0000 | 0.1388 | 4.0/10.0 |
| `fixed_t7` | 0.9710 | 7.0000 | 0.1387 | 7.0/7.0 |
| `oracle` | 0.9718 | 7.0000 | 0.1388 | 4.0/10.0 |
| `stateless_regime` | 0.9718 | 7.0000 | 0.1388 | 4.0/10.0 |

## Primary arm vs Fixed / ACT / baselines (numbers only)

- coordinator_regime: ACC=0.9718 C̄=7.0000 LPC=0.1388
- vs fixed_t7 ΔACC=+0.0008 ΔLPC=+0.0001
- regime+traj vs ACT ΔACC=+0.0000 ΔLPC=+0.0000
- regime+traj vs CONV ΔACC=+0.0000 ΔLPC=+0.0000
- BP separation: easy=4.00 hard=10.00 gap=6.00 (separates_n=5/5)
- SD ΔACC mean=+0.0000 (coord_beats_n=0/5)
- OT easy degrades_n=5 hard degrades_n=0

## Notes

- No closing verdict (T7–T8 later). Fixed sweep = OT(T) measurement.
- Option-1 locks: C̄=7±0.25; ACT/BP 4/10; Fixed T=7.
- Affinity/C₃/hum **not** retuned. Parent T11.* PREREGs untouched. t12b untouched.

## Re-run

```bash
cd /workspace/scp/experiments/phase-routing/t12
python3 scripts/run_t6_primary.py
```
