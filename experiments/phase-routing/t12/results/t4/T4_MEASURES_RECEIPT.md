# T12 T4 Measures Receipt

- **Status:** **PASS**
- **PT:** 2026-09-23 19:16:00 PT
- **Seed:** 20260921 (smoke seeds [20260921, 20260922])
- **Primary arm:** `coordinator_regime`

## §4.1 Measures

| ID | Value |
|----|------:|
| ACC | 0.9667 |
| C̄ | 7.0000 |
| LPC (ACC/C̄) | 0.1381 |
| NLL | 0.4114 |
| T_b mean | 7.0000 |
| BP easy/hard | 4.0/10.0 (gap=6.0) |
| SD ΔACC | 0.0 |
| OT easy degrades | {'T_star': 4, 'degrades_past': True} |
| OT hard | {'T_star': 12, 'degrades_past': False} |

- **Regression lock:** available=True pass=True

## Re-run

```bash
cd /workspace/scp/experiments/phase-routing/t12
python3 scripts/run_t4_measures.py
```

## Notes

- Smoke batch only (CPU, width 64). Affinity / C₃ / hum **untouched**.
- No PREREG.sha256. No freeze.
