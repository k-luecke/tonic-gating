# T12 T5 Freeze Receipt

- **Status:** **PASS** / **FROZEN**
- **PT:** 2026-09-23 19:29 PT
- **PREREG.sha256:** `bf757ee4fc01e5c40b45f035150e7b716d12f7d8b30d745b9a7669f79bbf307c`
- **Draft content sha (gate):** `8e04b801d9da5e1c94221ecff07d3555dc3c4f630599434160173f61388ffbd1`
- **Rule-19 receipt:** `results/t5/T5_RULE19_GATES.json`
- **Affinity / C₃ / hum:** UNTOUCHED
- **T12b:** not frozen (held)

## Re-run

```bash
cd /workspace/scp/experiments/phase-routing/t12
python3 scripts/freeze_prereg.py            # status
python3 scripts/freeze_prereg.py --authorize  # emit (idempotent if already frozen + gates PASS)
```
