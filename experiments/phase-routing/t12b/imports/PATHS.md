# T12b imports — documented paths

| Import | Path | Notes |
|--------|------|-------|
| **hum_measure** | `imports/hum_measure` → `/workspace/scp/t11/imports/hum_measure` | Symlink. Frozen C₃ / λ / a₀ / Affinity `__all__` — **do not retune**. |
| **ScoreAffinityProvider** | `hum_measure.affinity_inputs` | Same L2 as T11.3 primary / addendum. |
| **T11.3 sealed** | `generators/sealed` → `../../t11_3/generators/sealed` | Read-only reuse. |
| **stateless_argmax reference** | `../t11_3/addendum_stateless_argmax.py` | Pattern for argmax control; do not edit T11.3. |
| **T_hold(n=3)** | `t11/imports/hum_measure/T11_N3_THOLD_RECEIPT.md` | Operating hold **65** (coupled). |

Routing helpers for a full primary may import from `experiments/phase-routing/t11/routing/` (HumMeasureOrchestrator) without copying.
