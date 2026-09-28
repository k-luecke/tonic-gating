# T12b Primary AGGREGATE

- **PT:** 2026-09-23 20:01:01 PT
- **PREREG.sha256:** `851124b3c4bc66cb8e7c96c4f501084f4de186574f4f8d9446cac5879f62a3f0`
- **Seeds:** [20260923, 20260924, 20260925, 20260926, 20260927]
- **Primary contrast:** `hum_measure vs one_step_memory`
- **Labeled secondary:** `majority_k3`
- **Affinity:** `7dae877a3975c1ba…` untouched
- **G9/G10:** counted HumState; mid-gap lure active

## CSR_gap (seed-mean)

| Arm | CSR_gap | role |
|-----|--------:|------|
| `hum_measure` | 0.652538 | primary counted |
| `stateless_argmax` | 0.000000 | primary control |
| `one_step_memory` | 0.298708 | primary null baseline |
| `majority_k3` | 0.639002 | labeled secondary |

## Deltas

| Contrast | Δ CSR_gap |
|----------|----------:|
| CSR_gap(hum)-CSR_gap(one_step_memory) | 0.353830 |
| CSR_gap(hum)-CSR_gap(stateless_argmax) | 0.652538 |
| CSR_gap(hum)-CSR_gap(majority_k3) | 0.013536 |

- **seed_interval:** 0.05
- **null_holds (hum ≤ one_step + interval):** False
- **support/null hint:** `support_lean` (numbers-only close deferred)
