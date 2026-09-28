# T12 T3 Calibration Receipt

- **Status:** **PASS**
- **PT:** 2026-09-23 19:15:51 PT
- **Target C̄:** 7.0 ± 0.25
- **ACT C̄:** 7.0000 (PASS)
- **Coordinator C̄:** 7.0000 (PASS)
- **Fixed T=7 C̄:** 7.0000 (PASS)
- **Convergence C̄:** 7.0000 (informational; within_tol=True)
- **Separation retained:** PASS

## Locked values

| Param | Value |
|-------|------:|
| ACT prior_mean | 7.0 |
| ACT easy_center | 4.0 |
| ACT hard_center | 10.0 |
| CONV ε | 0.05 |
| CONV τ_easy | 1.335 |
| CONV τ_hard | 3.337 |
| CONV τ_default | 2.336 |
| Fixed T | 7 |

## Budget profiles (mean depth)

| Arm | easy | hard | gap |
|-----|-----:|-----:|----:|
| fixed_t7 | 7.00 | 7.00 | 0.00 |
| act | 4.00 | 10.00 | 6.00 |
| convergence | 4.00 | 10.00 | 6.00 |
| coordinator_regime | 4.00 | 10.00 | 6.00 |

## Notes

- ACT/BP centers = T1 T* = 4/10 (soft-hard-to-8 removed); equal-mix C̄≈7.
- Fixed T=7 matched-mean baseline. T1 T*_easy=4 / T*_hard=10 are the locks.
- Affinity / C₃ / hum **untouched**. No PREREG.sha256.
