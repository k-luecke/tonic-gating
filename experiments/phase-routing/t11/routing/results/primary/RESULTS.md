# RESULTS — T11 routing primary (multi-seed)

| Field | Value |
|-------|--------|
| **Primary id** | `t11-routing-primary-v1` |
| **Status** | complete |
| **PREREG sha256** | `c62bfda8bab88df14677e47e20f28c99ba20595bc0f9273050484d4ae92e079a` |
| **HANDOFF sha256** | `056be9628d3a07edd9fa0a2bff32b7e01ebb8fdd9a36b7b66d0ec5f7e39c8eaf` |
| **ALL_SEQUENCES sha256** | `8f4d11bd961018c6fa39670271872e9a6fa4efebc36de7224138376addff6720` |
| **classifier_mlp_v1.json** | `d482216ce91423658851529cf78653cbebb034abe516d1a4d9c0402c0e490a59` |
| **Primary start** | 2026-09-23T15:16:57 PT |
| **Primary end** | 2026-09-23T15:16:57 PT |
| **Wall clock** | 0.7 s (0.01 min) |
| **Seeds** | `[20260923, 20260924, 20260925, 20260926, 20260927]` |
| **Sequences (pooled)** | 60 |
| **T_hold(n=3)** | 65 |
| **ρ_eff** | 0.9891373214 |

## Verdicts

| Hypothesis | Verdict |
|------------|---------|
| H1 | **null** |
| H2 | **inconclusive_zero_thrash** |
| H3 | **supported** |
| H4 | **report-only-null** |
| H5 | **not_active** |
| H6 | **report-only** |

### Persistence trade-off (H2 vs H3)

- Pre-freeze prediction: H2 easy PASS; H3 risk FAIL at T_hold=65.
- Observed tradeoff flag: **False**
- H3 vs prediction: `contrary_to_pre_freeze_risk_prediction`
- H2 caveat: M2 is segment-level A→B→A with constant oracle. Sealed Distraction sequences change oracle across segments, so segment-level thrash often zeros for all arms; stateless arms have no mid-segment switch by construction. Token-level thrash not scored in locked smoke harness.

## Pooled arm summary

| Arm | Status | M1 CSR | M2 thrash | M3 mean lat | M4 renewal |
|-----|--------|--------|-----------|-------------|------------|
| `hand_rule` | full | 1.000 | 0 | 0.000 | 1.000 |
| `classifier_stub` | stub | 0.996 | 0 | 0.000 | 1.000 |
| `classifier_mlp` | calibrated | 0.977 | 0 | 0.000 | 1.000 |
| `hum_local` | local_dynamics_smoke | 0.790 | 0 | 0.625 | 0.600 |
| `hum` | gnuradio_hum_measure | 0.813 | 0 | 0.542 | 0.600 |

### M1 by type (pooled, primary arms)

**hum:** Ambiguous=0.917, Clean=0.708, Distraction=0.833, Renewal=0.720, Window-break=0.825
**classifier_mlp:** Ambiguous=1.000, Clean=1.000, Distraction=1.000, Renewal=1.000, Window-break=0.963
**hand_rule:** Ambiguous=1.000, Clean=1.000, Distraction=1.000, Renewal=1.000, Window-break=1.000

## H1 detail

- Ambiguous pooled: hum=0.9166666666666666, clf=1.0, hand=1.0
- hum≥clf: False; hum>hand: False; clean_tie_ok: True
- Contamination: classifier_mlp fit on seeds 20260924–26 (train) + 20260927 (val); only 20260923 is held out. Pooled H1 classifier numbers include train/val seeds.

## H2 / H3 detail

- M2 pooled: {'hum': 0, 'classifier_mlp': 0, 'hand_rule': 0}
- M3 pooled: {'hum': 0.5416666666666666, 'classifier_mlp': 0.0, 'hand_rule': 0.0}; abs_gap=0.5416666666666666

## Artifacts

- `results/primary/score_summary.json` sha256 `abda25202c3c6350e53fac9ae39bc43ca6cf27b9bbaffeb64af76f1c2d045dc0`
- `results/primary/RESULTS_detail.json` sha256 `42d5aa7930867eba3747916999402d1fb673c1b924acc1f04adb3d4f9221f0c7`
- `results/primary/run_meta.json` sha256 `2812843a9a09491ec2eb34002835f46f298380b311be9254ebd9b0a0b44c3497`

## Non-mutations

- Sealed batch NOT modified.
- PREREG.md NOT edited (remains FROZEN).
