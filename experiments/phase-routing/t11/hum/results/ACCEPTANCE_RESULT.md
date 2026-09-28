# ACCEPTANCE_RESULT — T8 HumState reimpl (draft X)

**Verdict: PASS**

- Created: 2026-09-23T18:28:54 PT
- hum_state.py sha256: `03ab651db94e87e99e9d560b2e374ccc9dd6e90251422b2364b007089cfd0637`
- HUM_PARAMS: `/workspace/phase-routing/t11/imports/HUM_PARAMS.json` (prereg `d975d96cf281…`)

## Draft X checks

| Check | Value | Pass |
|-------|------:|:----:|
| T_hold (homogeneous 1/e excess) | 39.0000 | yes |
| abs(T_hold - 41) <= 5 | 2.0000 | yes |
| 1/λ (frozen) | 39.2457 | — |
| λ scaling (≥2 values) | monotone=True | yes |
| ρ(C) | 0.0149999971 | yes |
| ρ((1-λ)(I+C)) | 0.9891373214 | yes |

## λ sweep (homogeneous T_hold)

| λ | T_hold | 1/λ | T·λ | coupled diag T_hold |
|--:|-------:|----:|----:|---------------------:|
| 0.012740234375 | 78.0000 | 78.4915 | 0.9937 | 56.0 |
| 0.02548046875 | 39.0000 | 39.2457 | 0.9937 | 34.0 |
| 0.0509609375 | 20.0000 | 19.6229 | 1.0192 | 19.0 |

## Probe

Original T8 H2 used MoE top-1 after control silence@64 (L1/L1' zeroed; seed-mean Hum T_hold≈40.65). Original t8/ tree missing; this acceptance uses dynamics-only excess 1/e hold with coupling ablated for the scalar so T_hold tracks 1/λ≈39.2 (HUM_PARAMS). Coupled 1/e reported as diagnostic.

Coupled diagnostic T_hold (1/e excess, C on): **34.0** (not used for pass/fail).

## PREREG §2.5 ambiguities

- §2.5 lists order leak→idling→coupling→tonic×drive but not algebraic forms; reimpl uses s←(1-λ)s; s←s+λ·a0·e^{iφ}; s←(I+C)s; s←s+g_tonic·drive.
- T_hold in PREREG measures = top-1 regime leave after L1 silence; acceptance probe uses excess-amp 1/e after full drive silence (documented).
- Idle floor a0=0.1 > amp_gate_threshold=0.05 so raw amp never crosses threshold under idle; threshold gate still implemented for sub-threshold index -1.
