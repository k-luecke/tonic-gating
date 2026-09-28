# T12b ARTIFACT_INVENTORY

Snapshot: 2026-09-23 18:27:27 PT

Scaffold only — no primary results. Sealed batch reused via symlink (not copied).

## Local artifacts

| Path | Bytes / type | sha256 |
|------|-------------:|--------|
| `README.md` | 4507 | `d59e0cd20fd83892092be84764650beb2a5785a0eaca1436ea0f464d37c34b90` |
| `docs/T12B_CROSSOVER_PREDICTION.md` | 2231 | `eee4fd552abfb6a4d3feebdd778a68aefe01693f80658a3eef03c055c6a8189a` |
| `docs/T12B_PREREG_DRAFT.md` | 8185 | `de838c6c09f02db35287d87cfd1b12adaa60e0ccfd37edf08c9eddd7b70ce941` |
| `docs/T12B_RULE19_GATES.md` | 2511 | `49358fc79e8a6d713d831a20071bfd1d4003e763c18d7d8c10a3fb2c5cc8441a` |
| `generators/SEALED_POINTER.md` | 773 | `5b58910a27986035d168eb19d28f95235c2ef66c683a8787270fc956296af000` |
| `generators/sealed` | symlink → `../../t11_3/generators/sealed` | — |
| `imports/PATHS.md` | 824 | `6a789524c84db7901865e6bc3648328759d6bf6e74b7b94a02b9a0df57425b32` |
| `imports/hum_measure` | symlink → `/workspace/scp/t11/imports/hum_measure` | — |
| `scripts/arms.py` | 4415 | `7a0f99d71b11e773cd2077d6c7b88bf52df2b45e1c394707be16823d93485ddd` |
| `scripts/freeze_prereg.py` | 493 | `fe1e96a29ffb67afc6426cbe20af1cf862ace52826ad8f821ffebd8ee7e33530` |
| `scripts/gap_inject.py` | 4342 | `d9b96390bc378719fcc9ce3de8ee012d6b7077f10d7689d805e9cf76d4cd45dd` |
| `scripts/primary_run.py` | 4539 | `10e849f0b844adf723f88a220100aaa84e9dc0f58834877b1e1ba36d6f6862ea` |

## Locked external refs (read-only)

| Path | Bytes | sha256 |
|------|------:|--------|
| `../t11_3/PREREG.md` (untouched) | 9039 | `27ab4a9573ec8c01ed043d72ef6c83ba47fdd4a764a373acb7453381bb9bd442` |
| `../t11_3/PREREG.sha256` (untouched) | 76 | `3cd61aa86cf975c89d4b72d1672a322325b8af657584357979835d2885a203d9` |
| `../t11_2/PREREG.sha256` (untouched) | 76 | `3f55e3f1ac442b0ca623910ef5ffa01e17ac4f93414dae0eab1cfb96877e8e8b` |
| `../t11_3/generators/sealed/ALL_SEQUENCES.jsonl` (reuse) | 114108 | `646df30bbeacdc77432e9cb1daa366640ca0cfeb2e01216d4cb5f03a9df404c6` |

## PREREG draft (this ticket)

- **Status:** DRAFT (do not emit `PREREG.sha256`)
- **Path:** `docs/T12B_PREREG_DRAFT.md`
- **sha256:** `de838c6c09f02db35287d87cfd1b12adaa60e0ccfd37edf08c9eddd7b70ce941`

## Hard locks verified at inventory time

- T11.3 PREREG.sha256 content: `27ab4a9573ec8c01ed043d72ef6c83ba47fdd4a764a373acb7453381bb9bd442`
- T11.2 PREREG.sha256 content: `593762c4c19e85d1720f6b46edbb93869258c4047ad023e42e144416aaed20d1`
- C₃ / hum / Affinity `__all__`: not retuned in this tree
- Hold for crossover: **T_hold(n=3)=65** (documented in T11_N3_THOLD_RECEIPT; not TBD)

## G9 / G10 wire (2026-09-23 PT)

| Artifact | Path |
|----------|------|
| HumMeasureArm + stub refuse | `scripts/arms.py` |
| HP every drive | `scripts/highpass_drive.py` (T8 roofing α) |
| G10 gap inject | `scripts/gap_inject.py` (`inject_gap_with_g10`) |
| Dry-run | `scripts/primary_run.py --dry-run` → `results/DRY_RUN.json` |

No `PREREG.sha256`. Affinity/C₃/hum packages read-only.


## Update 2026-09-23 19:47:58 PT — majority_k3 secondary + Rule 19 PASS (no freeze)

| Artifact | sha256 / note |
|----------|---------------|
| `docs/T12B_PREREG_DRAFT.md` | `408eaaaca59d0c210e1b1ef50d2494213662da1059b3542c38ce866651801b21` (was `6aaa5d91…`) |
| `docs/DIFF_6aaa5d91_to_current.md` | draft diff summary |
| `docs/T12B_RULE19_GATES.md` | +G11 majority_k3 |
| `scripts/arms.py` | +`MajorityK3Arm` labeled secondary |
| `scripts/rule19_freeze_gates.py` | G1–G11 runner |
| `scripts/freeze_prereg.py` | gates-only; emit refused while DRAFT |
| `results/T12B_RULE19_GATES.{json,md}` | **PASS**; `prereg_sha256_emitted=false` |

**No `PREREG.sha256`.** Affinity/C₃/hum untouched. Kyle calls freeze after reviewing PASS + draft diff.


## Update 2026-09-23 20:01:07 PT — FREEZE GO + primary (Kyle + GNURadio)

| Artifact | sha256 / note |
|----------|---------------|
| `docs/T12B_PREREG_DRAFT.md` | `408eaaaca59d0c210e1b1ef50d2494213662da1059b3542c38ce866651801b21` (unchanged at freeze) |
| `docs/DIFF_9daf251c_to_6aaa5d91.md` | earlier hop (reconstructed; exact 9daf bytes missing) |
| `docs/DIFF_6aaa5d91_to_current.md` | prior hop (kept) |
| `PREREG.md` + `PREREG.sha256` | `851124b3c4bc66cb8e7c96c4f501084f4de186574f4f8d9446cac5879f62a3f0` |
| `results/T12B_RULE19_GATES.{json,md}` | PASS + emit |
| `results/T12B_PRIMARY_AGGREGATE.{json,md}` | primary × 5 seeds |
| `results/T12B_SEED_20260923..27.json` | per-seed receipts |

**Affinity/C₃/hum untouched.** Parent T11.* PREREGs untouched. No T12b.2.

