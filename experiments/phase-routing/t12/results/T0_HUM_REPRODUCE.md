# T0_HUM_REPRODUCE — GNURadio acceptance receipt

**Overall: PASS**

- Created: 2026-09-23T18:28:54 PT
- Tree: `/workspace/scp/experiments/phase-routing/t12`
- PREREG draft sha (untouched): `78091e84b88e006fb27dc3f7648d0e545387060ab96140f6afe6567c232a6977`

## Bullets

| # | Bullet | Pass | Key numbers |
|---|--------|:----:|-------------|
| 1 | `hum_measure` imports | yes | `/workspace/scp/t11/imports/hum_measure` |
| 2 | T8 hum unit tests (acceptance) | yes | T_hold=39.0 (target 41±5); 1/λ≈39.245745822474326 |
| 3 | T8 rerun persistence measurement | yes | MoE seed-mean T_hold=26.846875; shift ΔL=0.01707748666642206 |
| 4 | Locks: n=2 / HP every drive / amp never phase-gated | yes | homog T_hold=39.0; HP easy/hard=9.0/9.0 |
| 5 | Affinity / C₃ / hum_state untouched | yes | see shas below |

## Persistence vs T8 reference

| Measure | T8 reference | This run |
|---------|-------------:|---------:|
| Dynamics homog T_hold (1/e) | 39.0 | 39.0 |
| 1/λ | ≈39.2457 | 39.245745822474326 |
| MoE hum seed-mean T_hold (rerun primary) | 26.846875 | 26.846875 |
| MoE hum ΔL shift0.3 (regression lock) | 0.01707748666642206 | 0.01707748666642206 |
| n=2 homog T_hold | — (same family ≈39) | 39.0 |
| n=2 coupled diagnostic T_hold | — | 62.0 |

## Frozen package shas (must match baseline)

| File | sha256 |
|------|--------|
| `hum_measure/__init__.py` | `b51cda27b48e7ce29c801b5a86bb8f613a38cb4ba261e40459dbdb842b56721a` (OK) |
| `hum_measure/affinity_inputs.py` | `7dae877a3975c1ba4ca264f03f572d6749cce39a114ea3adbcffc010c7852c50` (OK) |
| `hum_measure/COUPLING_HASHES.json` | `482fb1b53bfddc0e771145a0783f3b65104231d537cbba507d4783d13a27902b` (OK) |
| `C_n3` | `40e3c0c95849ad67af6738e7842c1a4d24783dafa742ed247d03391db37a1493` (OK) |
| `hum_state.py` | `03ab651db94e87e99e9d560b2e374ccc9dd6e90251422b2364b007089cfd0637` (OK) |

## Affinity / C₃ note

- Affinity `__all__` **not edited** (T11 n=3 specialist package left frozen).
- C₃ coupling hash unchanged; T12 builds **n=2** via same `make_coupling` formula only in smoke under `results/t0/`.
- No T11 / T11.2 / T11.3 PREREG hash emit or edit.

## Commands (re-run from t12/)

```bash
cd /workspace/scp/experiments/phase-routing/t12
python3 scripts/run_t0_hum.py          # all T0 bullets
# or stepwise:
python3 tests/test_hum_import.py
python3 scripts/run_t8_acceptance.py
python3 scripts/check_t8_rerun_persistence.py
python3 scripts/smoke_hum_n2.py
```

## Documented paths (see imports/PATHS.md)

| Name | Path |
|------|------|
| hum_measure | `imports/hum_measure` → `/workspace/scp/t11/imports/hum_measure` |
| t8_hum dynamics | `imports/t8_hum` → `.../t11/hum/` (`acceptance_t8.py`, `hum_state.py`) |
| T8 primary results | `imports/t8_results_primary` → `.../t8/results/primary/` |
| T8 PREREG | `imports/t8_PREREG.md` → `/workspace/scp/t11/imports/t8_PREREG.md` |

## New / edited file shas (this receipt run)

| Path | Bytes | sha256 |
|------|------:|--------|
| `ARTIFACT_INVENTORY.md` | 3597 | `621fcff98834f2af784f7124ba9b7b7eaa92ad42e2459db34b3d98f747629e2e` |
| `README.md` | 3749 | `3d23934d17881207babea8f3df786b9966c099d346db49c6b2e790cf7937c447` |
| `imports/PATHS.md` | 2570 | `1a895171def9e53f3438378f2060b70cde7ac213c9f1e5eef5fdcb06647a3bb5` |
| `results/T0_HUM_REPRODUCE.md` | 4919 | `05423277269ea2a4ac858339f4fc87c2513a0376c182af0983ba467f0c7a496c` |
| `results/t0/ACCEPTANCE_RESULT.json` | 3416 | `3e86757751c174a40b0b5269ceeb4c932b751eb48a44e57ca696e2645cca556e` |
| `results/t0/ACCEPTANCE_RESULT.md` | 1796 | `651663eceddace8369a04a71e510ed2e4fa422c0851fd68339220ea4da30503e` |
| `results/t0/SMOKE_HUM_N2.json` | 1967 | `f7875fc45fddd617b89bc1e14ab272208d6d4ac47664f18c33af58f099533e1a` |
| `results/t0/SMOKE_HUM_N2.md` | 877 | `bfdb6ff30d050f932f2e80419264958ac938302b2023808baafbe2874a024498` |
| `results/t0/T0_SUMMARY.json` | 2784 | `46e08c94713229431c4cd71509308153e7f379a035403ebd2bd07650c6e5b466` |
| `results/t0/T8_RERUN_PERSISTENCE_CHECK.json` | 2210 | `054af218d9aa4bc38a3e5e8bf2d56e3c4df43cf97f47fdb134f24e3555fab174` |
| `scripts/check_t8_rerun_persistence.py` | 4470 | `5171a15a1dae9e7472e7c715bd4c8ed065f0acc123f98d3c2ab25825f5181d6c` |
| `scripts/freeze_prereg.py` | 490 | `ab323d06c9b4068e7625202c1b325e88125173b3e756d80720b2548e9ff542d8` |
| `scripts/highpass_drive.py` | 1908 | `216ea5320f6b48ff2c81e725c81feb1628b9d653ec6c743e2ceb9935beefe500` |
| `scripts/run_t0_hum.py` | 11749 | `ed783cb7c691e30beff915d5b4d5399766b0826a4cbb19f0632fb829e38cabbb` |
| `scripts/run_t8_acceptance.py` | 1696 | `e8ea393fae0da42fb008ee37de90f4b775da401e395aab63a49d3ce0de413766` |
| `scripts/smoke_hum_n2.py` | 7701 | `a60e46077be826de36d7da1421ffbd450813cddc51bfd2073af853dc250a7ab9` |
| `tests/test_hum_import.py` | 741 | `354a17a4fa07c28da23aaeb2354177e62c18d60c68d19f7850e88922352e01cc` |
| `tests/test_t0_hum.py` | 529 | `30d4de584188bd6453926cd0cad0a2e6079b5cca201c91084dda3fe934ca4e8f` |
