# T13 ARTIFACT_INVENTORY

Snapshot: 2026-09-24 08:42:16 PT

**PRIMARY COMPLETE.** PREREG.sha256 `3176eb5e…` (gate-defect re-freeze). Affinity untouched. Parents untouched. Verdict **mixed**.

## Local artifacts (excludes *.npz checkpoints)

| Path | Bytes | sha256 |
|------|------:|--------|
| `ARTIFACT_INVENTORY.md` | 2231 | `9d2859576c5cf2553ee0d1f8c1916ecaca54f54db134f1f58d3266daecbe145c` |
| `FREEZE_META.json` | 1482 | `aeeb10a730b450e0af79f09e66095f6f327331830347fa6d114411a83b6408b5` |
| `PREREG.md` | 26890 | `3176eb5ea3458047b820ff5511644d20a697a7a9485d5118560ff7b71e1f981d` |
| `PREREG.sha256` | 76 | `c36845f3516006e9e6608907e494531149cc81ebb87411f78dd26b13bddfcf01` |
| `README.md` | 2529 | `19f8d0b4b6fc9b14c7abb90a2e23c3672497111eb19a8c6ac025c07df8308613` |
| `docs/T13_PREREG_DRAFT.md` | 26785 | `1e3f7fe80c4763b827e1bb9cb566adf83551255c5bd4aad11073b91639ea0119` |
| `docs/T13_RULE19_GATES.md` | 5940 | `a9c68d7c249442f4c3d5145c372305a2d450a44f57b0826a5feaf604782e63bd` |
| `results/.gitkeep` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `results/T13_ACC_GATES.json` | 3210 | `23afa73e4a529fb3df447113ca95ec4ab8fa55584a472b2f248aac5c8937d815` |
| `results/T13_ACC_GATES.md` | 510 | `929d2033ef486314ebae9c875619a47fdf372f48183b1a10dc9844e8e21d86e0` |
| `results/T13_PRIMARY.json` | 3933 | `618babecdfd2e8669d036d41d94536ae3684b37c495bd16ff55a5fffb91a049f` |
| `results/T13_PRIMARY.md` | 908 | `f3563b66b435fda72637fa6d4b90a51aafac4544a06ab1549988d09588edea21` |
| `results/T13_RULE19_GATES.json` | 7895 | `6eae4465f8e9f4ede1e7da463663ea483102ff13b07c8ca93ab47e0720fa61f9` |
| `results/T13_RULE19_GATES.md` | 2980 | `1d7e0cb58f1b4e656e04d0e4fa6ddb476b4c5a5df1e7885202091b3333bf05ae` |
| `results/T13_TRAIN_MIXTURE.json` | 2952 | `bc64a60c9574e1574552423ce431452804cd0aef1484d992f536789edb6afc83` |
| `results/T13_TRAIN_MIXTURE.md` | 558 | `c41f2968f1a96c2c7d72ffeb77aefeaf0743b12d6855d1e78eae4f227244331e` |
| `results/primary/T13_PRIMARY.json` | 3933 | `618babecdfd2e8669d036d41d94536ae3684b37c495bd16ff55a5fffb91a049f` |
| `results/primary/T13_PRIMARY.md` | 908 | `f3563b66b435fda72637fa6d4b90a51aafac4544a06ab1549988d09588edea21` |
| `scripts/eval_acc_gates.py` | 8362 | `bbee1ac50a3d06e62ae4cb2d1b151778f852c61b0c0859c049213fc82744e63b` |
| `scripts/mythos_config.py` | 4307 | `526740b76b9ebfee032928682ae2f2af6bc65f5d5d2b6a3a3387b50f002a60ac` |
| `scripts/mythos_model.py` | 10930 | `4bd413220a9b5f6114674d6b8adab0890150400b03df83d9fd3ed3301c2f6dad` |
| `scripts/primary_run.py` | 11872 | `a8538632baafcb9cdf89baae5b20aecea2aecfe17f7ae8eb5e1a23da377e381e` |
| `scripts/rule19_freeze_gates.py` | 27132 | `7474482c85ff776fdd7bce1ab0194e74ee7995f4f566dfdac1a25c0b31a33317` |
| `scripts/train_mixture.py` | 5452 | `db0532b568f311e0b620d44ef5e0538799af5bb81f51c4edf4eec8088c024968` |

## Checkpoints: 11 `.npz` under `results/checkpoints/` (~633KB each)

## Pins
- **PREREG.sha256:** `3176eb5ea3458047b820ff5511644d20a697a7a9485d5118560ff7b71e1f981d`
- **Draft content sha:** `1e3f7fe80c4763b827e1bb9cb566adf83551255c5bd4aad11073b91639ea0119`
- **Affinity:** `7dae877a3975c1ba4ca264f03f572d6749cce39a114ea3adbcffc010c7852c50` UNTOUCHED
- **Parents:** T11.2 `593762c4…` / T11.3 `27ab4a95…` / T12 `bf757ee4…` / T12b `851124b3…`

## Gates / primary
- G9 PASS (train)
- G13 Acc PASS max_dev≈0 (manipulation check; pairwise∪N=36; max-dev not RMS)
- G17 Acc PASS Acc=0.235 ≤ 0.273 (n=1000)
- Primary verdict **mixed** (H_int three-clause clears on slice; slice ≉ beat shared_ctx)
