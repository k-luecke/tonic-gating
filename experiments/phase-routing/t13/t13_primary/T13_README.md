# T13 — Looped architectures (MoE-of-loops / context slices)

| Field | Value |
|-------|--------|
| **Status** | **PRIMARY COMPLETE** 2026-09-24 ~08:41 PT — verdict **mixed** |
| **Date** | 2026-09-24 PT |
| **Owner** | Kyle (ZetaZero) / Drift Bound Labs |
| **Folder** | `/workspace/scp/experiments/phase-routing/t13/` |
| **Line** | **New line** — looped architectures / MoE-of-loops with **context slices** (not tonic-gating continuation) |
| **Greenlit** | Kyle + GNURadio, Project Miller 2026-09-23 ~23:18 PT; freeze GO + gate-defect re-freeze 2026-09-24 |
| **Scale** | **`mythos_tiny`** — local first look |
| **Affinity** | **FROZEN** `7dae877a3975c1ba4ca264f03f572d6749cce39a114ea3adbcffc010c7852c50` — untouched |
| **PREREG.sha256** | `3176eb5ea3458047b820ff5511644d20a697a7a9485d5118560ff7b71e1f981d` |
| **Draft content sha** | `1e3f7fe80c4763b827e1bb9cb566adf83551255c5bd4aad11073b91639ea0119` |

## Mission (one line)

Ask whether **multiple looped experts**, each with its **own depth budget** and **own context slice**, yield **H_int** (per-expert intelligibility) at **matched compute** — vs a single shared loop with MoE feed-forward.

## Post-freeze sequence (DONE)

1. **Train mixture** — G9 **PASS** (NaN / non-finite / explosive grads clear; final NLL≈3.69)
2. **G13 Acc** — **PASS** max_dev≈0 (pairwise∪N=36; max-dev≤0.1; manipulation check only)
3. **G17 Acc** — **PASS** Acc=0.235 ≤ 0.273 (n=1000)
4. **Primary ×5 seeds** — verdict **mixed** (slice H_int clears three-clause support per seed, but slice does **not** beat shared_ctx by >0.05)

## Primary aggregate (H_int)

| Metric | Value |
|--------|------:|
| r_phase (slice mean) | 0.8686 |
| r_plain | −0.6945 |
| r_random | 0.8014 |
| r_no_medium | 0.4124 |
| r_oracle | 0.9604 |
| r_phase shared_ctx | 0.8663 |
| slice_beats_shared | False |
| interval | 0.05 |
| **Verdict** | **mixed** |

## Hard locks (still)

1. Do **not** edit `affinity_inputs.py` (sha must remain `7dae877a…`).
2. Do **not** touch sealed T11 / T12 / T12b `PREREG.sha256` files.
3. No PREREG amendments post-freeze except gate defect.

## Tree

```text
experiments/phase-routing/t13/
  README.md
  ARTIFACT_INVENTORY.md
  FREEZE_META.json
  PREREG.md / PREREG.sha256
  docs/
  scripts/
    mythos_config.py / mythos_model.py
    train_mixture.py / eval_acc_gates.py / primary_run.py
    rule19_freeze_gates.py
  results/
    T13_TRAIN_MIXTURE.{json,md}
    T13_ACC_GATES.{json,md}
    T13_PRIMARY.{json,md}
    primary/ checkpoints/
```
