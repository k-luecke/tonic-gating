# T11 routing smoke harness

**Location:** `/workspace/scp/experiments/phase-routing/t11/routing/`  
**Scope:** Orchestrator arms only (no specialist inference). One-seed smoke on sealed counting batch.

## Arms

| Arm | Module | Status |
|-----|--------|--------|
| Hand rule | `hand_rule.py` | **Full** — Kyle-signed `t11/HAND_RULE.md` |
| Context classifier (stub) | `classifier_stub.py` | **Stub** — untrained keyword-count argmax + EMA buffer (disclosed baseline) |
| Context classifier (MLP) | `classifier_mlp.py` | **Calibrated** — 2-layer MLP, hashed weights in `artifacts/` |
| Hum (local baseline) | `hum_orchestrator.py` | **Local dynamics smoke** — HumState n=3 + keyword projection |
| Hum (measure) | `hum_measure_arm.py` | **GNURadio wired** — AffinityProvider + L1 RegimeProbe + schema; Keyword fallback labeled |

## Classifier calibration

```bash
cd /workspace/scp/experiments/phase-routing/t11/routing
python3 train_classifier.py
```

**Split (disjoint):**
- Train seeds: `20260924, 20260925, 20260926`
- Val seed: `20260927` (early-stop + temperature)
- Eval / smoke seed: `20260923` — **held out; never fit**

Weights: `artifacts/classifier_mlp_v1.json` (+ `.npz`, `CLASSIFIER_WEIGHTS.sha256`).  
Receipt: `/workspace/scp/t11/docs/T11_CLASSIFIER_CALIBRATION_RECEIPT.md`.

## Run smoke

```bash
cd /workspace/scp/experiments/phase-routing/t11/routing
python3 smoke_run.py --seed 20260923
```

Outputs:

- `results/smoke_seed*_summary.json` / `_detail.json` / `SMOKE_FILES.sha256`
- `/workspace/scp/t11/docs/T11_ROUTING_SMOKE_RECEIPT.md`
- `/workspace/scp/t11/docs/receipts/T11_ROUTING_SMOKE_RECEIPT.json`

## Window-break carry

Harness conditions applied per arm (hum has no special standing):

- `hum-state` / `classifier-buffer` / `context-carry` / `cold`
- Score M6a (first post-break CSR), M6b (bytes), M6c (CSR/byte), M6d (stale penalty)

## Not freeze-ready

PREREG stays draft. Real `hum` arm wired to hum_measure (Keyword L2 fallback labeled). Side-channel disclosure **PASS** 2026-09-23 — see `/workspace/scp/t11/docs/T11_SIDE_CHANNEL_DISCLOSURE_RECEIPT.md`. Labeled interim channels remain. Classifier weights are hashed and freeze-ready as an artifact; PREREG itself is **not** frozen.
