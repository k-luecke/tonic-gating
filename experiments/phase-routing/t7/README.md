# T7 — Inferred φ_ref Phase Gate (harder order-3 task)

Separately preregistered follow-on to the frozen primary phase-routing experiment.
Lives under `phase-routing/t7/` so parent `results/primary/` and root `PREREG.md` stay untouched.

See **`PREREG.md`** for hypotheses, frozen knobs, and disclosure. One-seed smoke: **`results/SMOKE.md`**. Freeze checklist: **`READY_TO_FREEZE.md`**.

## Arms (primary counting)

| Arm | Router | Native \(C_\mathrm{route}\) |
|-----|--------|------------------------------|
| LR | Learned top-2 | 1024 MACs |
| LR-goal | concat(hidden, goal_emb[task]) → Linear top-2 | 2048 MACs |
| HR1 | Token hash top-1 | 0 MACs |
| PG-inferred | `Linear(128→2)`→atan2 φ_ref; half-wave cosine; top-4 | **256 MACs + 8 cos** (not “8”) |

Shared backbone: 4-layer width-128 decoder, MoE at layer 2, shared latest-marker task embedding.
**Forbidden for PG-inferred routing:** marker → `task_phases[task_id]`.

## Harder data

Order-3 affine recurrence with task coeffs in `data/metadata.json`. Default `p_flip=0.0`
after single-arm calibration (0.15 was unlearnable — see PREREG disclosure). Target LR
clean heldout ~0.5–1 nat; smoke landed ~1.26 (within 0.3–1.5 acceptance).

## Quickstart

```bash
cd phase-routing/t7
export PYTHONPATH=$PWD
../.venv/bin/python -m t7_routing.data --seed 20260921
../.venv/bin/python -m pytest -q
./run.sh smoke   # one-seed LR, LR-goal, HR1, PG-inferred + harness
```

Or per arm:

```bash
./run.sh lr
./run.sh lr_goal
./run.sh hr1
./run.sh pg_inferred
./run.sh harness
```

## Layout

```
t7/
  t7_routing/     # self-contained package (model, data, routers, costs)
  train.py
  harness.py
  tests/
  data/           # order-3 dataset + MANIFEST.sha256
  results/
    exploratory/smoke/   # one-seed checkpoints + harness JSONs
    SMOKE.md
  PREREG.md
  READY_TO_FREEZE.md
```

## Do not

- Touch `../results/primary/` or `../PREREG.md`
- Run 5-seed primary before `results/PREREG.sha256`
- Claim by-construction noise/H2 wins for inferred PG
- Claim PG-inferred \(C_\mathrm{route}=8\)
