# Phase-Gated Expert Routing

Exploratory, pre-registration implementation of T0–T4 for Kyle / ZetaZero. **No
`PREREG.md` has been written or frozen and no multi-seed evidence is claimed.**

## Status

| Ticket | Implemented scope |
|---|---|
| T0 | deterministic data, manifest, environment, tests, pipeline |
| T1 | learned top-2 router (LR), Switch-style auxiliary loss |
| T2 | parameter-free deterministic token hash top-1 (HR1) and two-hash top-2 (HR2) |
| T3 | fixed expert phases, learnable task-phase table, positive half-wave-cosine Phase Gate (PG) |
| T4 | common clean/noise/shift/dropout/switching evaluation harness |

All reported runs use the one exploratory seed `20260921`, 600 steps, batch 32,
and otherwise common initialization/training rules. Future primary runs must retain
these seed/init rules and are not to begin until a preregistration is frozen.

## Data and model

The vocabulary has marker tokens `0..7` and a shared content alphabet `8..63`.
Each 128-token stationary sequence starts with its task marker and follows
`x_next = (m[t] x + b[t]) mod 56`. The switching split inserts a distinct marker
and restarts the new recurrence. Exact rules and shapes are in `data/metadata.json`.

The shared model is a 4-layer, width-128, 4-head decoder. Layer 3 replaces its FFN
with eight `128 -> 256 -> 128` GELU experts. Known task identity is represented by
a learned 8 x 128 embedding of the most recent marker and broadcast into every
token representation. This path is identical for every routing arm and updates at
the switching marker.

### Disclosed specialization repair

The original marker-only LR run learned the language (heldout about 0.0015 versus
per-task unigram about 4.0) but its top-1 utilization was nearly task-independent:
0/8 tasks reached dominant share 0.25. The task was known at sequence start, but
two attention blocks did not reliably preserve that identity at the MoE interface.
The shared most-recent-marker embedding above repaired the interface without
changing any routing-arm semantics or data. The original LR auxiliary weight 0.01
was retained; no arm comparison was used to tune it.

| Exploratory trial | Steps | Loss | dominant tasks >=0.25 | Outcome |
|---|---:|---:|---:|---|
| Original marker-only LR | 600 | ~0.0015 | 0/8 | failed, disclosed JSON/log |
| Shared task-embedding LR | 600 | 0.001323 | 8/8 | accepted |

See `data/CHANGELOG.txt` and
`results/exploratory/t1_initial_interface_failure_seed20260921.json`.

## Routing arms

- **LR:** linear hidden-state router, softmax normalized top-2 gates, Switch-style
  `E * sum(load * importance)` auxiliary loss with final weight **0.01**.
- **HR1:** fixed `(5*token+1) mod 8`, one expert, no parameters, aux=0.
- **HR2:** HR1 plus fixed `(3*token+2) mod 8`; the hashes cannot collide modulo 8,
  equal gates, no parameters, aux=0.
- **PG:** expert phases are fixed every pi/4; the eight task phases are learnable.
  Gain is `max(0, cos(phi_e-phi_task))`, normalized over only positive gains.
  At exact alignment the target expert has gain 1, the +/-pi/2 boundaries and the
  opposite expert have gain 0, and experts in the negative half-plane (including
  +/-3pi/4) have gain 0. Generic phase has four positive experts; exact alignment
  has three. Floating-point learned boundary offsets can make individual tasks have
  three or four, giving the measured mean 3.875 here.

## Final one-seed exploratory metrics

All eight per-task losses for all arms are well below their unigram baselines.
Hash losses are within 0.00011 nats of LR (well inside the 0.1 implementation check).
Router parameter counts are LR 1,032; HR1/HR2 0; PG 8; total counts include the
shared model.

| Arm | heldout L | active experts/token | total parameters | CPU train time |
|---|---:|---:|---:|---:|
| LR | 0.001323 | 2.000 | 1,026,440 | 40.41 s |
| HR1 | 0.001298 | 1.000 | 1,025,408 | 32.93 s |
| HR2 | 0.001431 | 2.000 | 1,025,408 | 32.93 s |
| PG | 0.001377 | 3.875 | 1,025,416 | 37.56 s |

## T4 harness

`harness.py` measures a clean layer-3 hidden standard deviation once per checkpoint,
then reuses it for 0.5/1.0-standard-deviation Gaussian noise and fixed-direction
shift. It also evaluates each single-expert dropout and the switching split. Every
arm/seed/cell has a JSON under `results/exploratory/harness/`.

Exact fields: `L`, `L_per_task`, `C_route` (mean positive dispatches/token), `C_exp`
(experts with nonzero aggregate dispatch), dispatch load-balance CV, robustness
`exp(-(L_cell-L_clean))`, `delta_L_drop`, and `T_sw`. Per-task dominant profiles are
derived only from clean heldout dispatch mass. `T_sw` is the first lag from the new
marker with strictly >0.5 dispatch mass on the new task's dominant profile.

Clean sample:

| Arm | C_route | C_exp | load CV | T_sw mean | mean delta L drop |
|---|---:|---:|---:|---:|---:|
| LR | 2.000 | 8 | 0.200 | 1.488 | 0.242763 |
| HR1 | 1.000 | 8 | 0.029 | 0.507 | 0.074532 |
| HR2 | 2.000 | 8 | 0.029 | 0.738 | 0.000258 |
| PG | 3.875 | 8 | 0.005 | 0.000 | 0.003073 |

HR switching times are chance crossings of token-hash profiles, not evidence of
task-aware switching; success fractions and the exact definition are stored in JSON.

## Reproduction

```bash
cd /workspace/phase-routing
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./run.sh data       # regenerate and verify MANIFEST
./run.sh test       # pytest -q
./run.sh lr         # or hr1, hr2, pg
./run.sh harness
./run.sh all        # data, tests, four arms sequentially, harness
```

`SEED`, `STEPS`, and `PYTHON` are configurable environment variables. The central
`phase_routing.factory.make_model` accepts `lr`, `hr`/`hr1`, `hr2`, and `pg`.

## Key artifacts

- `phase_routing/model.py`, `hash_router.py`, `phase_gate.py`, `factory.py`
- `train.py`: common single-seed trainer
- `harness.py`: T4 evaluation
- `tests/`: data, LR, hash, PG geometry/factory tests
- `results/exploratory/*metrics*.json` and checkpoints
- `results/exploratory/harness/*.json`
- `data/MANIFEST.sha256`, `metadata.json`, `CHANGELOG.txt`
