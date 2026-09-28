# Handoff: T12 — The Loop Budget

| Field | Value |
|-------|--------|
| **Date** | 2026-09-23 PT |
| **Owner** | Kyle (ZetaZero) / Drift Bound Labs |
| **Status** | handoff — scaffold + PREREG **DRAFT** only; no freeze; no train |
| **Folder** | `/workspace/scp/experiments/phase-routing/t12/` |
| **Parent** | T11.3 **CLOSED null (window)** — PREREG `27ab4a9573ec8c01ed043d72ef6c83ba47fdd4a764a373acb7453381bb9bd442` |
| **Parallel** | T12b (routing-only evidence gaps) — sibling `../t12b/` **greenlit** 2026-09-23 |

Source: Kyle handoff paste 2026-09-23 ("T12 — The Loop Budget"). Encode faithfully; do not invent retunes.

**Hard locks (do not violate):**

- Do **not** retune C₃ / hum / Affinity `__all__`.
- Do **not** touch T11 / T11.2 / T11.3 PREREG hashes.
- High-pass **every** hum drive; amplitude readout **never** phase-gated.
- T8 shift regression lock remains in force.
- One preregistered run, five seeds; **no T12.2** unless freeze finds a gate defect.

---

## 0. Why T12 exists

T11.3 closed the model-level orchestration claim as **null (window)**: ScoreAffinity-windowed `hum_measure` beat the classifier on Ambiguous CSR, but a **stateless argmax** of the same affinity scored 1.000 — the lift was the observation window, not integrator state.

The first job a **persistent coordinator** can do that **lookup cannot** is the **loop budget**: how many recurrent steps to spend, as a function of regime and trajectory — not which specialist to call once.

T12 seats a recurrent-depth toy under matched compute and asks whether coordinator-budgeted depth beats fixed / ACT / convergence baselines.

---

## 1. Mission (H1–H4, Null, Stopping)

### Hypotheses

| H | Claim (plain) |
|---|----------------|
| **H1 — regime budgeting** | Coordinator sets depth from **regime** so easy regimes spend less and hard regimes spend more, improving accuracy or cost vs matched fixed depth. |
| **H2 — trajectory budgeting** | Coordinator that also uses **trajectory** (within-example state) beats ACT and convergence-threshold halt under the same ceiling. |
| **H3 — overthinking is regime-dependent** | Extra depth helps on hard / disturbed regimes and hurts or is neutral on easy / clean; OT(T) curves separate by regime. |
| **H4 — persistence under silenced estimate** | With the regime estimate **silenced**, coordinator + persistence (hum) still budgets better than a **stateless regime** readout of the same features. |

### Null (any one closes support)

- Fixed depth and/or ACT **match** coordinator on primary comparisons; **or**
- Optimal depth **T\*** is the **same** across regimes (no regime budgeting to do); **or**
- Under silencing, coordinator / SD is **no better** than the **stateless regime** arm.

### Stopping rule

- **One** preregistered primary run, **five** seeds.
- **No T12.2** unless the freeze script finds a **gate defect** (hollow freeze / broken parity / missing regression lock). Retune-by-series is forbidden.

### Out of scope

- MoE / T13 expert routing revisit
- Real pages / BrowserGym / MiniWoB as primary task
- Plasticity / online weight change
- Hardware / edge deployment

### Compute budget

- Looped model ≈ **4-layer-equivalent**, width **128**
- Train **&lt; 1 h** wall
- Full arm sweep **&lt; 6 GPU-h** (or CPU-equivalent disclosed)

---

## 2. Prior work (cite briefly)

| Line | What it locked |
|------|----------------|
| **T7** | Deterministic recurrence LM data + 4×128 MoE/router harness; harder order-3 follow-on. Recovered on box under `/workspace/phase-routing/t8/_recovered/`. |
| **T8** | Hum persistence / T_hold / shift cells; PREREG `d975d96c…`; rerun 2026-09-23. **Shift regression lock.** GNURadio owns hum confirm. |
| **T11 / T11.2 / T11.3** | Model-level orchestration → **CLOSED null (window)** at T11.3. Frozen C₃ / hum / Affinity `__all__`. PREREGs **untouched**. |
| ACT / convergence literature | Halt policies T12 compares against (Graves ACT; residual-norm convergence). |

---

## 3. System (task, model, arms)

### 3.1 Task — T7 modified, easy/hard split

- Start from T7-style deterministic recurrence sequences (recovered generator).
- **Modify** into an explicit **easy / hard** difficulty split (T1 ticket).
- Disturbance grid (noise / shift / silence / switching as locked at freeze) applied with **parity** across arms.
- Seeds: **5** (lock list at freeze; draft placeholder seeds follow T8/T11 family `20260921…` unless Kyle overrides).

### 3.2 Model — prelude / recurrent / coda

```text
e  →  Prelude  →  h_0
h_t ← A·h_{t-1} + B·e + Block(h_{t-1}, e)     # ρ(linear) < 1
     + loop-index embedding
h_T → Coda → logits / prediction
```

| Item | Draft lock |
|------|------------|
| Width | 128 |
| `T_max` | 12 |
| Train depth | **random depth** then freeze policy / evaluate under arms |
| Convergence metric | \(\|h_t - h_{t-1}\| / \|h_t\|\) |

Do **not** implement in this scaffold turn.

### 3.3 Arms

| Arm | Definition (draft) |
|-----|---------------------|
| **Fixed T=6** | Constant depth near calibration target \(\bar C = 6\) |
| **Fixed sweep** | Sweep T ∈ {1…T_max} — **measurement**, not tuning |
| **ACT** | Learned halt (Adaptive Computation Time) |
| **Convergence threshold** | Halt when relative state change &lt; ε |
| **Coordinator regime** | Hum / coordinator budgets depth from **regime** estimate |
| **Coordinator regime+traj** | Regime + within-example **trajectory** features |
| **Oracle** | Ideal depth / regime (report-only ceiling; not support) |
| **Stateless regime** | Instantaneous regime readout — **H4** control under silenced estimate |

### 3.4 Silencing, disturbance, parity

- **Silencing:** zero or hold the regime estimate channel for H4 cells (exact recipe at freeze).
- **Disturbance grid:** carry T7/T8 spirit (noise, shift, clean) with T12 easy/hard factor; lock cells before freeze.
- **Parity:** same init family, same data draws per seed, matched compute accounting (loops × width, not parameter-count games).
- **Hum drive rules:** high-pass **every** hum drive; amplitude readout **never** phase-gated; T8 **shift regression lock** must pass before counting.

### 3.5 Calibration

- Target mean compute \(\bar C = 6 \pm 0.25\) (loop steps or equivalent).
- Exact calibration values / ε / ACT priors: **TBD at T3** (do not invent numbers in draft).

---

## 4. Measures

| Symbol | Meaning |
|--------|---------|
| **ACC** | Task accuracy / NLL as locked for the T7-derived task |
| **\(\bar C\)** | Mean compute (loops used) |
| **LPC** | Loss-per-compute / accuracy-per-compute primary efficiency |
| **OT(T)** | Overthinking curve: metric vs forced or chosen depth T, by regime |
| **BP** | Budget profile: depth distribution by regime |
| **SD** | Silenced-delta: coordinator vs stateless under silence (H4) |
| **T_b** | Budgeted depth chosen by coordinator (per example / regime) |
| **Regression locks** | T8 shift lock; hum unit imports; fixed/ACT parity smoke |
| **Primary comparisons** | H1–H4 support rules locked at freeze (seed-mean + interval rule TBD T5) |

---

## 5. Tickets T0–T8 (order)

| Ticket | Acceptance (plain) |
|--------|---------------------|
| **T0** | Tree imports `hum_measure`; T8 hum unit tests + T8 rerun persistence measurement **reproducible from this tree** (paths documented). **GNURadio owns confirming hum.** |
| **T1** | Difficulty-split generator + recurrent-depth toy + arm stubs compile/smoke (no primary). |
| **T2** | Disturbance grid + silencing + parity harness wired. |
| **T3** | Calibration pass: \(\bar C = 6\pm0.25\); lock ε / ACT / threshold values. |
| **T4** | Measures ACC, \(\bar C\), LPC, OT(T), BP, SD, T_b emit on smoke. |
| **T5** | `freeze_prereg.py` gates + DRAFT→FROZEN only after greenlight; **no** T11.x hash touch. |
| **T6** | One primary run × 5 seeds under frozen PREREG. |
| **T7** | RESULTS + inventory + regression locks reported. |
| **T8** | Close ticket: support / null / mixed in PREREG vocabulary; commit results; **no T12.2** unless gate defect. |

---

## 6. Rules

1. **Matched compute** — report loops × width honestly; Fixed sweep is measurement.
2. **Sweep = measurement** — do not select T\* from primary seeds and re-claim.
3. **Commit results at close** — artifacts under `results/` + inventory hashes.
4. **Frozen inheritance** — C₃ / hum / Affinity `__all__` / T11.x PREREGs untouched.
5. **Scaffold this turn only** — no model implementation, no training.

---

## 7. T12b — parallel track (routing-only gaps)

**Greenlit 2026-09-23.** Scaffolding in parallel under `../t12b/`.

Sibling folder: `experiments/phase-routing/t12b/` (draft README only).

Intent: close **evidence gaps** from T11.3 (routing-only) using:

- t11_3 harness
- sealed generators
- frozen hum
- stateless argmax control
- one-step memory baseline

T12b does **not** block T12 loop-budget work and must not retune frozen packs.

---

## 8. Artifact home

```text
experiments/phase-routing/t12/
  README.md
  ARTIFACT_INVENTORY.md
  docs/T12_HANDOFF.md          ← this file
  docs/T12_PREREG_DRAFT.md
  imports/                     ← hum_measure + T7 recovered links
  generators/                  ← T1 placeholder
  model/                       ← T1 placeholder
  controllers/                 ← arm placeholders
  results/.gitkeep
  scripts/freeze_prereg.py     ← T5 stub
  tests/test_hum_import.py
```
