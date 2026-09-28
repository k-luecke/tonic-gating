# PREREG DRAFT — T12 The Loop Budget

| Field | Value |
|-------|--------|
| **Status** | **DRAFT** — do **not** freeze; do **not** emit `PREREG.sha256` |
| **Date** | 2026-09-23 PT |
| **Owner** | Kyle (ZetaZero) |
| **Parent** | T11.3 **CLOSED null (window)** — PREREG `27ab4a9573ec8c01ed043d72ef6c83ba47fdd4a764a373acb7453381bb9bd442` |
| **Artifact home** | `experiments/phase-routing/t12/` |
| **Run location** | Linux `/workspace/scp` (scaffold + draft this turn; no primary) |
| **Parallel** | T12b routing-only gaps — `../t12b/` **greenlit** 2026-09-23 (scaffolding in parallel) |

> **DRAFT.** Sections below lock **intent** for freeze review (T5). Emitting a hash before gates + Kyle greenlight voids the draft for that run.
>
> **Do not** retune C₃ / hum / Affinity `__all__`. **Do not** touch T11 / T11.2 / T11.3 PREREG hashes.
>
> **Amendment 2026-09-23 PT (Kyle + GNURadio):** H4 operational silencing (zero/hold at controller input); T1 acceptance checks before T2–T4; Affinity/hum locks unchanged (n=2, HP every drive, amplitude never phase-gated).
>
> **Amendment 2026-09-23 PT (Kyle + GNURadio — Option 1 / T1 budgets):** Recalibrate C̄ target **6→7 ±0.25**; Fixed **T=7**; ACT/BP centers **4/10** matching T1 T*; **remove soft-hard-to-8** deviation. Affinity/C₃/hum still frozen. Status remains **DRAFT** (no `PREREG.sha256`).

---

## 1. Mission (H1–H4 + Null + Stopping)

### 1.1 One-sentence mission

First job a **persistent coordinator** can do that **lookup cannot** = the **loop budget** (how many recurrent steps to spend).

### 1.2 Hypotheses

| H | Primary claim |
|---|----------------|
| **H1 Regime budgeting** | Coordinator depth from regime improves ACC and/or LPC vs Fixed T=7 under matched \(\bar C\). Easy spends less; hard spends more (BP separates). |
| **H2 Trajectory budgeting** | Coordinator **regime+traj** beats **ACT** and **Convergence threshold** on primary LPC/ACC under same `T_max`. |
| **H3 Overthinking regime-dependent** | OT(T) shows extra depth helps hard/disturbed and is neutral/hurtful on easy/clean; separation exceeds seed interval (rule locked at freeze). |
| **H4 Persistence under silenced estimate** | With regime estimate **silenced at the controller input** (operational: may still be computed for disclosure/receipts, but **zeroed or held** so it is **not fed into the budget decision**), coordinator (hum persistence) beats **Stateless regime** on SD / ACC (same features). |

### 1.3 Null (any clause → claim not supported)

1. Fixed and/or ACT **match** coordinator on primary comparisons; **or**
2. Optimal **T\*** is **the same** across regimes; **or**
3. Under silencing (estimate zeroed/held at controller input), SD shows coordinator **no better** than Stateless regime.

### 1.4 Stopping rule

- **One** preregistered primary run, **five** seeds.
- **No T12.2** unless freeze finds a **gate defect**.
- Failures reported as null / mixed in this vocabulary — no retune series.

### 1.5 Out of scope

MoE T13, real pages, plasticity, hardware.

### 1.6 Compute ceiling

Looped ≈4-layer-eq width 128; train &lt;1 h; sweep &lt;6 GPU-h (disclose CPU if used).

---

## 2. Where this comes from (boundary)

T11 series closed null (window): wherever the regime is readable from current evidence, a stateless read is optimal; persistence adds lag and forgetting when evidence is always present. Remaining hypothesis: decisions an instantaneous read cannot make — loop budget (trajectory/regime depth) and evidence gaps (T12b).

T8: always-on state holds a regime after control stops; high-pass every drive; amplitude readout never phase-gated. T8 shift is a regression lock.

Parent close: T11.3 PREREG `27ab4a9573ec8c01ed043d72ef6c83ba47fdd4a764a373acb7453381bb9bd442`.

---

## 3. System

### 3.1 Task (T7 modified — easy/hard)

| Item | Draft lock |
|------|------------|
| Base | T7-style deterministic recurrence LM (recovered: `imports/t7_*_recovered`) |
| Split | **Easy / hard** difficulty (T1 adapts generator; exact coeffs at freeze) |
| Disturbance grid | Clean + noise + **shift** (T8 regression) + silence cells for H4; exact σ/δ at freeze |
| Parity | Same seed → same sequences for all arms; matched init family |
| Seeds | **5** (list locked at freeze; draft intent: T8/T11 family unless overridden) |

### 3.2 Model (prelude / recurrent / coda)

| Item | Draft lock |
|------|------------|
| Structure | Prelude → recurrent block → coda |
| Update | \(h \leftarrow A\cdot h + B\cdot e + \mathrm{Block}(h,e)\) with \(\rho(\mathrm{linear}) &lt; 1\) |
| Loop index | Embedding of step \(t\) into block |
| Train | Random depth train, then freeze; evaluate under arms |
| `T_max` | **12** |
| Width | **128** |
| Convergence | \(\|h_t-h_{t-1}\|/\|h_t\|\) |

### 3.3 Arms

| Arm | Role |
|-----|------|
| Fixed T=7 | Matched-mean baseline near calibration |
| Fixed sweep | Measurement sweep over T |
| ACT | Learned halt |
| Convergence threshold | Halt on relative Δh |
| Coordinator regime | Regime-budgeted depth |
| Coordinator regime+traj | Regime + trajectory |
| Oracle | Report-only ceiling |
| Stateless regime | H4 control |

### 3.4 Silencing (H4 — operational lock, 2026-09-23 PT)

**Operational silencing (Kyle + GNURadio lock):** the regime estimate **may still be computed** for disclosure / receipts, but is **zeroed or held at the controller input** — it is **not fed into the budget decision**. Exact zero-vs-hold recipe for the held channel remains free at T5 freeze; the operational cut is at the controller input, not “delete the estimator.”

- **Stateless regime** still reads instantaneous features (H4 control).
- **Coordinator** may use persistent hum state only as allowed by freeze text; it must **not** see the live (un-silenced) estimate on H4 cells.
- **Scoring:** step-grain scoring **after the evidence point** stays (T11.3 fix). Silencing does not move the scored window.

### 3.4.1 Scoring note (T11.3 fix — retained)

Primary H4 / SD comparisons use **step-grain** metrics on steps **after the evidence point** (same boundary spirit as T11.3). Do not score pre-evidence filler as support.

### 3.5 Hum / Affinity inheritance (frozen)

| Rule | Lock |
|------|------|
| Hum size | **n = 2** (one unit per difficulty class: easy / hard) |
| High-pass | **Every** hum drive is high-passed |
| Amplitude readout | **Never** phase-gated |
| T8 shift | **Regression lock** must pass before counting |
| C₃ / hum / Affinity `__all__` | **No retune** |
| T11 / T11.2 / T11.3 PREREG | **Do not re-emit / do not edit** |

### 3.6 Calibration targets

| Target | Draft (amended 2026-09-23 PT) |
|--------|--------|
| \(\bar C\) | **7 ± 0.25** |
| Fixed T | **7** (matched-mean baseline) |
| ACT / BP centers | **easy=4 / hard=10** (= T1 T*; soft-hard-to-8 removed) |
| ε, ACT priors, threshold | **Locked at T3** (ε=0.05; ACT prior_mean=7; CONV τ_e/h≈1.335/3.337) |

### 3.7 Parity & seeds

Five seeds; matched compute accounting (loops × width); Fixed sweep results are **measurement**, not a selector for primary claims.

---

## 4. Measures, tickets, rules, priors

### 4.1 Measures

| ID | Name | Use |
|----|------|-----|
| ACC | Accuracy / task NLL | Primary quality |
| \(\bar C\) | Mean loops used | Compute |
| LPC | Loss- or acc-per-compute | Primary efficiency |
| OT(T) | Overthinking vs T by regime | H3 |
| BP | Budget profile by regime | H1 |
| SD | Silenced delta vs stateless | H4 |
| T_b | Coordinator-chosen depth | H1/H2 descriptive |
| Regression locks | T8 shift + hum import/smoke | Gate |
| Primary comparisons | H1–H4 support rules | Freeze-locked |

### 4.2 Tickets T0–T8 (acceptance bullets)

| Ticket | Acceptance |
|--------|------------|
| **T0** | `hum_measure` imports from this tree; T8 hum unit tests + T8 rerun persistence measurement reproducible via documented paths; **GNURadio confirms hum**. |
| **T1** | Easy/hard generator + recurrent toy + controller stubs smoke. **Must PASS T1 acceptance checks (§4.2.1)** before T2–T4 spend. Wire **stateless per-item baseline stubs** (ACT + Convergence threshold) in the **first T1 commit**. |
| **T2** | Disturbance + silencing + parity harness. |
| **T3** | Calibration \(\bar C=7\pm0.25\); lock ε/ACT/threshold to T1 T*=4/10; Fixed T=7. |
| **T4** | All §4.1 measures emit on smoke batch. |
| **T5** | `scripts/freeze_prereg.py` gates PASS → frozen PREREG only after greenlight. |
| **T6** | One primary × 5 seeds under frozen PREREG. |
| **T7** | RESULTS + inventory; regression locks green. |
| **T8** | Close: support/null/mixed; commit results; no T12.2 unless gate defect. |

### 4.2.1 T1 acceptance checks (Kyle greenlight — smoke only; no train, no freeze)

**Gate:** T1 smoke **must PASS** before any T2–T4 spend. Fail → stop and report; do not proceed to disturbance/calibration/measures tickets.

| Check | Requirement | Fail action |
|-------|-------------|-------------|
| **Fixed-depth sweep (easy vs hard)** | Loops **1–16**, **no controller**. Must show a **measurable easy/hard gap** in where loss bottoms out (**H1** signature). | Report FAIL; stop |
| **Overthinking signature (H3)** | Easy **degrades past its optimum** while hard **does not yet** (crossover / continued improvement on hard). | If **no crossover** / **plateau-only** → report **FAIL** and stop — **H3 untestable** |
| **Stateless baseline stubs (first T1 commit)** | Wire **stateless per-item baseline stubs** for **ACT** and **Convergence threshold** in the **first T1 commit**, not later (T11.3 lesson: baselines delayed → hollow comparisons). Stubs may be non-trained / placeholder logic that emit depth choices with the same I/O as future trained arms. | Block T1 PASS until stubs land |

**Out of T1:** no model training, no PREREG freeze, no Affinity/C₃/hum retune (n=2, HP every drive, amplitude never phase-gated — §3.5 unchanged).

### 4.3 Rules

1. **Matched compute** across arms.
2. **Sweep = measurement** (Fixed sweep does not tune primary).
3. **Commit results at close** under `results/` + `ARTIFACT_INVENTORY.md`.
4. Scaffold/draft this turn — **no training**.

### 4.4 Prior work (brief)

- **T7** recovered generator/harness (`/workspace/phase-routing/t8/_recovered/`).
- **T8** persistence + shift lock (PREREG `d975d96c…`; rerun 2026-09-23).
- **T11.3** CLOSED null (window); parent hash above.
- ACT / convergence halt baselines (literature).

### 4.5 Explicit reminders (non-negotiable)

- High-pass **every** hum drive.
- Amplitude readout **never** phase-gated.
- **T8 shift regression lock**.
- Calibration \(\bar C=7\pm0.25\); ACT/BP **4/10** (=T1 T*); Fixed **T=7**; soft-hard-to-8 **removed**.
- Parent: T11.3 CLOSED null (window) PREREG `27ab4a9573ec8c01ed043d72ef6c83ba47fdd4a764a373acb7453381bb9bd442`.

---

## Freeze gate note (T5 later)

`scripts/freeze_prereg.py` is a **stub**. It must refuse to emit `PREREG.sha256` until Rule-19-style gates (hum import, shift regression, calibration bounds, arm parity smoke) PASS and Kyle greenlights. This file remains **DRAFT** until then.
