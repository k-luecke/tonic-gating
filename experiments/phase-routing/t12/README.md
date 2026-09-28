# T12 — The Loop Budget

| Field | Value |
|-------|--------|
| **Owner** | Kyle (ZetaZero) |
| **Date** | 2026-09-23 PT |
| **Status** | **CLOSED mixed** 2026-09-23 PT — H3 support; H1/H2/H4 null; PREREG `bf757ee4…`; no T12.2 |
| **Folder** | `/workspace/scp/experiments/phase-routing/t12/` |
| **Mission doc** | [`docs/T12_HANDOFF.md`](docs/T12_HANDOFF.md) |
| **PREREG** | [`PREREG.md`](PREREG.md) / [`PREREG.sha256`](PREREG.sha256) — **FROZEN** `bf757ee4…` (draft tip `8e04b801…`) |
| **Parent** | T11.3 CLOSED null (window) — PREREG `27ab4a9573ec8c01ed043d72ef6c83ba47fdd4a764a373acb7453381bb9bd442` |

## Mission (one line)

First job a persistent coordinator can do that lookup cannot = **loop budget**.

| H | Short |
|---|--------|
| **H1** | Regime budgeting |
| **H2** | Trajectory budgeting vs ACT / convergence |
| **H3** | Overthinking is regime-dependent |
| **H4** | Persistence under silenced estimate (zeroed/held at **controller input**; estimate may still be computed for disclosure) |

**Null:** fixed/ACT matches; or T\* same across regimes; or SD no better than stateless.

**Stopping rule:** one preregistered run, **five** seeds; **no T12.2** unless freeze finds a gate defect.

## Hard locks

- Do **not** retune C₃ / hum / Affinity `__all__`.
- Do **not** touch T11 / T11.2 / T11.3 PREREG hashes.
- Hum **n=2** (easy/hard). High-pass every hum drive; amplitude readout never phase-gated; T8 shift regression lock.

## Tickets (T0–T8 in order)

| Ticket | Acceptance |
|--------|------------|
| **T0** | **PASS** 2026-09-23 — receipt `results/T0_HUM_REPRODUCE.md` (T_hold=39.0 in 41±5; n=2/HP/amp; Affinity/C₃ untouched). Re-run: `python3 scripts/run_t0_hum.py`. |
| **T1** | **PASS** 2026-09-23 — receipt `results/t1/T1_SWEEP_RECEIPT.md` (H1 T*=4 vs 10; H3 easy OT + hard not-yet; ACT+Convergence stubs). Re-run: `python3 scripts/run_t1_smoke.py`. No train / no freeze. |
| **T2** | **PASS** 2026-09-23 — disturbance grid (clean/noise/shift/silence) + H4 silencing (zero/hold at controller input) + parity harness; silence→prior depth follows Fixed/ACT prior (=7 after Option-1). Receipt `results/t2/`. Re-run: `python3 scripts/run_t2_smoke.py`. |
| **T3** | **PASS** 2026-09-23 — C̄=7.0±0.25 locked; ε=0.05; ACT centers **4/10** (=T1 T*; soft-8 removed); CONV τ 1.335/3.337; Fixed T=7; BP gap=6. Receipt `results/t3/`. Re-run: `python3 scripts/run_t3_calibrate.py`. |
| **T4** | **PASS** 2026-09-23 — ACC, C̄, LPC, OT(T), BP 4/10, SD, T_b on smoke (+ T8 regression lock PASS) under C̄=7 locks. Receipt `results/t4/`. Re-run: `python3 scripts/run_t4_measures.py`. |
| **T5** | **PASS** 2026-09-23 — Rule-19 gates PASS; `PREREG.sha256=bf757ee4…`. Re-run: `python3 scripts/freeze_prereg.py` (status) / `--authorize` (emit). |
| **T6** | **PASS** 2026-09-23 — primary × 5 seeds `20260921..20260925`; receipts `results/t6/`. Affinity untouched; gate defect=false. |
| **T7** | **PASS** 2026-09-23 — `results/T12_RESULTS.md` + inventory; regression locks green (via T6). |
| **T8** | **PASS / CLOSED mixed** 2026-09-23 — `docs/T12_CLOSING.md`; no T12.2. |

## Layout

```text
t12/
  README.md
  ARTIFACT_INVENTORY.md
  docs/T12_HANDOFF.md
  docs/T12_PREREG_DRAFT.md
  imports/                  ← hum_measure symlink + T7 recovered links + PATHS.md
  generators/easy_hard.py   ← T1 easy/hard
  generators/disturbance.py ← T2 clean/noise/shift/silence grid
  harness/parity.py         ← T2 parity (same seed → same seqs)
  model/recurrent_toy.py
  controllers/              ← ACT/Conv/Fixed/Coordinator/Stateless + silencing + calibration
  measures/emit.py          ← T4 §4.1 emitters
  results/t0..t4/
  scripts/run_t{0,1,2,3,4}_*.py
  scripts/freeze_prereg.py  ← T5 (Rule-19 gated)
  scripts/rule19_freeze_gates.py
```

## T7 generator search (2026-09-23)

| Location | Result |
|----------|--------|
| Inside `/workspace/scp` | **No** T7 generator source tree |
| On box | **FOUND** — `/workspace/phase-routing/t8/_recovered/` |
| Linked here | `imports/t7_phase_routing_recovered`, `imports/t7_routing_freeze_recovered` |

## T12b parallel track

Routing-only evidence gaps live under sibling [`../t12b/`](../t12b/) — **greenlit 2026-09-23**. Does not block T12.

## Smoke commands

```bash
cd /workspace/scp/experiments/phase-routing/t12
python3 scripts/run_t0_hum.py
python3 scripts/run_t1_smoke.py
python3 scripts/run_t2_smoke.py
python3 scripts/run_t3_calibrate.py
python3 scripts/run_t4_measures.py
# freeze still refuses:
python3 scripts/freeze_prereg.py            # status / gates
python3 scripts/freeze_prereg.py --authorize  # emit iff PASS
```

## Locked calibration (T3 — Option 1 / T1 budgets)

| Param | Value |
|-------|------:|
| C̄ target | **7 ± 0.25** |
| ACT prior_mean / easy / hard | **7 / 4 / 10** |
| CONV ε / τ_easy / τ_hard | 0.05 / 1.335 / 3.337 |
| Fixed T | **7** |
| T1 T*_easy / T*_hard | **4 / 10** (locks = measurement; soft-hard-to-8 removed) |

Draft content diff: [`docs/DIFF_78091e84_to_current.md`](docs/DIFF_78091e84_to_current.md) (`78091e84…` → `99afbb3b…` → `8e04b801…`).

Affinity / C₃ / hum **untouched**. T12b untouched. PREREG FROZEN `bf757ee4…`. **CLOSED mixed** — H3 **support** (OT easy degrades 5/5, hard 0/5; T* 4 vs 12); H1/H2/H4 **null** (ACT/Fixed match; traj ties ACT; SD 0/5). See `docs/T12_CLOSING.md` + `results/T12_RESULTS.md`. No T12.2.
