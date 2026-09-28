# T12b Rule 19 — freeze gates

**Status:** Documented 2026-09-23 PT (scaffold + amendment + `majority_k3` secondary). Hum / C₃ / Affinity `__all__` frozen.  
T11.2 PREREG `593762c4…` and T11.3 PREREG `27ab4a95…` **untouched** (read-only reuse).

**Freeze hold:** Do **not** emit `PREREG.sha256` until **G10 in-gap perturbation** is locked/wired **and** G9 HumState path is counted — HumState wire alone does not unlock freeze.

## Gates (must ALL PASS before emitting T12b `PREREG.sha256`)

| # | Gate | Check | Fail action |
|---|------|-------|-------------|
| **G1** | **Gaps zero affinity** | For every injected gap step **except** the locked single-frame in-gap perturbation (G10), affinity vector magnitude sum (or L∞) is **exactly 0.0** — assert in `gap_inject` + freeze sampler on dry traces | Block freeze |
| **G2** | **One-step memory present** | Primary ARMS / config includes arm id `one_step_memory` (repeat-last-decision baseline) | Block freeze |
| **G3** | **Crossover prediction pre-run** | `docs/T12B_CROSSOVER_PREDICTION.md` exists, dated, cites T_hold(n=3)=65, predicts CSR pattern for G∈{3,6,12,24} **before** primary outputs exist | Block freeze |
| **G4** | **No C₃ retune** | Coupling / λ / a₀ / Affinity `__all__` hashes match frozen hum_measure receipts; no local retune files | Block freeze |
| **G5** | **T11.3 hash untouched** | `t11_3/PREREG.sha256` still `27ab4a9573ec8c01ed043d72ef6c83ba47fdd4a764a373acb7453381bb9bd442` | Block freeze |
| **G6** | **T11.2 hash untouched** | Parent T11.2 PREREG still `593762c4c19e85d1720f6b46edbb93869258c4047ad023e42e144416aaed20d1` | Block freeze |
| **G7** | **Sealed reuse not rewritten** | `generators/sealed` is symlink (or equivalent pointer) to t11_3 sealed; ALL_SEQUENCES sha256 unchanged | Block freeze |
| **G8** | **Routing-only** | Primary path does not invoke specialist GGUF exec / SCP action loop | Block freeze |
| **G9** | **Stub ≠ counted arm** | Counting / freeze path must use frozen `HumState` / `HumMeasureOrchestrator` (amplitude gate; high-pass on drive if applicable). `HumMeasureArmStub` (hold-last on zero affinity) is **dry-run only** — same mechanics as `one_step_memory` inside gaps; if stub is the counted `hum_measure` arm, null is uninformative by construction. **G9 stays.** | Block freeze |
| **G10** | **In-gap perturbation locked** | Primary: **single-frame spurious ScoreAffinity flip mid-gap** (wrong specialist peak for one frame, then zeros resume) wired in harness; dry-run asserts one-frame non-zero + surrounding zeros. Alt (doc only): mid-gap regime change in scores. **Hold freeze** until G10 passes — HumState wire alone is insufficient. Expected: `one_step_memory` latches spurious label through rest of gap; `hum_measure` resists (amplitude/HP + state). Without G10, G≤24 ≪ T_hold=65 makes bare silence uninformative (hum survives by construction; ties one_step when label shouldn’t change). | Block freeze |
| **G11** | **`majority_k3` labeled secondary present** | ARMS / config includes arm id `majority_k3` (k=3 stateless majority over ScoreAffinity argmax window), marked **labeled secondary** — **not** part of primary contrast (`hum_measure` vs `one_step_memory`). Without G11, a positive primary would force T12b.2 to separate smoothing vs held state. | Block freeze |

## Script (scaffold)

| Role | Path |
|------|------|
| Gap inject + zero assert | `scripts/gap_inject.py` |
| Arms incl. one_step_memory | `scripts/arms.py` — stub class dry-run only (G9) |
| Dry-run primary | `scripts/primary_run.py --dry-run` |
| Freeze entry | `scripts/freeze_prereg.py` — runs Rule 19; **default = gates only** (no `PREREG.sha256`); `--authorize` emit held for Kyle after PASS + draft review |
| Rule 19 runner | `scripts/rule19_freeze_gates.py` — G1–G11 incl. G9 (real HumMeasureArm) + G10 (mid-gap lure) + G11 (`majority_k3` secondary) |

```bash
# Self-test gap zeros
python3 experiments/phase-routing/t12b/scripts/gap_inject.py --self-test

# Dry-run wires arms + sealed path (no full primary)
python3 experiments/phase-routing/t12b/scripts/primary_run.py --dry-run

# Rule 19 gates only (no PREREG.sha256 unless --authorize after Kyle greenlight)
python3 experiments/phase-routing/t12b/scripts/rule19_freeze_gates.py
python3 experiments/phase-routing/t12b/scripts/freeze_prereg.py
# Do NOT pass --authorize until Kyle reviews PASS + draft diff
```

## Relation to T11.3 Rule 19

T11.3 §5.1 affinity-argmax seal (≥90% Ambiguous) already passed on this sealed batch. T12b **does not re-run** that seal as a retune; it **adds** gap/lag integrity gates above. Do not re-emit T11.3 PREREG.
