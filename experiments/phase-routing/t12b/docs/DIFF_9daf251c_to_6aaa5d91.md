# T12b PREREG draft hop — `9daf251c…` → `6aaa5d91…`

| Field | Value |
|-------|--------|
| **PT** | 2026-09-23 19:59 PT |
| **Prior (pamphlet) draft content sha256** | `9daf251c…` (**exact prior bytes missing** on box lineage — see Reconstruction) |
| **Closest inventory prior on box** | `de838c6c09f02db35287d87cfd1b12adaa60e0ccfd37edf08c9eddd7b70ce941` (ARTIFACT_INVENTORY snapshot 2026-09-23 18:27 PT; 8185 B scaffold draft) |
| **New draft content sha256** | `6aaa5d9171dff4534650dbcc997096c205efc4480b2bd202e2e5d152e92cee4a` (git `16c7c90` first box lineage of this file) |
| **Status at hop** | Still **DRAFT** — no `PREREG.sha256` |
| **Affinity / C₃ / hum** | UNTOUCHED (`7dae877a…`) |
| **Parent T11.2 / T11.3 PREREG** | untouched |
| **Later hop** | `docs/DIFF_6aaa5d91_to_current.md` (`6aaa5d91…` → `408eaaac…`, majority_k3) — **unchanged by this file** |

## Reconstruction honesty

Box git lineage begins at `16c7c90` with draft already at `6aaa5d91…`. Exact pamphlet bytes for prefix `9daf251c` were **not** recovered from:

- `git` blobs / follow history for `T12B_PREREG_DRAFT.md`
- Drive folder `1r5kLfSwHPQgUQLhLtRmMnDTfZziIeNTD` fullText search for `9daf251c`
- Mechanical reverse of G10 amendments from `6aaa5d91…` (did not hash to `9daf251c` or `de838c6c`)

This diff therefore **reconstructs intent** of the pamphlet → `6aaa5d91…` hop from commit message, README / Rule-19 / inventory, and the content of `6aaa5d91…` itself. Treat as **documented reconstruction**, not a byte-identical unified diff against `9daf251c`.

## Why this hop (Kyle + GNURadio)

Unlock an **informative** persistence null under evidence gaps:

1. **G9 — HumState wire (counted path):** scaffold `HumMeasureArmStub` (hold-last) must **not** be the counted `hum_measure` arm. Counted path = frozen **HumState / HumMeasureOrchestrator** + ScoreAffinity L2 + amplitude gate + high-pass on drive. Stub remains dry-run only.
2. **G10 — mid-gap ScoreAffinity lure:** hold freeze until **in-gap perturbation** is locked — **not** only until HumState wire. Primary = **single-frame spurious ScoreAffinity flip mid-gap** (wrong specialist peak one frame, then zeros). Bare silence with G ≤ 24 ≪ T_hold(n=3)=65 makes hum survive by construction and **ties `one_step_memory`** when the true label should not change; lure makes hold-last latch the spurious label.
3. Document **`one_step_memory` failure mode:** latches lure through rest of gap; hum should resist via amplitude/HP + state.

## Material draft changes (reconstructed)

| Area | Pamphlet / scaffold intent | At `6aaa5d91…` |
|------|----------------------------|----------------|
| Freeze hold | HumState / G9 alone may have seemed sufficient | **Hold freeze until G10 wired** — HumState wire insufficient |
| Gap assert (G1) | Gap steps affinity exactly zero | Zero **except** locked G10 perturbation frame |
| §3.1.1 | Absent or TBD | Primary mid-gap single-frame spurious flip; alt regime-change doc-only |
| §3.1.2 | Absent | Expected one_step latch vs hum resist |
| §4.2 Null | hum vs one_step under bare silence | Null **with G10**; bare silence not informative when G ≪ 65 |
| Rule 19 | G1–G9-ish | **G10** added; G9 stays (stub ≠ counted) |
| Arms table | hum / stateless / one_step; stub risk | Explicit: stub dry-run only; counted = HumState path |
| Affinity / parents | Untouched | Untouched |

## Non-changes

- Primary contrast already **hum_measure vs one_step_memory** (unchanged through later `408eaaac…` hop).
- Gap grid G∈{3,6,12,24}, lag L∈{2,4,8}, seeds `[20260923…20260927]`, T_hold=65.
- No `majority_k3` yet (that is the **next** hop → `408eaaac…`).
- No primary run; no PREREG emit; Affinity/C₃/hum / T11.* PREREG untouched.

## Pointers

- Inventory prior sha: `de838c6c…` @ `ARTIFACT_INVENTORY.md` (18:27 PT scaffold)
- First git bytes: `6aaa5d91…` @ commit `16c7c90`
- Successor diff: `docs/DIFF_6aaa5d91_to_current.md`
