# READY_TO_FREEZE — T7

Smoke is complete and looks sane. **Do not** create `PREREG.sha256` from this agent —
parent should hash/upload after review.

## What parent should hash / upload

1. **`t7/PREREG.md`** → write `t7/results/PREREG.sha256` containing:
   - `sha256` of `PREREG.md` bytes
   - `created_iso` in Pacific/Easter (PT)
2. Anchor that hash file on Drive (same practice as parent primary) — **no** public gist.
3. Optionally archive for provenance:
   - `t7/data/MANIFEST.sha256` (and/or full `t7/data/`)
   - `t7/results/SMOKE.md` + `t7/results/exploratory/smoke/` (non-counting)
   - source tree under `t7/` (`t7_routing/`, `train.py`, `harness.py`, `tests/`)

## After freeze

- Only then run 5-seed primary: arms `lr lr_goal hr1 pg_inferred`, seeds `20260921..20260925`, into `t7/results/primary/`.
- Do **not** retune from multi-arm peeking.
- Do **not** edit parent `../results/primary/` or `../PREREG.md`.

## Smoke sanity checklist (already done)

- [x] pytest green (21 tests)
- [x] One-seed artifacts under `results/exploratory/smoke/`
- [x] LR clean L in 0.3–1.5 with specialization ≥4/8
- [x] PG-inferred φ_ref from hidden (tests: marker-independent; noise changes gates)
- [x] No NaNs; harness wrote noise/shift/T_sw cells
- [x] PREREG candidate complete with disclosure of knob changes

## Suggested sha256 command (parent)

```bash
cd phase-routing/t7
sha256sum PREREG.md
# then write results/PREREG.sha256 with hash + created_iso PT
```
