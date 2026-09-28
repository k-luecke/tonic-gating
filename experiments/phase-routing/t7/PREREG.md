# PREREG — T7: Inferred φ_ref Phase Gate vs LR / LR-goal / HR1 (Harder Task)

| Field | Value |
|-------|--------|
| **Title** | T7 — Inferred φ_ref Phase Gate on order-3 recurrence (follow-on to primary) |
| **Date** | 2026-09-21 / 22 (PT) |
| **Owner** | Kyle (ZetaZero) |
| **Status** | **FROZEN** — `results/PREREG.sha256` written; multi-seed primary authorized |
| **Relation** | Separately preregistered follow-on. Parent primary under `../results/primary/` is **FROZEN** — not edited by this experiment. |

This document freezes T7 **before** any multi-seed primary comparison. Exploratory one-seed numbers in §disclosure are **non-counting**.

---

## 1. Hypotheses (exact)

**H1:** On the harder task, PG-inferred matches LR clean quality (seed-mean \|L_PG−L_LR\| ≤ 0.1 nats; LR specialization ≥4/8 dominant), at lower or comparable documented native routing cost than LR's 1024 MACs (document PG-inferred cost; cost clause is informational if ops aren't comparable — still report native units, never combine), and degrades less than LR under primary cells noise σ=0.3 and shift δ=0.3 under the same dual bootstrap rule as parent PREREG (CI excludes 0 AND point > width).

**Also report (not collapsing into H1):** PG-inferred vs **LR-goal** on the same robustness cells — if PG≈LR-goal, inference recovered the goal; if not, say so.

**H2:** On switching split, T_sw(PG-inferred) < T_sw(LR) under same dual bootstrap. LR-goal T_sw reported for confound context; H2 comparator is PG-inferred vs LR only.

**H0/mixed:** same vocabulary as parent. Failures reported in those words.

**Critical labeling:** Unlike primary, noise and H2 are **NOT** by construction for PG-inferred (φ_ref comes from noised hidden). Cost is whatever the phase head costs — label honestly.

---

## 2. System specification (frozen)

### 2.1 Data (harder than primary)

| Item | Frozen value |
|------|----------------|
| Data seed | `20260921` |
| RNG | `numpy.random.PCG64` |
| Vocab | markers `0..7`; content `8..63` (56 symbols) |
| Sequence length | 128 |
| Rule | **Order-3** affine: \(x_{t+1}=(a[t]x_t+b[t]x_{t-1}+c[t]x_{t-2}+d[t]) \bmod 56\) |
| Coeffs | `a=(1×8)`; `b=(1,1,1,1,2,2,2,2)`; `c=(1,1,2,2,1,1,2,2)`; `d=(1,5,9,13,17,23,29,37)` in `data/metadata.json` |
| Flip noise | `p_flip=0.0` (machinery retained; see §disclosure — 0.15 unlearnable on this backbone) |
| Train | 256 sequences/task → `[2048,128]` |
| Heldout | 64/task → `[512,128]` |
| Switching | 512 sequences; mid-sequence distinct marker restarts new recurrence (3 content seeds after marker) |
| Manifest | `data/MANIFEST.sha256` |

### 2.2 Shared model

| Item | Frozen value |
|------|----------------|
| Architecture | 4-layer decoder, width 128, 4 heads, FFN width 256 |
| MoE layer | Layer index 2; 8 experts `128→256→128` GELU |
| Task interface | Shared **latest-marker task embedding** broadcast (parity with primary) — identical for all arms |
| Constraint | PG-inferred and LR must **not** use marker→phase / must **not** get goal at router. LR-goal gets goal **only** at router |

### 2.3 Routing arms (counting primary)

| Arm | Role | Definition |
|-----|------|------------|
| **LR** | Primary | Linear hidden→8; softmax; top-2; aux 0.01. \(C_\mathrm{route}=1024\) MACs |
| **LR-goal** | Primary | `LearnedRouterGoal`: concat(hidden, router-only goal_embedding[task])→Linear; top-2; aux 0.01. \(C_\mathrm{route}=2048\) MACs |
| **HR1** | Primary baseline | Hash `(5·token+1) mod 8`; top-1; aux=0 |
| **PG-inferred** | Primary | Fixed expert phases \(2\pi e/8\). φ_ref from `Linear(width→2)` → `atan2(sin,cos)`. Gains `max(0, cos(φ_e−φ_ref))`; top-4 like parent PG. **Forbidden:** marker → `task_phases[task_id]` for routing. Tokens ignored for φ_ref |

**Secondary only (optional one-seed reference, NOT in H1):** supplied PG (marker table) — only if cheap; do not train 5 seeds for primary.

### 2.4 PG-inferred native cost (honest)

- Phase head: `Linear(128→2)` = **256 MACs/token**
- Plus **8** half-wave cosine comparisons + normalize/top-4
- **Do NOT claim \(C_\mathrm{route}=8\)**. Report `C_route_MACs_per_token=256` and `C_route_ops=8` separately; never combine with LR MACs into one score. Cost clause of H1 is informational if units aren't comparable.

### 2.5 Training (every primary seed/arm)

| Item | Frozen value |
|------|----------------|
| Optimizer | AdamW, lr **0.002**, weight decay 0.01 |
| Schedule | Linear warmup 50 steps, then cosine down to 0.1× base |
| Steps | **1000** (retuned from planned 1200; see §disclosure) |
| Batch | **32**, shuffle with `Generator.manual_seed(seed)`, drop_last |
| Grad clip | 1.0 |
| Seeds (primary later) | **`[20260921, 20260922, 20260923, 20260924, 20260925]`** |
| Device | CPU OK; record device in metrics |

---

## 3. Disturbance model (copy parent T4)

Grid: \(\{0,\ 0.1,\ 0.3,\ 1.0\}\) for router-input noise and MoE-layer shift.

- **Noise** → routing-decision hidden **only** (experts un-noised aside from shift).
- **Shift** → every MoE-layer hidden (routing **and** expert input).
- Per-arm clean heldout MoE hidden std; shared seeded base tensors across arms.

**Primary cells:** \(\sigma_n=0.3\), \(\delta=0.3\).

---

## 4. Measures (copy parent T4)

| Symbol | Definition |
|--------|------------|
| \(L\) | Mean content-token NLL on heldout (markers excluded) |
| \(C_\mathrm{route}\) | Native routing-decision cost (see §2.4) |
| \(C_\mathrm{exp}\) | `mean_active_experts_per_token × 65536` |
| \(B\) | Load-balance CV |
| \(\Delta L\) | \(L_\mathrm{cell}-L_\mathrm{clean}\) |
| \(T_\mathrm{sw}\) | First lag after switching marker with >0.5 mass on new-task dominant set; index 0 = immediate |

**Bootstrap:** 10 000 paired-seed percentile 95% CI; dual rule (CI excludes 0 **and** point > width).

---

## 5. Primary comparison rule

1. After `t7/results/PREREG.sha256` exists, train LR, LR-goal, HR1, PG-inferred on seeds `20260921..20260925` into `t7/results/primary/` (20 runs).
2. Run harness per seed; score H1/H2 exactly as §1.
3. Report **supported**, **H0**, or **mixed**.
4. **No multi-seed before hash.** **No retune from multi-arm peeking.**

---

## 6. Rules of the road

1. Do not edit parent `../PREREG.md` or `../results/primary/`.
2. Fix parameters from single-arm smoke only.
3. Same data splits and seeded disturbances across arms within a seed/cell.
4. Cost channels stay in native units.
5. H2 is PG-inferred vs LR only; LR-goal is confound context.
6. Never claim by-construction noise/H2 wins for inferred PG.

---

## 7. Mandatory disclosure — one-seed exploratory (non-counting)

### 7.1 Calibration knobs changed from initial defaults (single-arm LR only)

| Knob | Planned | Frozen | Reason |
|------|---------|--------|--------|
| `p_flip` | 0.15 | **0.0** | With order-3 + p_flip=0.15, LR heldout stuck ≳2.2–4.5 (at/above unigram); flip noise blocked learning on width-128 backbone |
| Coeffs | large multipliers | **A=1, B half-2, C mild pattern, D=offsets** | Large coeffs unlearnable; milder order-3 still harder than primary (~0.0013) |
| Steps | 1200 | **1000** | With frozen coeffs, 1000 steps → LR ~1.26 nats, 7/8 specialization |

Hardness vs primary comes from **order-3 recurrence**, not irreducible flip entropy (disclosed). Seed-token entropy alone contributes ~0.06 nats; remaining loss is approximation error on the modular order-3 map.

Preferred band was 0.5–1 nat; smoke LR landed **1.26** (within retune acceptance 0.3–1.5).

### 7.2 One-seed smoke (seed `20260921`, steps 1000) — see also `results/SMOKE.md`

| Arm | heldout \(L\) | dom ≥0.25 | \(T_\mathrm{sw}\) | \(\Delta L\) noise 0.3 | \(\Delta L\) shift 0.3 | \(C_\mathrm{route}\) |
|-----|--------------:|----------:|------------------:|-----------------------:|-----------------------:|----------------------|
| LR | 1.255 | 7/8 | 1.814 | 0.0043 | 0.0100 | 1024 MACs |
| LR-goal | 1.260 | 8/8 | 0.344 | 0.0012 | 0.0120 | 2048 MACs |
| HR1 | 1.327 | 0/8† | 1.026† | 0.000 | 0.0038 | 0 MACs |
| PG-inferred | 1.272 | 8/8 | 4.992 | 0.0026 | 0.0066 | 256 MACs + 8 cos |

† HR1 not task-aware for specialization / H2.

Notes from smoke (non-counting):
- \|L_PG−L_LR\| ≈ 0.016 < 0.1 (clean quality tie on this seed).
- PG-inferred noise ΔL ≠ 0 (NOT by-construction).
- T_sw(PG-inferred) > T_sw(LR) on this seed (H2 not looking supported one-seed).
- LR-goal switches fastest (goal channel); confound context for H2.

**No multi-seed comparison was seen before writing this file.**

---

## 8. Freeze procedure

1. Parent reviews smoke + this PREREG.
2. Parent writes `t7/results/PREREG.sha256` = sha256(this file) + `created_iso` (PT) and anchors on Drive.
3. Only then run 5-seed primary into `t7/results/primary/`.

---

## 9. Freeze statement

Editing H1/H0/H2 thresholds or primary measures after peeking at T7 multi-seed outputs voids this preregistration for that run. Parent primary remains the frozen counting run for the supplied-control comparison; T7 is a separate inferred-φ_ref follow-on.
