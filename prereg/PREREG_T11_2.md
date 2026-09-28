# PREREG — T11 Model-Level Orchestration (FROZEN 2026-09-23)

| Field | Value |
|-------|--------|
| **Status** | **FROZEN** 2026-09-23 PT (T11.2 Rule 19 re-seal Linux CPU) — gates 1–4 PASS on `t11_2_dev` with live specialist exec; stub cannot PASS gate 4 |
| **HANDOFF** | `t11/HANDOFF.md` sha256 `056be9628d3a07edd9fa0a2bff32b7e01ebb8fdd9a36b7b66d0ec5f7e39c8eaf` |
| **Parent line** | T8 hum H2 (persistence); T9 selection; T10 closed not informatively tested |
| **Run location** | Linux `/workspace/scp` CPU primary (Kyle reverse 2026-09-23); Windows Vulkan later comparison only. Artifacts: `experiments/phase-routing/t11_2/` |

## Arms
Hum · Context classifier · Hand rule · Hum+SCP (H5) · Oracle (ceiling, not compared)

## Hypotheses
H1–H4 orchestration; H5 combo scored **separately** (cannot rescue H1–H4 null). H0 live on H1.

### H6 (window break — harness condition, not a hum hypothesis — Kyle rewrite 2026-09-23)
Carrying state across a window break is a **harness feature** (serialize at break, reload after). The harness decides what survives. Hum gets **no special standing**.

- **Sequence type:** Window break — end context window entirely (new model call, no history). Same seeds as other types. Breaks both **inside a task** and **at a task boundary** (stale-state cost).
- **Carry modes (apply to every arm):**
  1. Hum-state — amplitude + phase per specialist + tiering (few numbers; cost = serialized byte size).
  2. Classifier-buffer — addendum memory buffer for classifier-with-memory (cost = buffer bytes).
  3. Context-carry — last N tokens re-pasted (baseline; cost = N tokens, report bytes too).
  4. Cold start — carry nothing (floor; cost = 0).
- **Question:** which carried state buys the most correct first steps after a break **per byte carried**? Does stale state hurt when the break coincides with a task change? Hum wins only if its few numbers match context-carry at a fraction of the cost — open, not guaranteed.
- **Null:** no carry mode beats cold start on M6a after cost normalization (beyond seed interval), or M6d shows carried state hurts on task-change breaks.

## Sequence types (five)
Clean · Ambiguous · Distraction · Renewal · **Window break (H6 harness condition)**

## Measures (draft — lock numbers before freeze)

| Id | Name | Definition |
|----|------|------------|
| M1 | Correct-specialist rate (CSR) | Fraction of segments where hot specialist = oracle specialist |
| M2 | Thrash count | Unnecessary switch-away-and-back events on distraction spans |
| M3 | Switch latency (T_sw) | Segments (or steps) from true regime change to first correct hot specialist |
| M4 | Renewal recall | On returning regimes, fraction where demoted specialist is correctly re-selected |
| M5 | Invalid-action rate | SCP/harness invalid actions at matched coverage (H5 only) |
| M6a | CSR first post-break step | Correct-specialist rate on first step after window break, **per arm × carry mode** |
| M6b | Bytes (or tokens) carried | Size of serialized/re-pasted state for that mode |
| M6c | CSR per byte | M6a / M6b — primary efficiency comparison across carry modes (undefined at cold start; report M6a alone vs cold) |
| M6d | Stale-break penalty | Drop in M6a when break = task change vs inside-task break (same seeds) |

**Primary comparisons (draft):**

- **H1:** M1 on Ambiguous — hum ≥ classifier; hum > hand rule. Sanity: Clean → all tie within seed interval.
- **H2:** M2 on Distraction — hum lower than classifier and hand by more than the seed interval. Operating persistence **T_hold(n=3) = 65**; sealed distraction spans must sit below it ({6,8,9} do).
- **H3:** M3 — \|latency_hum − latency_classifier\| ≤ 2 segments.
- **Pre-freeze expectation (Kyle 2026-09-23):** H2 likely passes easily at sealed 6–9-token distractions vs T_hold=65; H3 is where the same long hold may fail (late switch on Clean). If so → persistence trade-off, not a surprise.
- **H4:** M4 on Renewal — report per arm; support rule locked before freeze (propose hum ≥ classifier).
- **H5:** vs hum-alone under Ambiguous/Distraction: M1 rises, M2 falls; M5 no worse at matched coverage. **Scored separately; cannot rescue H1–H4 null.**
- **H6:** compare M6a and M6c across **all arms × carry modes**. Hum-state wins only if few numbers ≈ context-carry first-step CSR at a fraction of byte/token cost. Report M6d. Hum has no special standing.

Cost units (switch latency / reload / cold-start tokens) stay **separate** from H1–H6 success/fail.

## Thresholds (TBD — draft proposals only)

- Seeds: propose `[20260923, 20260924, 20260925, 20260926, 20260927]` (window-break uses the **same** seeds)
- Thrash seed interval: …
- H3 latency gap ≤ 2 segments (locked from HANDOFF)
- Tiering k (warm), m (cold): …
- λ = **0.02548046875**, **ρ_eff = effective_linear_ρ = 0.9891373214** (n=3 rebuilt C₃), a₀ = **0.1**, ρ(C) ≈ **0.015**, amp_gate_threshold = **0.05** (T8 PREREG sha `d975d96c…`; blob `imports/HUM_PARAMS.json`)
- **T_hold(n=3) = 65** (operating: coupled rebuilt C₃, unforced excess-amp 1/e after 64-token drive→silence; receipt `docs/T11_N3_THOLD_RECEIPT.md`). Homogeneous diagnostic = 39 ≈ 1/λ. **Not** T8 n=8 MoE/diag numbers.
- Distraction vs operating T_hold(n=3): sealed spans n_tokens ∈ {6,8,9} (6–9 tokens) are all well inside 65 → **PASS** (live freeze-relevant). Receipt: `docs/T11_N3_THOLD_RECEIPT.md` (sha256 `3c60bb8e…`). PREREG **FROZEN** 2026-09-23 PT.
- n=3 ρ check: ρ(C₃)<1 and effective_linear_ρ<1 → **PASS** (`imports/hum_measure/COUPLING_HASHES.json` `n3_*_check`; C₃ sha `40e3c0c9…`)
- Noisy return-to-idle (bounded stubbornness): **PASS** — T_return max=1 within horizon 512; receipt `docs/T11_N3_NOISY_RETURN_RECEIPT.md` sha `b436b1f54efd…`
- Context-carry N: … (propose N ∈ {32, 128} as two cells, or single N locked at smoke)
- Classifier-buffer size: … (from addendum; hash with classifier)
- Hum-state bytes: report measured sizeof(amplitude, phase, tier) × n_specialists
- H6: no numeric win threshold for hum alone — compare M6c ranks; null as above

## Coupled persistence mechanism (Kyle 2026-09-23, pre-freeze)

Homogeneous (leak / diagonal alone) hold ≈ **39 ≈ 1/λ** as designed. With rebuilt C₃, operating **T_hold(n=3) = 65** and **ρ_eff = effective_linear_ρ = ρ((1−λ)(I+C₃)) = 0.9891373214**.

**Mechanism:** coupled hold time goes roughly as **1/(1 − ρ_eff)** (continuous scale ≈ 92.1; discrete excess-1/e probe after finite drive measures **65** — same order, not exact equality). Coupling has pushed ρ_eff close enough to one that **ρ_eff, not λ, sets the time constant**. This is why T8's λ sweep moved hold by only ~15% for a fourfold change in λ: the leak was never the dominant term once coupling was on.

**Paper-2 claim rewrite:** “tunable leak” → **“tunable effective spectral radius, of which the leak is one part”** (truer and more useful). Record **ρ_eff beside λ** as a frozen operating parameter.

### Pre-freeze predictions (persistence trade-off)

A 65-token hold is longer than a typical segment, so on Clean sequences the coordinator will be **slow to switch** when the task genuinely changes; the late-switch cost measure (M3 / H3) exists for exactly this.

- **H2** likely **passes easily** at sealed 6–9-token distractions against a 65-token hold.
- **H3** is where the **same** property may **fail**.
- If both come out that way, that is the persistence trade-off measured at the model level — **not a surprise**.

### Noisy return-to-idle (bounded stubbornness)

ρ_eff ≈ 0.989 is close enough to one that drive noise could in principle push holds longer. **Single-system check** (n=3): after drive + noisy evidence + silence, state **returns to idle within a bounded time** (T_return max = 1 token within horizon 512 across σ∈{0.02,0.05,0.1} × 5 seeds) → **PASS**. Stubbornness is not unbounded. Receipt: `docs/T11_N3_NOISY_RETURN_RECEIPT.md` (sha256 `b436b1f54efd0c985d24eac3251452c080eb6ac8fce65da4f4004c21d89512c7`).

## T8 coordinator gate
Kyle authorized 2026-09-23 (tightened): reimpl HumState from frozen T8 PREREG §2.5 (`d975d96c…`). Acceptance on T8 params **before any T11 sequence**.

**Clause (freeze this):** coordinator reproduces T8 within X, where **X = |T_hold − 41| ≤ 5 tokens and 1/λ scaling on ≥2 λ**.

**Result 2026-09-23: PASS** — T_hold=39 (|Δ|=2); λ×{½,1,2} → T_hold {78,39,20}; hum_state sha256 `03ab651db94e87e99e9d560b2e374ccc9dd6e90251422b2364b007089cfd0637`. Probe = dynamics excess-amp 1/e after drive→silence (homogeneous); not original MoE top-1 H2 harness (missing). See `hum/results/ACCEPTANCE_RESULT.md`.
**Status:** blocked — Linux box has params-only; Windows `DESKTOP-C6EAR7I` offline.

## Hand-rule gate
**SIGNED 2026-09-23 by Kyle Luecke.** Canonical text: `t11/HAND_RULE.md` (sha256 `28a3d07295ddf6558261b2f8c5065d6cf4934bc2c8457ef45cbbf081bc281334`). First-32 tokens, case-insensitive whole-word; refuse list before specialist rule; abstention counted separately. Written before T2 data.

## Freeze gates
1. Content-defined oracle specialist (no planted tag shared with gate inputs).
2. Downstream identity lock: arms 1 vs 4 differ only in reference source.
3. Classifier calibrated on disjoint split; hand rule written pre-data (**Kyle-signed 2026-09-23**, `HAND_RULE.md` hashed).
4. Every arm can fail its clause (T10 lesson).
5. T8 acceptance-test-passing reimpl (Kyle-authorized 2026-09-23) committed **before any T11 sequence**; PREREG states “reproduces T8 within X” with X frozen; fail → wait for original.
6. A3 text-only: **DONE keep A3** — 9/10 vs Qwen2.5-1.5B+SCP-D4 6/10; no mmproj; receipt `A3_TEXT_ONLY_PROBE_RECEIPT.md` (2026-09-23). Browser = A3-Qwen3.5-2B text-only.
7. H6 present as harness carry condition (3 modes + cold, all arms, per-byte score) — not as a hum-only persistence claim.

## Primary
Not started. **PREREG frozen 2026-09-23 PT** — launch per sealed batch + frozen arms.


---

## T11.2 Rule 19 re-seal (2026-09-23 PT)

- **Gates:** Rule 19 mechanical freeze (`t11/scripts/rule19_freeze_gates.py`); stub/dry_run/placeholder **cannot** PASS gate 4.
- **Batch:** `experiments/phase-routing/t11/generators/t11_2_dev` (ALL_SEQUENCES sha `fb5af293…`); sealed-v1 untouched.
- **Gate 4 live:** 1 ep/family under SCP (ARM_E_D4) via llama.cpp CPU + BrowserGym + MiniWoB `7fd85d71…`; env rewards present; `exec_mode=live_scp_cpu`.
- **Backend:** `CPU` (ngl=0); llama-cli `0.4.1-dev` commit `66fba63`.
- **Frozen:** Hum / AffinityProvider / C₃ untouched; ScoreAffinityProvider wire stays.
- **Emit path:** only `t11/scripts/freeze_prereg.py --authorize` (no bare sha256sum).
- **Re-sealed_pt:** 2026-09-23 15:38:25 PT
