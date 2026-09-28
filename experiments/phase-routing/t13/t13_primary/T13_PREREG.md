# PREREG — T13 Looped architectures (MoE-of-loops / context slices)

| Field | Value |
|-------|--------|
| **Status** | **FROZEN** 2026-09-24 08:30 PT — Rule-19 T13 PASS; emit via rule19_freeze_gates --emit-hash |
| **Date** | 2026-09-24 ~08:30 PT (G13 measurement-procedure gate defect) |
| **Owner** | Kyle (ZetaZero) |
| **Greenlit** | Kyle + GNURadio, Project Miller 2026-09-23 ~23:18 PT — paper drafts far enough; **new line** open |
| **Line name** | **Looped architectures / MoE-of-loops with context slices** — **not** tonic gating 2.0 |
| **Artifact home** | `experiments/phase-routing/t13/` |
| **Run location** | Linux `/workspace/scp` (frozen; train → G13/G17 Acc → primary × seeds) |
| **Scale** | **`mythos_tiny`** — local first look |
| **Affinity / C₃ / hum** | **FROZEN** — Affinity `affinity_inputs.py` sha `7dae877a3975c1ba4ca264f03f572d6749cce39a114ea3adbcffc010c7852c50` |

> **FROZEN.** Rule-19 T13 freeze-blocking gates PASS. Affinity / C₃ / hum untouched. Parent T11/T12/T12b PREREGs not re-emitted. G13/G17 Acc deferred post-train (criterion locked). Sequence: train mixture (G9) → G13+G17 Acc → primary × seeds.
>
> **Do not** retune C₃ / hum / Affinity `__all__`. **Do not** touch sealed T11 / T12 / T12b `PREREG.sha256` files.
>
> **GNURadio:** re-eyes Affinity surface when this draft lands; default = **no Affinity work** on T13.
>
> **Gate defect (2026-09-24):** prior freeze `12305912…` superseded — G13 lacked Δφ sampling / aggregation / role. Procedure pinned in §4.0; re-freeze required (Affinity untouched).

---

## 0. Why T13 exists (new line)

Parent arcs are **closed**:

| Parent | Close | Implication |
|--------|-------|-------------|
| **T11.3** | null (window) | When evidence is always present, persistence does not beat lookup |
| **T12** | mixed — ACT ≈ oracle at toy | **Per-expert depth budgeting is a solved control**, not the scientific contribution |
| **T12b** | mixed — gaps support_lean vs one_step; ≈ majority_k3 | Smoothing-class under silence; not a mandate to continue tonic-gating |

T13 opens a **new line** after paper drafts are far enough (greenlit 2026-09-23 ~23:18 PT): **looped architectures** — multiple looped experts, each with **own budget** and **own context slice**, at **matched compute**. This is **not** tonic-gating continuation and **not** a T12.2 / T12b.2.

---

## 1. Related work and delta

| Cite | Role |
|------|------|
| **Mixture-of-Depths** (Raposo et al., 2024) | Token/expert routing over **depth** — compute allocated dynamically along a shared stack |
| **Mixture-of-Recursions** (Bae et al., 2025) | Shared recursive / looped computation with MoE-style feed-forward routing |

**Delta (T13):** **multiple looped experts**, each carrying **its own depth budget** **and its own context slice**, versus a **single shared loop** with MoE feed-forward, evaluated at **matched compute**.

| Axis | Shared-loop MoE-FF (MoD/MoR-class control) | T13 MoE-of-loops |
|------|--------------------------------------------|------------------|
| Loop | One shared recurrent / recursive body | **K** looped experts |
| Budget | Shared or token-routed depth | **Per-expert** depth budget |
| Context | Shared hidden / shared cache | **Per-expert context slice** (stratification) |
| Compute | Matched to formula §3 | Matched to **same** formula §3 |

**Open part:** **context-slice stratification** — whether slicing context per expert (vs sharing one loop state) is what buys **H_int** (intelligibility, §4) at matched FLOPs. Depth alone is **not** the claim (T12 already put ACT on the oracle ceiling at toy scale).

---

## 2. Coordinator / Point 1 (stateless OK)

**Loop update = leaky integrator.** Every looped expert carries **persistence** in its own state. Therefore a **coordinator above them can be stateless** (allocate / gate without its own integrator).

| Arm class | Role | Expectation |
|-----------|------|-------------|
| **Stateless allocator** | Primary coordinator control | Routes budgets / slices without hum state |
| **Hum-as-allocator** | Optional control arm | Including hum as one allocator arm vs stateless allocator is **fine as control** |
| **Hum win?** | — | **Prereg: we do not expect hum to win** (**expected-null**). Persistence already lives inside experts; adding hum on the allocator is not the primary claim. |

Honest clause: a hum > stateless allocator result would be **surprising** under this draft and would require freeze-review reinterpretation — it is **not** the support path for the primary hypothesis (**H_int**, §4).

---

## 3. Matched compute accounting (**BEFORE any arm**)

**One formula — all arms cost-match to it before arm definitions are interpreted as fair.**
**Attention accounting is chosen here — not deferred to mythos_tiny numerics, not left as equal options.**

### 3.0 Chosen primary vs secondary (locked now)

| Role | Choice | Formula |
|------|--------|---------|
| **Primary (binding; G15 / G18)** | **Per-slice attention counted** (strict for the mixture) | \(C_{\mathrm{match}} = L_{\mathrm{eff}}\times K\times W\times S + A_{\mathrm{attn}}^{\mathrm{slice}}\) with \(A_{\mathrm{attn}}^{\mathrm{slice}} = \sum_{e=1}^{K} L_e\cdot 4\,S_e\,W^{2}\) (draft QKV+proj proxy; iso-match cancels the leading 4) |
| **Secondary (report-only)** | **Full-context-equivalent attention** | \(C_{\mathrm{match}}^{\mathrm{sec}} = L_{\mathrm{eff}}\times K\times W\times S + A_{\mathrm{attn}}^{\mathrm{full}}\) with \(A_{\mathrm{attn}}^{\mathrm{full}} = L_{\mathrm{eff}}\cdot 4\,S_{\mathrm{full}}\,W^{2}\) — every arm billed as if attending over full context |

Primary H_int / support / null use **only** iso-\(C_{\mathrm{match}}\) under the **primary** line. Secondary table is descriptive. **Do not** re-pick after seeing which choice favors an arm.

### 3.1 Base symbols

| Symbol | Meaning | Draft lock |
|--------|---------|------------|
| \(L_{\mathrm{eff}}\) / \(L_{\max}\) | Effective loop steps / hard cap | **DRAFT \(L_{\max}=4\)**; ACT-style early exit allowed as **control only**; Σ depths (or shared \(L\)) iso-matched across arms |
| \(L_e\) | Depth steps for expert \(e\) | \(\sum_e L_e = L_{\mathrm{eff}}\) on MoE-of-loops arms |
| \(K\) | Experts (or MoE FF width-in-experts) | **Draft \(K=4\)** (chance for G17 = \(1/K=0.25\)); hash at freeze |
| \(W\) | Width (hidden size) | **DRAFT lock \(W=64\)** (mythos_tiny; amendable at Kyle freeze) |
| \(S_e\) | Context-slice tokens for expert \(e\) | **Equal slice** \(S_e = S_{\mathrm{full}}/K = 16\) when slicing; \(\sum_e S_e = S_{\mathrm{full}}\) |
| \(S\) / \(S_{\mathrm{full}}\) | Tokens in the FF / full-context term | **DRAFT lock \(S_{\mathrm{full}}=64\)**; shared-loop uses \(S_{\mathrm{full}}\) as one slice |

### 3.2 Per-arm primary \(C_{\mathrm{match}}\) lines (loops × experts × width + attention)

**mythos_tiny DRAFT locks (pending Kyle freeze amend):** \(K=4\), \(W=64\), \(S_{\mathrm{full}}=64\), equal \(S_e=16\), \(L_{\max}=4\), vocab \(=256\), train **2000 steps**, seeds §7. Reference depths below; iso-calibrate so every arm is within **±5%** of the slice-arm reference (G15).

| Arm | Primary \(C_{\mathrm{match}}\) algebra | Numeric (DRAFT reference) |
|-----|----------------------------------------|---------------------------|
| **`moe_of_loops_slices`** | \(\sum_e L_e\cdot(W\cdot S_e + 4\,S_e\,W^{2})\); equal \(L_e=L_{\max}=4\), \(S_e=16\) | \(4\times 4\times(64\cdot16 + 4\cdot16\cdot64^{2}) = \mathbf{4{,}210{,}688}\) — **iso reference** |
| **`shared_loop_moe_ff`** | \(L_{\mathrm{eff}}\cdot(K\cdot W\cdot S_{\mathrm{full}} + 4\,S_{\mathrm{full}}\,W^{2})\); \(L_{\mathrm{eff}}=L_{\max}=4\) | \(4\times(4\cdot64\cdot64 + 4\cdot64\cdot64^{2}) = \mathbf{4{,}259{,}840}\) (**+1.17%** vs reference; within ±5%) |
| **`moe_of_loops_shared_ctx`** | \(\sum_e L_e\cdot(W\cdot S_{\mathrm{full}}) + L_{\mathrm{eff}}\cdot 4\,S_{\mathrm{full}}\,W^{2}\) | At equal \(L_e=4\) (\(L_{\mathrm{eff}}=16\)): \(\mathbf{16{,}842{,}752}\) (overspend). **Iso-calibrated** \(L_e=1\) (\(L_{\mathrm{eff}}=4\)): \(\mathbf{4{,}210{,}688}\) (= reference) |
| **`fixed_depth_loops`** | Same algebra as the arm it calibrates; fixed \(L_e\)/\(L_{\mathrm{eff}}\) (no ACT claim) | Match paired body to within ±5% of **4,210,688** |
| **`stateless_allocator` / `hum_allocator`** | Allocator coordinator-only; **body** \(C_{\mathrm{match}}\) = paired body arm | Body = paired arm numeric (no unpaid body compute) |

**Secondary (full-context-equivalent) per arm:** replace every \(A_{\mathrm{attn}}^{\mathrm{slice}}\) with \(L_{\mathrm{eff}}\cdot 4\,S_{\mathrm{full}}\,W^{2}\) (and for slice FF keep \(\sum_e L_e W S_e\) or bill \(L_{\mathrm{eff}} K W S_{\mathrm{full}}\) as disclosed). Report secondary in a side table only.

**Accounting rules:**

1. Write primary \(C_{\mathrm{match}}\) **first**; arms are iso-\(C_{\mathrm{match}}\) under **§3.0 primary**.
2. ACT / early exit → bill **realized** \(L_{\mathrm{eff}}\) (batch mean); match others to that mean (±**5%**).
3. **No arm may claim advantage from unpaid depth, unpaid experts, unpaid context tokens, or unpaid attention.**
4. **Before hash emit:** evaluate primary \(C_{\mathrm{match}}\) **per arm** within **±5%** (G15); attention-accounting choice stated as above (G18).

Placeholder sha for freeze inventory of the formula block: `COMPUTE_FORMULA_SHA256 = <emit-at-freeze>`.

---

## 4. Intelligibility metric (H_int) — preregistered primary (**BEFORE any arm**)

**Primary metric = H_int.** Arms (§5) **reference** this algebra; they do **not** invent an intelligibility definition. T9 never tested intelligibility; medium harvest confirmed **selection**, not intelligibility. Selection ≠ “we can read what each expert did.”

**Intelligibility metric (H_int), preregistered definition.**

For each input \(x\), the medium aggregate is \(B(x) = \sum_e f_e(x)\cdot e^{i\varphi_e}\). Recover expert \(e\)'s contribution *from the aggregate alone*: \(\hat{c}_e(x) = \mathrm{ReLU}(\mathrm{Re}(B(x)\cdot e^{-i\varphi_e}))\). Predicted effect of removing expert \(e\): \(\widehat{\Delta L}_e(x) = L(\mathrm{readout}(B(x) - \hat{c}_e(x)\cdot e^{i\varphi_e})) - L(\mathrm{readout}(B(x)))\). Actual effect: \(\Delta L_e(x) = L\) with expert \(e\) silenced (\(f_e\) set to zero, everything else unchanged) \(- L(\mathrm{readout}(B(x)))\). H_int score: Pearson \(r\) between \(\widehat{\Delta L}_e\) and \(\Delta L_e\) over all \((e, x)\) on the held-out split, reported with a seed interval (**DRAFT lock: interval width 0.05** on the seed-mean); also per-expert \(r\) and the rank agreement of experts by mean \(\Delta L\).

**Controls.** Same granularity as the medium readout: **per expert \(e\), per example \(x\)** — every control produces \(\widehat{\Delta L}_e(x)\) scored with the same Pearson \(r(\widehat{\Delta L}_e, \Delta L_e)\) over held-out \((e,x)\).

(1) **Random-tag medium:** recovery by projection onto expert \(e\)'s tag; if its \(r\) matches phase, claim is "tagged superposition." (2) **Plain sum (floor):** no tags; \(\hat{c}_e\) = aggregate scaled by \(e\)'s gate weight. (3) **Oracle (ceiling):** \(\hat{c}_e =\) true \(f_e\cdot e^{i\varphi_e}\) from stored parts. (4) **No-medium attribution baseline:** \(\widehat{\Delta L}_e(x)\) from **activation-norm** **or** **gradient × input** for expert \(e\) on example \(x\), with **no medium at all** (no phase tags, no \(B(x)\), no medium readout). Framing: (1)/(2) test whether **phase tagging** matters; (4) tests whether the **medium/readout** matters — can **tie the primary** like T11.3 stateless argmax. **(4) is inside the verdict rule below, not a secondary-only control.**

**Verdict rule (H_int support — ALL clauses required).** H_int **support** if and only if:

1. \(r_{\mathrm{phase}} > r_{\mathrm{plain}}\) by more than the seed interval, **and**
2. \(r_{\mathrm{phase}} \ge r_{\mathrm{random}} - 0.05\), **and**
3. \(r_{\mathrm{phase}} > r_{\mathrm{no\_medium}}\) by more than the seed interval.

If (1)–(2) hold but (3) fails → label **medium_not_beyond_attribution** (primary does **not** win; cheap attribution ties or beats the medium). Phase claimed over tags only if \(r_{\mathrm{phase}} > r_{\mathrm{random}}\) by more than the interval **and** (1) and (3) hold. **FAIL condition for this draft's quality bar:** treating no-medium as secondary-only so primary could "win" without testing whether the medium buys intelligibility beyond cheap attribution.

Numeric locks: **G13 procedure LOCKED** (see §4.0 below). \(K\) experts and phases fixed and hashed (**draft \(K=4\)**); primary \(C_{\mathrm{match}}\) ±5% per arm; §3.0 attention choice locked (G18); **G17** docs-assert OK for freeze; probe Acc post-train / pre-primary.

**Honesty framing (Kyle):** if phase only matches random tags → paper is **tagged superposition as interpretability tool** (still a paper). If \(r_{\mathrm{phase}}\) fails clause (3) → medium is **not** buying intelligibility beyond cheap attribution — **not support**.


### 4.0 G13 crosstalk manipulation check (procedure)

**G13 Crosstalk bound — measurement procedure LOCKED (gate defect 2026-09-24).**

**Role:** **Manipulation check only.** Confirms the phase medium on the trained mixture still tracks T9's crosstalk curve *before* counted primary. G13 PASS is **not** evidence for the H_int / intelligibility claim and does **not** enter the §4 H_int verdict.

**Tolerance:** \(|X(\Delta\varphi) - \max(0, \cos\Delta\varphi)| \le 0.1\) (T9 crosstalk bound).

**Δφ sampling (LOCKED):** combined set =
1. all unordered pairwise phase gaps among the \(K\) hashed expert phases (mythos_tiny \(K=4\));
2. uniform grid \(N=36\) on \([0, 2\pi)\), spacing \(2\pi/36\).

**Aggregation (LOCKED):** **hard max-deviation** over the combined set — PASS iff
\(\max_i |X(\Delta\varphi_i) - \max(0, \cos\Delta\varphi_i)| \le 0.1\).
**Not** RMS, **not** mean deviation. Every sample must clear.

**Timing:** evaluate **post-train / pre-primary** — does **not** block freeze; blocks counted primary if FAIL after train. PENDING until trained mixture exists = evaluation only (do not fake Acc).

| Claim | Statement |
|-------|-----------|
| **H_int (primary)** | Phase-medium Pearson \(r(\widehat{\Delta L}_e, \Delta L_e)\) on held-out \((e,x)\); **support requires** beating plain, clearing random-tag floor, **and** beating no-medium attribution (verdict §4). |
| **Not primary** | Raw ACC lift of MoE-of-loops vs shared-loop (secondary / descriptive). |
| **Not primary** | Depth budgeting / ACT vs Fixed (T12: ACT ≈ oracle at toy → **solved control**). |
| **Not primary** | Hum-as-allocator win (**expected-null**). |

### 4.1 Depth budgeting (solved control)

T12 primary put **ACT = oracle** (ΔACC 0) at toy scale under frozen PREREG `bf757ee4…`. For T13, **per-expert depth is a calibration / control knob**, not a contribution claim. ACT/convergence early-exit may appear as mechanisms to realize \(L_{\mathrm{eff}}\) under \(C_{\mathrm{match}}\), but beating Fixed on depth is **out of primary**.

### 4.2 Context-slice stratification (open)

The open scientific part is whether **own context slice per expert** (vs shared context / shared loop state) is required for **H_int** at matched compute. Arm `moe_of_loops_shared_ctx` exists to make a **null** on slices informative.

---

## 5. Arms (draft grid — all iso-\(C_{\mathrm{match}}\); scored on H_int §4)

All arms are evaluated on the **same** H_int algebra (§4). Architecture differences change the medium / routing / slices; they do **not** redefine intelligibility.

| Arm | Class | Definition (draft) |
|-----|-------|--------------------|
| **`moe_of_loops_slices`** | **Primary** | K looped experts; per-expert depth budget; **per-expert context slice**; leaky-integrator state per expert; **stateless** allocator above — primary H_int arm |
| **`shared_loop_moe_ff`** | Primary control | Single shared loop (MoR/MoD-class); MoE feed-forward; **same** \(C_{\mathrm{match}}\) — H_int on analogous partition |
| **`moe_of_loops_shared_ctx`** | Ablation / secondary | Same as primary but **shared** context (no slice stratification) — isolates whether slices buy H_int |
| **`fixed_depth_loops`** | Depth control | Per-expert or shared fixed depth = calibrated \(L_{\mathrm{eff}}\); no ACT claim |
| **`stateless_allocator`** | Coordinator control | Allocator without hum / without coordinator state |
| **`hum_allocator`** | Expected-null control | Hum as one allocator arm — **we do not expect hum to win** |

**H_int controls (not architecture arms):** random-tag, plain-sum, oracle, **no-medium attribution** (activation-norm or grad×input; per-\((e,x)\); **inside verdict clause 3**) — named in §4.

Oracle / teacher depth (if any) = report-only, not a free compute arm.

---

## 6. Null / support vocabulary (draft)

Verdict rule from §4 is the binding numeric path for **support** / **null** on H_int.

| Label | Meaning (draft) |
|-------|-----------------|
| **support** | H_int supported **only if** \(r_{\mathrm{phase}} > r_{\mathrm{plain}}\) by more than the seed interval **and** \(r_{\mathrm{phase}} \ge r_{\mathrm{random}} - 0.05\) **and** \(r_{\mathrm{phase}} > r_{\mathrm{no\_medium}}\) by more than the seed interval; when slice contrast is in primary, slice arm beats shared-ctx on the same three-clause rule |
| **phase_over_tags** | Phase claimed over tags: \(r_{\mathrm{phase}} > r_{\mathrm{random}}\) by more than the interval (subset of support; requires the phase>plain clause too for full H_int support) |
| **tagged_superposition** | Phase medium \(r\) matches random-tag \(r\) (within interval / fails phase_over_tags): paper is about **tagged superposition as interpretability tool** — still a paper; **not** silent upgrade to phase claim |
| **medium_not_beyond_attribution** | \(r_{\mathrm{phase}}\) fails to beat \(r_{\mathrm{no\_medium}}\) by more than the interval: medium/readout is **not** buying intelligibility beyond cheap attribution (activation-norm / grad×input); can **tie primary** (T11.3-style) — **not** silent upgrade to medium-intelligibility |
| **support_lean** | Directional seed-mean clears interval but secondary / slice ablation is mixed (use sparingly; lock numeric rule at freeze) |
| **mixed** | Primary H_int positive on one contrast, null on the pre-registered slice or shared-loop control |
| **null** | Fails any support clause: does not beat \(r_{\mathrm{plain}}\) by more than the interval; or fails \(r_{\mathrm{phase}} \ge r_{\mathrm{random}} - 0.05\); or fails \(r_{\mathrm{phase}} > r_{\mathrm{no\_medium}}\) by more than the interval (→ prefer **medium_not_beyond_attribution** when only clause 3 fails); or slice stratification adds nothing at \(C_{\mathrm{match}}\) |
| **expected-null (hum)** | `hum_allocator` ≤ `stateless_allocator` on primary task metrics — **expected**; hum win is **not** support for H_int |
| **gate defect** | Harness / formula / Rule-19 failure → fix + re-freeze; **not** a scientific null |

Failures report in this vocabulary. **No silent upgrade** from null → support without re-freeze. **No silent upgrade** from tagged_superposition → phase_over_tags without clearing the interval.

---

## 7. Scale / seeds / stopping

| Item | Draft lock |
|------|------------|
| Scale | **`mythos_tiny`** — local first look |
| **mythos_tiny numerics** | **DRAFT locks (pending Kyle freeze amend):** \(K=4\), \(W=64\), \(L_{\max}=4\) (ACT early-exit **control only**), \(S_{\mathrm{full}}=64\), equal \(S_e=16\), vocab **256**, train **2000 steps**, H_int seed-interval **0.05** |
| Seeds | **Locked:** `[20260928, 20260929, 20260930, 20261001, 20261002]` |
| Stopping | **One** preregistered primary (**H_int** §4) × locked seeds after freeze; **no T13.2** retune series unless gate defect |
| Instability | **G9** T12 T1-style **training-instability check** (NaN / non-finite / explosive grads) — **runs at train**; language locked at freeze; **refuse counted primary** if FAIL |
| Affinity surface | **Default off.** Only if a later named gate explicitly adds it; GNURadio re-eyes then |
| Numeric locks (pre-hash) | **G13 procedure LOCKED (§4.0):** manipulation check; pairwise+N=36; max-dev≤0.1; Acc **post-train / pre-primary** (does **not** block freeze); **K + phases fixed and hashed**; \(C_{\mathrm{match}}\) ±5% per arm (§3.2); **attention-accounting locked (§3.0)**; **no-medium named**; **G17** n=1000 + Acc≤0.273 docs-assert OK for freeze |
| **T10 side-channel (G17)** | T10 random-tag hit **1.000** via **identity leak through tag placement**. **G17 criterion LOCKED (docs-assert OK for freeze):** linear (or equivalent) probe predicts expert identity from **tag placement alone** — expert content **removed or shuffled**. Probe sample size **n=1000** (placement-alone examples; cheap probe — **high power**). Chance \(p_0=0.25\) (\(=1/K\), draft \(K=4\)). **PASS if** Acc \(\le \mathbf{0.273}\), where **0.273 IS the one-sided 95% binomial upper bound** on chance at n=1000 under \(p_0=0.25\) (normal approx: \(0.25+1.645\sqrt{0.25\cdot0.75/1000}\approx0.2725\), rounded to three decimals **0.273**). **Retired (low-power):** prior draft n=200 / Acc≤0.30 — do **not** shrink n to keep 0.30. **Evaluate post-train / pre-primary** — docs pin does **not** block freeze; **FAIL / block counted primary** if Acc \(> 0.273\) after train. **G12–G15 do not cover this.** |

---

## 8. Rule 19 gates (own — listed)

See `docs/T13_RULE19_GATES.md`. Summary:

1. **DRAFT blocks emit** — no `PREREG.sha256` while status DRAFT.
2. **Matched-compute formula** present and arms iso-cost; **C_match ±5% per arm before hash**.
3. **Primary = H_int algebra** present **before** arm grid; arms reference §4.
4. **Related-work delta** MoD + MoR + context-slice stratification.
5. **Hum allocator expected-null** written.
6. **Affinity sha frozen** `7dae877a…`; no local Affinity copy.
7. **Parent T11/T12/T12b PREREG hashes untouched.**
8. **Scale `mythos_tiny`.**
9. **G9 Training-instability** — T12 T1-style; language at freeze; **runs at train**; refuse primary if FAIL.
10. **Stopping + null/support vocabulary** (incl. verdict rule + tagged_superposition honesty).
11. **Affinity surface default off.**
12. **H_int controls named (G12)** — random-tag / plain-sum / oracle.
13. **G13 Crosstalk bound** — **procedure LOCKED** (§4.0): manipulation check (not H_int evidence); Δφ = pairwise hashed phases ∪ N=36 grid on [0,2π); **max-deviation** ≤0.1 (not RMS); Acc post-train / pre-primary (does **not** block freeze).
14. **K + phases hashed** before hash emit (draft \(K=4\)).
15. **C_match ±5% per arm** under **primary** formula before hash (**G15**).
16. **G16 — No-medium inside verdict** — control (4) at per-\((e,x)\) granularity; **support requires** \(r_{\mathrm{phase}} > r_{\mathrm{no\_medium}}\) by more than the seed interval (not secondary-only).
17. **G17 — Tag-placement decodability (T10)** — placement-alone probe **n=1000**; chance **0.25**; Acc ≤ **0.273** (= one-sided 95% binomial UB at n=1000; **retired** low-power n=200/0.30); docs-assert OK for freeze; Acc evaluate **post-train / pre-primary** (blocks counted primary if FAIL); **G12–G15 do not cover**.
18. **G18 — Attention accounting locked** — primary = per-slice attention counted (§3.0 / §3.2 per-arm lines); secondary = full-context-equivalent; chosen **before** mythos_tiny numerics.

Script: `scripts/rule19_freeze_gates.py` — live; **`--emit-hash`** promotes draft→`PREREG.md` + emits `PREREG.sha256` when freeze-blocking gates PASS (G13/G17 Acc deferred).

---

## 9. What this draft does **not** authorize

- **Affinity retune** / edit of `affinity_inputs.py` or Affinity package `__all__`.
- **Silent carry** of frozen ScoreAffinity into T13 arms without a named gate.
- **Freezing** before compute formula §3 is locked **and** Rule-19 PASS **and** Kyle authorize.
- Emitting `PREREG.sha256` while status is **DRAFT**.
- Reopening T11.3 / T12 / T12b as retune; editing their sealed `PREREG.sha256`.
- Claiming tonic-gating 2.0 or treating T13 as T12b continuation.
- Treating depth-budget ACT wins as primary (T12 closed that as control).
- Treating hum-allocator wins as primary support (**expected-null**).
- Redefining intelligibility inside an arm without the §4 algebra.
- Claiming **phase over tags** when only tagged_superposition holds.
- Claiming medium intelligibility when \(r_{\mathrm{phase}}\) fails to beat \(r_{\mathrm{no\_medium}}\) (must report **medium_not_beyond_attribution**).
- Starting counted primary without **G13** crosstalk Acc PASS and **G17** placement Acc PASS after train (T10 side-channel; **G12–G15 do not cover**).
- Switching attention accounting after seeing which choice favors an arm (primary locked in §3.1 before numerics).

---

## 10. Frozen inheritance (read-only pins)

| Item | Pin |
|------|-----|
| Affinity `affinity_inputs.py` | `7dae877a3975c1ba4ca264f03f572d6749cce39a114ea3adbcffc010c7852c50` |
| T11.2 PREREG | `593762c4c19e85d1720f6b46edbb93869258c4047ad023e42e144416aaed20d1` |
| T11.3 PREREG | `27ab4a9573ec8c01ed043d72ef6c83ba47fdd4a764a373acb7453381bb9bd442` |
| T12 PREREG | `bf757ee4fc01e5c40b45f035150e7b716d12f7d8b30d745b9a7669f79bbf307c` |
| T12b PREREG | `851124b3c4bc66cb8e7c96c4f501084f4de186574f4f8d9446cac5879f62a3f0` |
| T13 `PREREG.sha256` | **emitted** (this freeze) |

---

## 11. Freeze checklist (future)

- [ ] Kyle + GNURadio review of this draft
- [x] Numeric `mythos_tiny` DRAFT locks: K=4, W=64, L_max=4, S_full=64, S_e=16, vocab=256, 2000 steps, seeds, H_int interval 0.05 (Kyle freeze amend OK)
- [ ] §3 formula tolerance (±5% or replacement) locked; per-arm assert before hash
- [ ] §3.0 primary = per-slice attn counted; secondary = full-context-equivalent; per-arm lines in §3.2 (G18) — locked before numerics
- [ ] §4 H_int algebra locked; **verdict embeds** \(r_{\mathrm{phase}} > r_{\mathrm{no\_medium}}\) as **required** for support (G16)
- [x] **G13 procedure LOCKED** (§4.0: manipulation check; pairwise+N=36; max-dev≤0.1); Acc post-train (does not block freeze) + K/phases hash language
- [x] G16 no-medium **inside verdict**; G17 n=1000, Acc≤0.273 docs-assert (= high-power binomial UB at p0=0.25; retired n=200/0.30); Acc post-train; G12–G15 do not cover G17
- [ ] Rule-19 script PASS on real harness (not skeleton-only)
- [x] G9 instability language locked (actual check runs at train; refuse primary if FAIL)
- [ ] Kyle freeze GO → promote draft → emit `PREREG.sha256` (G13/G17 Acc deferred post-train)

**Sequence after freeze:** train mixture (G9 T1 instability as gate) → G13 Acc + G17 Acc → primary × seeds. **No amendments post-freeze except gate defect.** FROZEN. No primary until train + G13/G17 Acc PASS. Hash emitted.
