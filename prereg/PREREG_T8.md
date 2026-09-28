# PREREG — T8: Hum State Routing vs T7 Baselines (esp. LR-smoothed)

| Field | Value |
|-------|--------|
| **Title** | T8 — Hum State Routing (reservoir dynamics → amplitude gate) on order-3 recurrence |
| **Date** | 2026-09-22 (PT) |
| **Owner** | Kyle (ZetaZero) |
| **Status** | **FROZEN** — `results/PREREG.sha256` written; multi-seed primary authorized |
| **Relation** | Separately preregistered follow-on to closed T7. Parent primary (`../results/primary/`) and T7 (`../t7/`) remain **FROZEN** — not edited by this experiment. Ticket ID in handoff table = “T7 PREREG freeze”; document lives under `t8/`. |

This document freezes T8 **before** any multi-seed primary comparison. Exploratory one-seed numbers in §disclosure are **non-counting**.

**T7 correction (fact, do not reverse):** Closed T7 **PG-inferred** = **per-token** `Linear(width→2) → atan2` φ_ref (no temporal mean in the shipped router). **LR-smoothed** (causal W=32 mean of MoE routing hidden → Linear) was exploratory, did **not** match PG-inferred, and is carried forward as the **strongest temporal-smoothing baseline**. Never claim “T7 used a 32-token mean” for closed PG-inferred.

---

## 1. Hypotheses (exact scoring)

Primary comparison for H1 robustness: **Hum full vs LR-smoothed**. T7 carry-overs (LR, LR-goal, HR1, PG-inferred) and T8 ablations (Hum-noL5, Hum+label, Stateless) are reported for context / mechanism; they do not redefine the H1 dual-bootstrap comparator unless noted.

### H1 (state vs function) — clean quality + disturbance vs LR-smoothed

On the shared T7 harder task (order-3 affine), **Hum full**:

1. **Clean quality tie:** seed-mean heldout content NLL \(L\) of Hum full is within **0.1 nats** of the best T7 carry-over arm’s seed-mean \(L\) among {LR, LR-goal, HR1, PG-inferred, LR-smoothed} (implementation-parity band as parent/T7). Specialization acceptance on Hum full: ≥4/8 tasks with top-1 dominant share ≥0.25 on each primary seed (same interface as T7).
2. **Disturbance improvement vs LR-smoothed (primary cells):** at router-input noise \(\sigma_n=0.3\) and at MoE-layer shift \(\delta=0.3\) (each arm’s own clean heldout MoE hidden std; shared seeded base tensors), let \(\Delta = L_\mathrm{cell}-L_\mathrm{clean}\). For each of these two cells separately, the paired seed difference
   \[
   D = \Delta_{\mathrm{LR\text{-}smoothed}} - \Delta_{\mathrm{Hum\ full}}
   \]
   must satisfy the **dual bootstrap rule** (identical to parent/T7):
   - percentile bootstrap 95% CI over the 5 seeds (10 000 resamples of the paired differences) **excludes 0**, **and**
   - the point estimate \(D\) **exceeds the width** of that 95% CI.
3. **Also report (not collapsing into H1):** Hum full vs LR, vs PG-inferred, vs Stateless on the same cells; native \(C_\mathrm{route}\) in native units (never combine MACs with ops).

**H1 supported** only if clauses 1 and 2 both hold. Partial → **mixed**. Clean fail or either robustness cell fail → **H0** or **mixed** in those words.

### H2 (persistence) — control silence → \(T_\mathrm{hold}\)

After **control silence** beginning at token index **64** (Level-1 / Level-1′ zeroed for \(t\ge 64\); MoE arms without Hum inputs: zero `task_embedding` contribution for \(t\ge 64\) as documented stand-in):

1. **State persistence (primary):** seed-mean \(T_\mathrm{hold}(\mathrm{Hum\ full}) > T_\mathrm{hold}(\mathrm{Stateless})\) and \(T_\mathrm{hold}(\mathrm{Hum\ full}) > T_\mathrm{hold}(\mathrm{LR})\) under the same dual bootstrap on paired differences (CI excludes 0 **and** point > width). \(T_\mathrm{hold}\) = tokens until top-1 expert leaves the pre-silence regime top (mode over \([0,64)\)), averaged over sequences (`measures.SILENCE_TOKEN=64`).
2. **λ scaling (counting ablation, frozen dynamics otherwise):** with all other Hum params fixed, evaluate \(T_\mathrm{hold}\) at three frozen leak probes
   \[
   \lambda \in \{0.012740234375,\ 0.02548046875,\ 0.0509609375\}
   \]
   (½×, 1×, 2× the T3-frozen λ). H2 λ-clause **supported** if seed-mean \(T_\mathrm{hold}\) is **strictly monotone decreasing** in λ (longer hold at smaller leak). Flat / non-monotone → null for this clause.

H2 supported if clause 1 holds; clause 2 reported as supported / H0 separately. H2 does not rescue a failed H1.

### H3 (salience) — inverted-U across tonic sweep

On the routine/distractor split (`measures.H3_SPLIT_RULES`; segments @ 0/32/64/96):

- **M** (miss) = fraction of routine-relevant segments with no expert share > baseline (\(1/8\)) within **4** tokens of segment start.
- **F** (false excitation) = fraction of novel-irrelevant segments where `g_tonic` hits upper **1.5** (N/A for pure LR / LR-smoothed / HR1 — no `g_tonic`).

Sweep baseline `g_tonic` via frozen tonic table (`tonic_level` ∈ {0,1,2,3,4} → {0.00, 0.25, 0.50, 0.75, 1.00}).

**H3 supported** if the composite \(C(\ell)=M(\ell)+F(\ell)\) (seed-mean) has a **unique minimum at an interior** tonic level \(\ell\in\{1,2,3\}\) (inverted-U / U-shaped cost; not flat, not monotone across 0..4). Bootstrap: report percentile 95% CI on \(C(\ell)-C(\ell^\star)\) for boundary levels vs the minimizing interior; dual rule optional for disclosure — **primary H3 call is the unique-interior-min rule on seed-means**. Flat or monotone \(C\) → **H0** for H3.

### H4 (capacity) — **report-only**

Forced simultaneous drive of \(m=1..8\) experts on HumState; **\(K_\mathrm{max}\)** = largest \(m\) with selection accuracy > **0.9** (driven ∩ top-\(m\) by amplitude share). Report \(K_\mathrm{max}\), interference proxies, and least-used `Rec_e`. **Not** part of H1/H2/H3 support/H0.

### Null wording (H0 / mixed)

- **H1 null:** no better than T7 **LR-smoothed** on the H1 disturbance cells (dual bootstrap fails on noise and/or shift), **or** clean quality gap > 0.1 nats vs best T7 arm.
- **H2 null:** no persistence (\(T_\mathrm{hold}\) Hum ≰ Stateless/LR under dual rule), **or** H2 λ-clause flat in λ.
- **H3 null:** flat or monotone salience curves \(C(\ell)\) (no unique interior minimum).
- Failures reported in the words **supported**, **H0**, or **mixed**. H2/H3/H4 do not redefine H1.

---

## 2. System specification (frozen)

### 2.1 Data (T7 harder task — symlink, not regenerated)

| Item | Frozen value |
|------|----------------|
| Path | `t8/data` → `../t7/data` |
| Data seed | `20260921` |
| RNG | `numpy.random.PCG64` |
| Vocab | markers `0..7`; content `8..63` (56 symbols) |
| Sequence length | 128 |
| Rule | **Order-3** affine: \(x_{t+1}=(a[t]x_t+b[t]x_{t-1}+c[t]x_{t-2}+d[t]) \bmod 56\) |
| Coeffs | `a=(1×8)`; `b=(1,1,1,1,2,2,2,2)`; `c=(1,1,2,2,1,1,2,2)`; `d=(1,5,9,13,17,23,29,37)` in `data/metadata.json` |
| Flip noise | `p_flip=0.0` |
| Train | 256 sequences/task → `[2048,128]` |
| Heldout | 64/task → `[512,128]` |
| Switching | 512 sequences; mid-sequence distinct marker restarts new recurrence |
| Manifest | `data/MANIFEST.sha256` (= T7; sha256 `acae1c51bc8ff0d9077a2637698610383cc56249e811290f0a752b680c2af30f` as recorded in T6 smoke) |

### 2.2 Shared model

| Item | Frozen value |
|------|----------------|
| Architecture | 4-layer decoder, width 128, 4 heads, FFN width 256 |
| MoE layer | Layer index 2; 8 experts `128→256→128` GELU |
| Task interface | Shared **latest-marker task embedding** broadcast (parity with T7) — identical for all arms unless an arm ablates it |
| Constraint | T7 carry-over arms unchanged (do not move thresholds or rewrite closed T7 code) |

### 2.3 Input levels (T1 — frozen projections)

| Level | Name | Rate | Frozen spec |
|------:|------|------|-------------|
| **0** | External control (ceiling) | per-seq | One-hot task → `label_bias`; **Hum+label only** |
| **1** | Regime probe | per-segment | Frozen linear probe over first **8** tokens’ embeddings from T7 LR primary seed `20260921`. Fit `train.npy`; eval heldout task-id accuracy **0.962891**. Artifact `results/t1/regime_probe.pt`; sha256 **`5603953a28706a72d23789e3e6f39eed4ee9e655cd404e1134c7ac03400971ad`** |
| **2** | Token→expert evidence | per-token | Fixed random `R^{128}→R^{8}`, seed `20260921`, scale `1/√128` ≈ `0.08838834764831843`. Not trained |
| **3** | — | — | **Rejected** (`NotImplementedError`) |
| **5** | Router affinity | per-token | Identity passthrough of router probs/logits, or disclosed frozen `Linear(128→8)` pseudo-affinity when live affinities would be endogenous |
| **6** | Expert usage | per-token | Normalized counts over last **16** tokens |
| **7** | Surprise | per-token (T−1) | Content-masked next-token CE, z-scored with frozen `results/t1/surprise_zstats.json`: **mean=1.2553684711456299**, **std=1.1743452548980713**, n=64512 |
| **1′** | Relevance | per-segment | Cosine of Level-1 vs identity goal codebook \(I_8\); threshold **0.5** |

### 2.4 Filter stack corners (T2)

Cascade on Level-5 affinity **per expert unit**: Roofing → Adaptive (MAMA-style) → Octave bank. Equations: `docs/FILTERS.md`.

| Parameter | Value |
|-----------|------:|
| `HP_PERIOD` | **64** |
| `SS_PERIOD` | **10** |
| `MAMA_FAST_LIMIT` / `MAMA_SLOW_LIMIT` | **0.5 / 0.05** |
| `MAMA_PHASE_SCALE_DEG` | **30°** |
| `OCTAVE_TAUS` | **4, 16, 64** |
| `OCTAVE_WEIGHTS` | **1/3 each** |

Polarity: α rises on regime change / falls at steady (handoff; differs from classic MAMA). Deterministic; no RNG.

### 2.5 Hum dynamics + gate (T3)

| Parameter | Value |
|-----------|--------|
| λ (leak) | **0.02548046875** (PARAMS.md lists 0.0254804687; code/accept: 0.02548046875) |
| measured τ (1/e tokens) | **30.10** (target 30; band 20–40) |
| a₀ | **0.1** |
| idle_rate | = λ |
| coupling seed | `20260921` |
| coupling spectral radius ρ(C) | **0.0149999971** (target 0.015) |
| effective linear ρ `(1−λ)(I+C)` | **0.9891373410** (< 1 echo-state) |
| g_tonic default | 1.0 |
| g_tonic_low / high (gate width) | **0.5 → top-1** / **2.0 → top-3**; else top-2 |
| amp_gate_threshold | **0.05** (sub-threshold → index −1, gate 0) |
| resting phases φ_e | \(2\pi e/8\) (**spacing only; phase is not a veto**) |
| W1, W5, W2, b0 | \(I_8\) |

Dynamics once per token: leak → idling → coupling → tonic×drive. Gate gains = \(|s_e|/\sum|s|\) (amplitude shares only). Coupling matrix: `results/t3/PARAMS.md`.

### 2.6 Salience (T4)

| Parameter | Value |
|-----------|------:|
| CUSUM \(k\) | **0.5** |
| CUSUM \(h\) | **5.0** |
| Habituation rate | **0.5** |
| Kick duration | **8** tokens |
| Floor kick | **0.10** |
| Relevance kick (unhabituated) | **1.00** |
| Relevance score threshold | **0.5** |
| Surprise raise target `G_TONIC_UPPER` | **1.50** |

Tonic baseline table (H3): level 0→0.00, 1→0.25, 2→0.50, 3→0.75, 4→1.00.

### 2.7 Routing arms (counting primary)

| Arm | Role | Definition |
|-----|------|------------|
| **Hum full** | **Primary** | FilterStack(L5) + L2 (+ optional L1/L7/1′ via salience) → HumState → amplitude gate. Full MoE LM (`hum_router.py` wiring). |
| **Hum-noL5** | Ablation | Same as Hum full **without** Level-5 affinity / filter path |
| **Hum+label** | Ceiling ablation | Hum full + Level-0 label bias |
| **Stateless** | Ablation / mirror | Same channels → softplus orthogonal map → **same** `gate_from_amplitudes`; **no** persistent state (`stateless_neuromod.py`). T5 accept: filters/label off for counted clean-L |
| **LR** | T7 carry-over | Linear hidden→8; top-2; aux 0.01; \(C_\mathrm{route}=1024\) MACs. Reuse T7 primary ckpts |
| **LR-goal** | T7 carry-over | concat(hidden, goal)→Linear; top-2; aux 0.01; 2048 MACs. Reuse T7 |
| **HR1** | T7 carry-over | Hash `(5·token+1) mod 8`; top-1. Reuse T7 |
| **PG-inferred** | T7 carry-over | **Per-token** `Linear(128→2)→atan2`; gains `max(0,cos(φ_e−φ_ref))`; top-4; 256 MACs + 8 cos. **Not** W=32. Reuse T7 |
| **LR-smoothed** | **H1 comparator** / T7 exploratory carry-over | Causal mean **W=32** of MoE routing hidden → Linear; top-2; aux 0.01; \(C_\mathrm{route}=1024\) MACs (+ 32 smoothing adds, not folded into MACs). Reuse `t7/results/exploratory/lr_smooth/` |

### 2.8 Training (every new primary seed/arm)

| Item | Frozen value |
|------|----------------|
| Optimizer | AdamW, lr **0.002**, weight decay 0.01 |
| Schedule | Linear warmup 50 steps, then cosine down to 0.1× base |
| Steps | **1000** |
| Batch | **32**, shuffle with `Generator.manual_seed(seed)`, drop_last |
| Grad clip | 1.0 |
| Seeds | **`[20260921, 20260922, 20260923, 20260924, 20260925]`** |
| Device | CPU OK; record device in metrics |
| Hum dynamics | **Fixed** (λ, a₀, C, filters, salience params). Learn only allowed projections / backbone / experts as specified per arm |
| Stateless | Neuromod W,b **frozen** after init (`requires_grad=False`); backbone + experts trainable (T5 recipe) |
| T7 carry-overs | **Do not retrain** — load existing primary / lr_smooth checkpoints |

**Budget note:** Do **not** train a full 5×9 grid from scratch. New multi-seed trainings = Hum full, Hum-noL5, Hum+label, Stateless (≤20 new runs). Evaluate all nine arms in the harness.

---

## 3. Disturbance grid + silencing + routine/distractor + capacity

### 3.1 Noise / shift (T7 copy)

Grid: \(\{0,\ 0.1,\ 0.3,\ 1.0\}\) for router-input noise and MoE-layer shift.

- **Noise** → routing-decision hidden **only**.
- **Shift** → every MoE-layer hidden (routing **and** expert input).
- Per-arm clean heldout MoE hidden std; shared seeded base tensors across arms.

**Primary H1 cells:** \(\sigma_n=0.3\), \(\delta=0.3\).

### 3.2 Control silencing (H2)

Zero Level-1 / Level-1′ after token **64**. Stand-in for non-Hum MoE: zero `task_embedding` for \(t\ge 64\). Measure \(T_\mathrm{hold}\`.

### 3.3 Routine / distractor split (H3)

Generator + `H3_SPLIT_RULES`: segments @ 0/32/64/96 → routine-relevant, routine-irrelevant, novel-relevant, novel-irrelevant. M / F as §1. Tonic sweep over frozen table.

### 3.4 Capacity (H4)

Forced drive \(m=1..8\); \(K_\mathrm{max}\) at accuracy > 0.9. Report-only.

---

## 4. Measures (exact)

| Symbol | Definition |
|--------|------------|
| \(L\) | Mean content-token NLL on heldout (markers excluded) |
| \(C_\mathrm{route}\) | Native routing-decision cost (LR/LR-smoothed=1024 MACs; LR-goal=2048; HR1=0; PG-inferred=256 MACs + 8 cos; Hum/Stateless=documented controller MACs/ops — never combine units) |
| \(C_\mathrm{exp}\) | `mean_active_experts_per_token × 65536` |
| \(B\) | Load-balance CV |
| \(\Delta L\) | \(L_\mathrm{cell}-L_\mathrm{clean}\) |
| \(T_\mathrm{sw}\) | First lag after switching marker with >0.5 mass on new-task dominant set; index 0 = immediate |
| \(T_\mathrm{hold}\) | Tokens until top-1 leaves pre-silence regime top after silence @ 64 |
| \(M\), \(F\) | Miss / false-excitation on H3 split |
| \(K_\mathrm{max}\) | Capacity (H4) |
| \(\mathrm{Rec}_e\) | Per-expert recall vs clean dominant oracle |

**Bootstrap:** 10 000 paired-seed percentile 95% CI; dual rule (CI excludes 0 **and** point > width) wherever §1 requires it.

Harness: `harness_t8.py` · `t8_routing/measures.py`.

---

## 5. Primary comparison rule / seeds / budget

1. After `t8/results/PREREG.sha256` exists, train **new** arms Hum full, Hum-noL5, Hum+label, Stateless on seeds `20260921..20260925` into `t8/results/primary/` (≤20 runs). **Reuse** T7 primary + lr_smooth checkpoints for carry-overs.
2. Run `harness_t8.py` per seed/arm; score H1–H4 exactly as §1.
3. Report **supported**, **H0**, or **mixed** per hypothesis.
4. **No multi-seed before hash.** **No retune of λ / a₀ / filters / salience / gate thresholds from multi-arm peeking.**
5. Do **not** train full 5×9 from scratch.

---

## 6. Rules of the road

1. Do not edit parent `../PREREG.md`, `../results/primary/`, or closed `../t7/` counting artifacts.
2. **No learned dynamics** on the hum state unit after seeing results (reservoir stays fixed).
3. **Stateless is the mirror** of Hum’s amplitude-gate readout without persistence — same gate API (`gate_from_amplitudes`).
4. Biology / neuromodulation citations are **motivation only** (not evidence; Beaulieu et al. 2020 neighbor idea for T5 — not meta-learned).
5. Fix parameters from single-system / one-seed calibration only (already done T1–T5); disclose all peeks (§7).
6. Same data splits and seeded disturbances across arms within a seed/cell.
7. Cost channels stay in native units; never claim PG-inferred used W=32.
8. H1 comparator is Hum full vs **LR-smoothed**; other arms are context / ablations.
9. Do not move T7 baselines or rewrite closed T7 router code.

---

## 7. Mandatory disclosure — everything seen before freeze (non-counting)

### 7.1 T7 correction

PG-inferred was **per-token**, not W=32. LR-smoothed (W=32) is the strongest temporal baseline carried into T8 H1.

### 7.2 T0 bitcheck (seed `20260921`) — bit-for-bit

| Arm | heldout \(L\) |
|-----|--------------:|
| LR | 1.255368582283 |
| LR-goal | 1.260292829384 |
| HR1 | 1.326833800782 |
| PG-inferred | 1.271556541678 |
| LR-smoothed (exploratory) | 1.261728026801 |

### 7.3 T7 primary seed-means (closed; n=5) — context only

| Arm | clean \(L\) | \(T_\mathrm{sw}\) | ΔL noise 0.3 | ΔL shift 0.3 |
|-----|------------:|------------------:|-------------:|-------------:|
| LR | 1.267270 | 3.2948 | 0.003782 | 0.015115 |
| LR-goal | 1.266655 | 0.5309 | 0.002085 | 0.017457 |
| HR1 | 1.296894 | 1.0255 | 0.0 | 0.002887 |
| PG-inferred | 1.272263 | 3.8672 | 0.002564 | 0.005999 |
| **LR-smoothed** | **1.276703** | **16.7672** | **0.000487** | **0.022063** |

T7 H1 verdict was **mixed**; H2 **H0**. Not re-litigated here.

### 7.4 T1 regime probe / surprise z-stats

- Probe heldout task-id accuracy **0.962891** (≥ 0.9).
- Probe sha256 `5603953a28706a72d23789e3e6f39eed4ee9e655cd404e1134c7ac03400971ad`.
- Feature extractor = T7 LR primary seed 20260921 (counting seed; disclosed peek). Backbone not fine-tuned.
- Surprise z: mean **1.2553684711456299**, std **1.1743452548980713**.

### 7.5 T3 λ calibration

λ = **0.02548046875** from **single-system** unforced persistence only (excess mean-amplitude → 1/e ≈ 30 tokens). No multi-seed, no heldout, no filter/salience peeking. a₀ = 0.1; ρ(C) ≈ 0.015.

### 7.6 T5 Stateless clean-L gap (seed `20260921`)

| Quantity | Value |
|----------|------:|
| T7 LR heldout \(L\) | 1.255369 |
| Stateless heldout \(L\) | 1.273114 |
| Gap | **0.017746** nats (≤ 0.05 accept) |

Router frozen; filters/label off for counted run. Pseudo Level-5 = frozen `Linear(128→8)`.

### 7.7 T6 harness smoke (seed `20260921`, non-counting)

| Arm | \(L\) | \(T_\mathrm{hold}\) | notes |
|-----|------:|--------------------:|-------|
| LR | ~1.25 | ~0.91 | MoE |
| LR-smoothed | ~1.26 | ~34.6 | long hold from smoothing state |
| Stateless | ~1.27 | ~1.84 | |
| Hum (controller-only) | n/a | **64** | no LM; M=0.0; \(K_\mathrm{max}=8\) |

Hum full MoE LM **not** trained multi-seed before this freeze. `HumRouter` is a **sketch** (`t8_routing/hum_router.py`).

### 7.8 What was **not** seen

**No multi-seed Hum primary comparison** was computed before writing this file. No 5×9 training grid.

---

## 8. What is NOT frozen yet / known gaps

| Gap | Note |
|-----|------|
| **Hum full LM training** | Not yet multi-seed; T6 used controller-only for Hum T_hold / M / F / K_max |
| **`HumRouter` wiring** | Sketched; full per-token salience path (Level-7 from LM) left for primary ticket |
| **Exact Hum \(C_\mathrm{route}\)** | Document native controller MACs/ops at first trained MoE run; do not invent a collapsed score |
| **Learnable Hum projections** | Which of affinity / W_in / backbone are trained vs frozen — finalize in primary train script without changing §2.5 dynamics |
| **Hum-noL5 / Hum+label recipes** | Same as Hum full except channel flags; confirm in train entrypoint |
| **PREREG.sha256** | Parent writes after review — **this agent must not** |
| **H2 λ-probe ckpts** | Three λ values share frozen C / a₀; only leak changes — may be eval-time override without full retrain if state is deterministic given drives; if train couples to λ, disclose and train the three probes |

---

## 9. Freeze procedure

1. Parent reviews accept docs T0–T6 + this PREREG.
2. Parent writes `t8/results/PREREG.sha256` = sha256(this file) + `created_iso` (PT) and anchors on Drive (no public gist).
3. Only then run multi-seed primary into `t8/results/primary/` per §5.

See also `READY_TO_FREEZE.md`.

---

## 10. Freeze statement

Editing H1–H4 thresholds, primary measures, λ / a₀ / filter / salience / gate constants, or the Hum vs LR-smoothed comparator after peeking at T8 multi-seed outputs voids this preregistration for that run. T7 and parent primary remain the frozen counting runs for their respective comparisons; T8 is a separate hum-state follow-on.
