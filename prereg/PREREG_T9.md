# PREREG — T9: Phase as a Medium (ensemble / selective readout / shared-ref drift / rare-expert)

| Field | Value |
|-------|--------|
| **Title** | T9 — Phase as a Medium (complex buffer + demod vs T8-hum `φ_ref`) |
| **Date** | 2026-09-22 (PT) |
| **Owner** | Kyle (ZetaZero) |
| **Status** | **CANDIDATE FREEZE** — awaiting parent `results/PREREG.sha256` |
| **Relation** | Separately preregistered follow-on to closed T7 and closed T8. Parent primary (`../results/primary/`), T7 (`../t7/`), and T8 (`../t8/`) remain **FROZEN** — not edited by this experiment. T8 closing citation: `RESULTS.md` sha256 **`c772ac41385501ec50a5221c740a0a1581863616922f4f1327209b4e59741bae`** (`t8/results/T8_CLOSED.md` / `t8/results/CLOSING.sha256`). |

This document freezes T9 **before** any multi-seed primary comparison. One-seed / smoke numbers in §disclosure are **non-counting**.

**Do not** create `PREREG.sha256` from this agent — parent hashes after review.

---

## 1. Hypotheses (exact scoring)

Mission: treat **phase as a medium**, not as an index into a router table. Experts write into a shared complex buffer; readout demodulates against `φ_ref` (primary: T8 hum).

Primary medium arm for H1–H4: **`phase_hum`** (phase+hum-ref). Controls: **`plain_sum`** (G_sum / baseline), **`phase_fixed`**, **`phase_hp_hum`**, **`random_tag`** (mirror), **`phase_multislot`** (capacity / interference probe). Carry-overs T7 **LR** / **LR-smoothed** may be reported for context; they do not redefine H1–H4 comparators unless noted.

### H1 (ensemble gain) — \(G > G_\mathrm{sum}\)

On the cooperative CRT heldout (`data_coop/`), let

\[
G = \mathrm{mean}(L_{\mathrm{only\,A}}, L_{\mathrm{only\,B}}) - L_{\mathrm{pair}}
\]

averaged over the eight frozen task pairs (see §2.1). \(G_\mathrm{sum}\) is the same quantity for **`plain_sum`**.

**H1 supported** if seed-mean \(G(\texttt{phase\_hum}) > G_\mathrm{sum}\) under the **dual bootstrap rule** (identical to parent/T7/T8): percentile bootstrap 95% CI over the 5 seeds (10 000 resamples of paired differences \(D = G(\texttt{phase\_hum}) - G_\mathrm{sum}\)) **excludes 0**, **and** the point estimate \(D\) **exceeds the width** of that 95% CI.

Also report (not collapsing into H1): clean ordinary heldout \(L\) and coop heldout \(L\) for all arms; \(G\) for `phase_fixed` / `phase_hp_hum` / `phase_multislot`.

### H2 (selective readout) — \(X(\Delta\varphi)\) within \(\pm 0.1\) of \(\max(0,\cos\Delta\varphi)\)

On the synthetic separation sweep (`measures_t9.separation_curve`; \(\Delta\varphi \in \{0°,15°,…,180°\}\)):

\[
X(\Delta\varphi) = \text{recovered demod scale after write at }\varphi_\mathrm{ref}+\Delta\varphi.
\]

**H2 supported** if, for every sweep point, \(|X(\Delta\varphi) - \max(0,\cos\Delta\varphi)| \le 0.1\) (equivalently MAE \(\le 0.1\) with no point outside the band). This is a **geometric / plumbing** criterion on the medium (seed-invariant when writers/norms are frozen); report once per primary seed for provenance.

ReLU half-wave is **load-bearing** for H2 (rule §7): without `ReLU`, anti-phase writes do not go to zero and the \(\max(0,\cos)\) shape fails.

### H3 (shared-ref drift) — \(A(\sigma_c)\) under drift

Shared phase random walk with \(\sigma_c\) on grid \(\{0,\ 0.01,\ 0.03,\ 0.05\}\) (`unit="cycles"` → step std \(= 2\pi\cdot\sigma_c\) rad; `numpy`/`torch` seed per run). Measure

\[
A(\sigma_c) = \frac{\text{recovered amplitude}(\sigma_c)}{\text{recovered amplitude}(0)}
\]

(unit probe + LM drift cells; see §4).

**Primary H3 cell:** \(\sigma_c = 0.05\).

**H3 supported** if seed-mean \(A(\sigma_c{=}0.05)\) for **`phase_hum`** exceeds that of **`phase_fixed`** under the dual bootstrap on paired differences \(D = A_{\mathrm{hum}} - A_{\mathrm{fixed}}\) (CI excludes 0 **and** point > width), **and** the T3 shared-drift PLV gates still hold on the probe: hum PLV to oracle \(> 0.8\), fixed PLV \(< 0.3\) by token 64.

Also report: `phase_hp_hum` \(A\) / PLV (informational; HP-hum is the offset-reject control from T8 `NEXT_PREREG`).

### H4 (rare-expert) — \(\min\mathrm{Rec}_e\) vs T8 hum & LR-smoothed

On ordinary clean heldout, \(\min\mathrm{Rec}_e\) as in `measures_t9.expert_recall` (dominant set = cumulative profile mass \(\ge 0.5\)).

**Comparators (frozen T8 closing report, seed-mean, n=5):** cite T8 `RESULTS.md` sha256 `c772ac41…`:

| Arm (T8) | seed-mean \(\min\mathrm{Rec}_e\) |
|----------|---------------------------------:|
| Hum full | **0.2865** |
| LR-smoothed | **0.3647** |

**H4 supported** if seed-mean \(\min\mathrm{Rec}_e(\texttt{phase\_hum})\) is **strictly greater** than **both** T8 Hum full (0.2865) **and** T8 LR-smoothed (0.3647) under dual bootstrap on paired differences vs each comparator (CI excludes 0 **and** point > width). Report \(\mathrm{Rec}_e\), \(\mathrm{Rec}_\mathrm{least\_used}\) as secondary.

If \(\min\mathrm{Rec}_e\) beats Hum but not LR-smoothed (or vice versa) → **mixed**. Below both → **H0** for H4.

### Phase vs random-tag attribution rule

`random_tag` is the structured-phase **mirror** (orthogonal real unit tags; same amplitude / half-wave ReLU path; **no** phase geometry).

| Outcome | Attribution |
|---------|-------------|
| \(G(\texttt{phase\_hum}) > G_\mathrm{sum}\) **and** \(G(\texttt{phase\_hum}) > G(\texttt{random\_tag})\) (dual bootstrap on both) | **Phase geometry** (H1 phase-specific) |
| \(G(\texttt{random\_tag}) \ge G(\texttt{phase\_hum}) > G_\mathrm{sum}\) | **Amplitude / multiplexing** win — **not** phase; do **not** claim H1 phase-supported |
| Neither medium arm beats \(G_\mathrm{sum}\) | **H0** for H1 |

H2’s \(\max(0,\cos)\) curve is phase-arm only; random-tag reports its own projection selectivity separately and does **not** rescue H2.

### Null (live) — gating lesson

**Null wording:** given gating / plain-sum / random-tag controls, the phase medium adds no reliable gain (H1–H4 fail jointly or selectively as above).

**Gating lesson (from closed T8, citation `c772ac41…`):** index/amplitude gating (Hum) did **not** clearly purchase rare-expert recall vs LR-smoothed (\(\min\mathrm{Rec}\) Hum 0.2865 ≰ LR-smoothed 0.3647); persistence and shift sensitivity were the same leaky-integrator property. T9 asks whether a **medium** (superposition + coherent demod) does better than index gates — null is live until primary says otherwise. Failures reported in the words **supported**, **H0**, or **mixed**. H2/H3/H4 do not redefine H1.

---

## 2. System specification (frozen)

### 2.1 Data

| Item | Frozen value |
|------|----------------|
| Ordinary path | `t9/data` → `../t7/data` (symlink; **not** regenerated) |
| Ordinary MANIFEST | sha256 of `MANIFEST.sha256` file = `acae1c51bc8ff0d9077a2637698610383cc56249e811290f0a752b680c2af30f` (= T7/T8) |
| Coop path | `t9/data_coop/` |
| Coop MANIFEST file sha256 | **`aeb60b5abc975ca6e102e7a132f67c33d0e5ffb8b51224879d8d0850d67d90c9`** |
| Data seed | `20260921` |
| Vocab | markers `0..7`; content `8..63` (56 symbols) |
| Seq len | 128 |
| Affine | T7 order-3: \(x\leftarrow (a x_t + b x_{t-1} + c x_{t-2} + d)\bmod 56\) |
| **Coop CRT rule** | \(x = \mathrm{CRT}(\mathrm{pred}_A \bmod 7,\ \mathrm{pred}_B \bmod 8)\); closed form \(x=(\mathrm{pred}_B\bmod 8)+8\cdot((( \mathrm{pred}_A\bmod 7)-(\mathrm{pred}_B\bmod 8))\bmod 7)\) |
| Coop layout | pos0=A, pos1=B, pos2..4 seeds, pos5..127 joint CRT recurrence |
| Frozen pairs | `(0,4), (1,5), (2,6), (3,7), (0,6), (1,7), (2,4), (3,5)` |
| Coop sizes | 128 train / 32 heldout per pair; 256 coop-switching |
| Spec | `docs/COOP_SPLIT.md` |

### 2.2 Shared model + frozen experts

| Item | Frozen value |
|------|----------------|
| Architecture | 4-layer decoder, width 128, 4 heads, FFN width 256 (T7/T8 parity) |
| MoE layer | Layer index 2; 8 experts `128→256→128` GELU |
| **Frozen experts** | `results/t1/frozen_experts.pt` sha256 **`2ef3bfa53b4209497e6c9efd4c25f6cd3c7e1fae709a151b78a764226888d9eb`** |
| Full plain-sum ckpt | `results/t1/plain_sum_seed20260921.pt` sha256 `06e360aed9cc30965575314f0dac39136959db87d081c0dd7041c1d4594b9210` |
| Freeze contents | Expert FFNs + shared backbone (embeddings, non-router blocks, final norm, LM head). **Router not frozen** (later arms may replace). |
| Plain-sum train | Seed `20260921`, 1000 steps, batch 32, lr `2e-3`, aux `0.01`; mix union T7 train (2048) + coop_train (1024) = 3072 |
| Ordinary heldout L (plain-sum) | **1.296938** (T1 accept) |
| Specialization | 8/8 dominant (share ≥ 0.25); pair_matters True (mean ΔL silence A/B > 0) |

**Primary rule:** experts stay frozen. Do **not** silently retrain the expert bank unless an arm’s recipe in a later amendment explicitly says so.

### 2.3 Buffer / demod (T2)

| Item | Frozen value |
|------|----------------|
| Buffer | \(B\in\mathbb{C}^{128}\) |
| Writer phase | \(\varphi_e = 2\pi e / 8\) |
| Normalize | \(f_e / \|f_e\|_{\mathrm{mean\,L2}}\) — fixed norms in `results/t2/expert_norms.json` |
| Write | \(B \mathrel{+}= \mathrm{normalize}(f_e)\,\mathrm{e}^{i\varphi_e}\) |
| Single-ref demod | \(y = \mathrm{ReLU}\big(\mathrm{Re}(B\,\mathrm{e}^{-i\varphi_{\mathrm{ref}}})\big)\) |
| Multi-slot (k=2) | Sum of demods at \(\varphi_{\mathrm{ref}}\) and \(\varphi_{\mathrm{ref}}+\pi\) |
| Module | `t9_routing/medium.py` |

Fixed per-expert mean-L2 norms (clean train batch 64×128):

| e | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| mean L2 | 24.9422 | 25.3764 | 18.5071 | 19.5900 | 26.2218 | 22.7313 | 23.0652 | 21.7174 |

### 2.4 Reference `φ_ref` (T3)

| Provider | `φ_ref` |
|----------|--------|
| **HumRef** (`phase_hum`) | Amplitude-weighted population phase of T8 `HumState` |
| **FixedRef** (`phase_fixed`) | \(2\pi\cdot\mathrm{label}/E\) frozen at sequence start |
| **HighPassedHumRef** (`phase_hp_hum`) | HumRef after high-passing **drives before the integrator** via T8 `RoofingFilter` (`HP_PERIOD=64`; see T8 `docs/FILTERS.md` / `docs/REFERENCE.md`) |

**Hum params (T8 import, unchanged)** — `results/t3/PARAMS.md`:

| Parameter | Value |
|-----------|--------|
| λ (leak) | **0.02548046875** |
| a₀ | **0.1** |
| idle_rate | = λ |
| coupling seed | `20260921` |
| coupling ρ | **0.0149999971** |
| effective linear ρ | 0.9891373410 |
| T8 `PARAMS.md` sha256 | `14d25fa3ac43bf288ab6025b6a6b29979cf3c97293173c48400ed6386f14ee0c` |
| Coupling matrix sha256 | `7e5661d744ca11668dc1123b7e2eb344ca4d280adc91a4ad39c3d7089b3980aa` |

Shared-drift probe (T3 accept): σ_c = 0.05, tokens 64, `unit=cycles`; HumRef PLV **0.842755** (> 0.8), FixedRef PLV **0.096284** (< 0.3).

### 2.5 Arms (counting)

| Arm | Role | Definition |
|-----|------|------------|
| **`plain_sum`** | Baseline / \(G_\mathrm{sum}\) | Frozen T1 LR MoE (real gates = weighted expert sum) |
| **`phase_hum`** | **Primary medium** | PhaseBuffer writes + HumRef demod |
| **`phase_fixed`** | H3 control | Same writes; FixedRef |
| **`phase_hp_hum`** | Offset-reject control | Same writes; HighPassedHumRef |
| **`random_tag`** | **Mirror** | Orthogonal real tags (`seed=20260921`); project + ReLU |
| **`phase_multislot`** | Capacity probe | Multi-slot k=2 demod |

Optional context (not required for H1–H4): T7 **LR** / **LR-smoothed** ckpt reuse.

### 2.6 Disturbance grids + seeds

| Grid | Values | Primary cell |
|------|--------|--------------|
| Shared drift \(\sigma_c\) | `{0, 0.01, 0.03, 0.05}` | **0.05** (H3) |
| Independent jitter \(\sigma_j\) | `{0, 0.05, 0.1, 0.2}` | report-only |
| Router noise / MoE shift | `{0, 0.1, 0.3, 1.0}` (T7/T8 copy) | report `0.3` |
| Expert dropout | `drop_e0` … (harness) | report-only |
| Separation sweep | \(\Delta\varphi\) 0..π step 15° | H2 |

| Item | Frozen value |
|------|----------------|
| Primary seeds (after hash) | **`[20260921, 20260922, 20260923, 20260924, 20260925]`** |
| Bootstrap | 10 000 paired-seed percentile 95% CI; dual rule wherever §1 requires it |
| **\(C_\mathrm{exp}\)** | **= 8** for all medium / random-tag arms **by design** (every expert writes) |
| \(C_\mathrm{route}\) (medium) | **896** demod FLOPs/token (`128 × (6+1)`); never combine with LR MACs |
| \(C_\mathrm{route}\) (plain_sum) | **1024** MACs (LR Linear 128→8) |

---

## 3. Measures (exact)

| Symbol | Definition |
|--------|------------|
| \(L\) | Mean content-token NLL on heldout (markers excluded) |
| entropy_floor_nats | Content→content mean conditional entropy (T7 entropy helper); reported beside \(L\) |
| \(C_\mathrm{route}\) | Native routing/demod cost (MACs **or** demod FLOPs — never collapsed across kinds) |
| \(C_\mathrm{exp}\) | Active experts writing per token; medium = **8** by design |
| \(\mathrm{Rec}_e\), \(\min\mathrm{Rec}_e\), \(\mathrm{Rec}_\mathrm{least\_used}\) | `measures_t9.expert_recall` |
| \(G\), \(G_\mathrm{sum}\) | Coop gain; \(G_\mathrm{sum}\) from `plain_sum` |
| \(X(\Delta\varphi)\) | Recovered demod scale vs \(\max(0,\cos\Delta\varphi)\) |
| PLV | \(\lvert\mathrm{mean}\,\mathrm{e}^{i(\varphi_a-\varphi_b)}\rvert\) |
| \(A(\sigma_c)\) | Recovered-amplitude ratio under shared drift vs clean |
| \(S(\sigma_j)\) | \(E_\mathrm{on}/(E_\mathrm{on}+E_\mathrm{off})\) under independent jitter |
| \(E_\mathrm{coh}\) | Coherent energy fraction of complex buffer (diagnostic) |

Harness: `harness_t9.py` · `t9_routing/measures_t9.py` · layer `t9_routing/layer.py`.

---

## 4. Primary comparison rule / budget

1. After parent writes `t9/results/PREREG.sha256`, run primary eval (and any allowed non-expert training recipes disclosed at train entrypoint) on seeds `20260921..20260925` into `t9/results/primary/`.
2. Score H1–H4 exactly as §1, including the phase vs random-tag attribution rule.
3. Report **supported**, **H0**, or **mixed** per hypothesis.
4. **No multi-seed before hash.** **No** retune of T8 hum λ / a₀ / coupling; **no** retrain of frozen T1 experts; **no** edit of T7/T8 counting artifacts.
5. **Budget:** total new GPU (or CPU-equivalent wall) work for T9 primary **&lt; 6 GPU-h**. Prefer eval-on-frozen over large retrain grids. Do not expand to a full multi-arm retrain grid that would blow the budget.

---

## 5. Rules of the road

1. Do not edit parent `../PREREG.md`, `../results/primary/`, closed `../t7/`, or closed `../t8/` counting artifacts.
2. **Experts frozen for primary** — no silent expert-bank retrain.
3. **`random_tag` is the mirror** of structured phase (same amplitude / ReLU path, no phase geometry).
4. **ReLU is load-bearing** for H2 / selective readout (`y=\mathrm{ReLU}(\mathrm{Re}(\cdot))`). Do not drop it to “improve” clean \(L\) without a new prereg.
5. Fix parameters from T0–T5 one-seed / single-system calibration only; disclose all peeks (§6).
6. Same data splits and seeded disturbances across arms within a seed/cell.
7. Cost channels stay in native units (demod FLOPs ≠ LR MACs).
8. T10 reconciliation across T7/T8/T9 scoreboards remains **held**.
9. Biology / communications metaphors are motivation only (not evidence).

---

## 6. Mandatory disclosure — everything seen before freeze (non-counting)

### 6.1 T0 — import / T8-hum bitcheck (seed `20260921`)

T9 re-eval of T8 hum ckpt: heldout \(L\) = **1.273584770038724** (bit-for-bit vs T8; abs Δ = 0). Pytest: 9 passed.

### 6.2 T1 — coop + plain-sum freeze (seed `20260921`)

- Frozen experts sha256 **`2ef3bfa5…888d9eb`**
- Coop MANIFEST file sha256 **`aeb60b5a…7d90c9`**
- Ordinary L = **1.296938**; specialization **8/8**; \(L_\mathrm{coop\_both}\) = **0.828313**
- mean ΔL silence A/B = **0.127165** / **0.375466** (pair_matters)

### 6.3 T2 — buffer / demod

Synthetic separation sweep traces \(\max(0,\cos)\) (H2 plumbing). Expert norms frozen in `results/t2/expert_norms.json`. Pytest: 36 passed (7 in `test_t2_medium.py`).

### 6.4 T3 — `φ_ref` / hum params

λ = **0.02548046875**, a₀ = **0.1**, ρ(C) ≈ 0.015 — **unchanged** from T8. Shared-drift PLV gates passed (hum 0.843 / fixed 0.096).

### 6.5 T4 — random-tag mirror

Orthogonal tags, seed `20260921`; projection + ReLU mirrors phase half-wave. Pytest: 29 passed (8 in `test_t4_random_tag.py`).

### 6.6 T5 — harness smoke (seed `20260921`, **plumbing only**)

Smoke under `results/exploratory/t5_harness_smoke/` (73 JSON files). **Medium-arm heldout \(L\) is high vs plain_sum** — expected: **no medium-path train**; smoke validates measures / cells, **not** H1.

| Arm | Smoke ordinary \(L\) | \(C_\mathrm{route}\) | \(C_\mathrm{exp}\) |
|-----|---------------------:|---------------------:|-------------------:|
| `plain_sum` | **1.2969** | 1024 MACs | ~2 |
| `phase_hum` | 3.4410 | 896 demod FLOPs | **8** |
| `phase_fixed` | 3.2917 | 896 | **8** |
| `phase_hp_hum` | 3.5398 | 896 | **8** |
| `random_tag` | 1.5375 | 896 | **8** |
| `phase_multislot` | 2.2011 | 896 | **8** |

Smoke \(G(\texttt{plain\_sum})=G_\mathrm{sum}\approx 0.251\); medium arms reported \(G=0\) in this untrained plumbing run — **non-counting**. Separation MAE vs \(\max(0,\cos)\) ≈ \(9\times10^{-9}\) (synthetic). Pytest: 52 passed (7 in `test_t5_harness.py`).

### 6.7 What was **not** seen

**No multi-seed T9 primary** was computed before writing this file. No `PREREG.sha256`. T8 hum λ not retuned; T1 experts not retrained.

---

## 7. What is NOT frozen yet / known gaps (TBD)

| Gap | Note |
|-----|------|
| **Medium-path training recipe** | T5 smoke used frozen plain-sum experts with medium FFN swap and **no** medium-path train (high \(L\)). Whether primary trains any non-expert adapters / readout / router replacement — **TBD** at train entrypoint; must not unfreeze expert FFNs without amendment |
| **LM-level \(A(\sigma_c)\) vs unit probe** | Harness records both; primary H3 call uses §1 (hum vs fixed dual bootstrap on \(A\) at \(\sigma_c=0.05\) + PLV gates). If LM \(A\) is degenerate under current wiring, fall back to disclosed probe \(A\)+PLV and mark **mixed** — **TBD** confirm on first primary seed |
| **Exact Hum \(C_\mathrm{route}\)** when HumRef runs live | Demod FLOPs (896) frozen for medium readout; any extra hum-controller MACs documented in native units at first primary run — do not invent a collapsed score |
| **`phase_multislot` H1 role** | Capacity / interference probe; **not** an H1 comparator |
| **H4 pairing vs T8 seeds** | T8 seed-means are fixed citations; T9 bootstrap is over T9’s 5 seeds vs those constants (one-sample dual rule on \(D_i = \min\mathrm{Rec}_e^{(i)} - c\)) |
| **PREREG.sha256** | Parent writes after review — **this agent must not** |
| **Parent handoff truncation** | `docs/HANDOFF.md` notes full parent equations/figure list were not fully inlined; this PREREG freezes from T0–T5 accepts + HANDOFF/COOP/REFERENCE — do not invent further numeric freezes beyond §1–§2 |

---

## 8. Freeze procedure

1. Parent reviews accept docs T0–T5 + this PREREG + `READY_TO_FREEZE.md`.
2. Parent writes `t9/results/PREREG.sha256` = sha256(this file) + `created_iso` (PT) and anchors on Drive (no public gist).
3. Only then run multi-seed primary into `t9/results/primary/` per §4.

---

## 9. Freeze statement

Editing H1–H4 thresholds, primary measures, frozen expert hash, hum λ / a₀ / coupling, demod ReLU, coop CRT rule, or the phase vs random-tag attribution rule after peeking at T9 multi-seed outputs voids this preregistration for that run. T7 and T8 remain the frozen counting runs for their respective comparisons; T9 is a separate phase-as-medium follow-on.
