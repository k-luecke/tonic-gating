# T13 Rule 19 — freeze gates

**Status:** Live 2026-09-24 ~08:30 PT (G13 measurement-procedure gate defect). Affinity / C₃ / hum **FROZEN**.  
T11 / T12 / T12b `PREREG.sha256` files **untouched**. Prior freeze `12305912…` **superseded** by this gate defect; re-emit required.

**Freeze hold:** Do **not** emit `PREREG.sha256` until (a) matched-compute formula + attention choice locked in PREREG §3, (b) H_int algebra + verdict (incl. no-medium clause) locked before arms, (c) all **freeze-blocking** gates PASS (G13 Acc + G17 Acc are **deferred post-train** — criterion locked now; do **not** block emit), (d) Kyle freeze greenlight.

## Gates (freeze-blocking must PASS before emit; Acc gates deferred post-train)

| # | Gate | Check | Fail action |
|---|------|-------|-------------|
| **G1** | **Status is not DRAFT (at emit)** | Emit path refuses while `docs/T13_PREREG_DRAFT.md` still contains **DRAFT** status; freeze only after authorize + promote | Block emit |
| **G2** | **Matched-compute formula present** | PREREG contains explicit cost formula (loops × experts × width + attention term); all arms declare cost-match to **primary** line | Block freeze |
| **G3** | **Primary = H_int algebra before arms** | PREREG places **H_int** algebra (aggregate recovery \(\hat{c}_e\), \(\widehat{\Delta L}_e\) vs \(\Delta L_e\), Pearson \(r\)) **before** the arm grid; arms reference the metric; depth budgeting labeled solved control | Block freeze |
| **G4** | **Related-work delta locked** | Cites Mixture-of-Depths (Raposo et al., 2024) + Mixture-of-Recursions (Bae et al., 2025); delta = multi-expert loops **own budget + own context slice** at matched compute | Block freeze |
| **G5** | **Hum allocator expected-null** | PREREG states hum-as-allocator arm is **control / expected-null** (coordinator may be stateless; we do **not** expect hum to win) | Block freeze |
| **G6** | **No Affinity retune** | `affinity_inputs.py` sha = `7dae877a3975c1ba4ca264f03f572d6749cce39a114ea3adbcffc010c7852c50`; no local Affinity copy/edit | Block freeze |
| **G7** | **Parent PREREGs untouched** | T11 / T11.2 / T11.3 / T12 / T12b `PREREG.sha256` contents unchanged | Block freeze |
| **G8** | **Scale = mythos_tiny** | Config / PREREG lock scale `mythos_tiny` + DRAFT numeric locks (K=4, W=64, L_max=4, S_full=64, S_e=16, vocab=256, 2000 steps, seeds) | Block freeze |
| **G9** | **Training-instability check (docs)** | T12 T1-style check language present (NaN / non-finite / explosive grads). **Runs at train**; **refuse counted primary** if FAIL | Docs PASS at freeze; refuse primary at train |
| **G10** | **Stopping + null/support vocabulary** | One preregistered primary (H_int) × locked seeds; verdict embeds no-medium; tagged_superposition / medium_not_beyond_attribution honesty; no silent ScoreAffinity carry | Block freeze |
| **G11** | **Affinity surface default off** | No Affinity harness work unless a later named gate; default = no Affinity arm surface | Block freeze |
| **G12** | **H_int controls named** | PREREG names **random-tag**, **plain-sum** (floor), **oracle** (ceiling), and points to no-medium (G16) | Block freeze |
| **G13** | **Crosstalk bound (procedure locked)** | **Manipulation check only** (not H_int evidence). Tolerance \(|X-\max(0,\cos)|\le 0.1\). **Δφ set:** pairwise hashed phases ∪ **N=36** grid on [0,2π). **Aggregation:** hard **max-deviation** (not RMS). Evaluate **post-train / pre-primary** — does **NOT** block freeze; blocks counted primary if FAIL. Do not fake Acc. | Docs procedure PASS for freeze; Acc deferred |
| **G14** | **K + phases hashed** | \(K\) experts and their phases fixed and hashed before hash emit (draft \(K=4\)) | Block freeze |
| **G15** | **C_match ±5% per arm before hash** | **Primary** \(C_{\mathrm{match}}\) (per-slice attention counted) evaluated per arm within **±5%** before emitting `PREREG.sha256` | Block freeze |
| **G16** | **No-medium inside H_int verdict** | Control (4) at **per-expert, per-example** granularity (activation-norm or grad×input, no medium). **Support requires** \(r_{\mathrm{phase}} > r_{\mathrm{no\_medium}}\) by more than the seed interval — not secondary-only. If only named outside the verdict → **FAIL** | Block freeze |
| **G17** | **Tag-placement decodability (T10)** | Probe on **tag placement alone** (expert content removed or shuffled) → expert identity. **n=1000**; chance \(p_0=0.25\); **PASS if** Acc \(\le \mathbf{0.273}\) (= one-sided 95% binomial UB ≈0.2725). **Retired:** n=200 / Acc≤0.30. **Docs-assert PASS OK for freeze**; actual probe Acc runs **post-train / pre-primary** — **FAIL / block counted primary** if Acc \(> 0.273\). **G12–G15 do not cover this.** | Docs PASS for freeze; Acc deferred |
| **G18** | **Attention-accounting decision locked** | PREREG §3.0 states **primary = per-slice attention counted** with per-arm loops×experts×width+attn lines (§3.2); **secondary = full-context-equivalent**. Chosen **before** mythos_tiny numerics / before arms look better under one choice | Block freeze |

## Script

| Role | Path |
|------|------|
| Rule 19 runner | `scripts/rule19_freeze_gates.py` — live; `--emit-hash` promotes + emits when freeze-blocking PASS |
| Primary placeholder | `scripts/primary_run.py` — empty stub |

```bash
python3 experiments/phase-routing/t13/scripts/rule19_freeze_gates.py
# --emit-hash only after Kyle freeze GO + freeze-blocking gates PASS
```

## Relation to closed parents

T12 closed ACT≈oracle (depth control solved at toy). T12b closed gaps mixed (smoothing-class secondary). T10 random-tag 1.000 via placement leak → **G17** required. T13 opens a **new line** — do not re-run T12/T12b seals as retune; do not re-emit parent PREREGs. Primary = **H_int** (intelligibility with no-medium inside verdict), not depth, not hum allocator. **Post-freeze sequence:** train mixture (G9) → G13 Acc + G17 Acc → primary × seeds.
