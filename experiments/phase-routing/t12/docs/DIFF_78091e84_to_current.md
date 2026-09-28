# T12 PREREG DRAFT content diff — `78091e84…` → current

| Field | Value |
|-------|--------|
| **Date** | 2026-09-23 PT |
| **Status** | DRAFT chain (no `PREREG.sha256` emit) |
| **Original tip** | `78091e84b88e006fb27dc3f7648d0e545387060ab96140f6afe6567c232a6977` |
| **After H4/T1 amend** | `99afbb3becedd221804da6526bd50e682c1b3124e8e67cfd6f428b750ff3567b` |
| **Current (C̄=7 amend)** | `8e04b801d9da5e1c94221ecff07d3555dc3c4f630599434160173f61388ffbd1` |
| **File** | `docs/T12_PREREG_DRAFT.md` |

Affinity / C₃ / hum **untouched**. T12b **untouched**.

---

## Recoverability note (a)

Exact bytes of tip `78091e84…` are **not** in box git history: the first content commit
(`16c7c90`) already stored the post-amendment draft at `99afbb3b…`. T0 receipts still
cite `78091e84…` as the pre-amend tip. Reconstructing a byte-identical original by
stripping the amendment blocks did **not** re-hit `78091e84…`, so section (a) below is a
**content summary** from the amendment callout + residual wording in `T12_HANDOFF.md`,
not a recoverable unified diff of original bytes.

Section (b) is an exact tip-to-tip content sha transition (`99afbb3b…` → `8e04b801…`).

---

## (a) `78091e84…` → `99afbb3b…` — H4 / T1 / scoring / grid (2026-09-23 PT)

Kyle + GNURadio amendment (recorded in draft callout). Summary of intent locked:

### H4 zero/hold (operational silencing)

- **Before (handoff / pre-amend spirit):** H4 = regime estimate “silenced”; coordinator
  persistence vs stateless regime under silence.
- **After:** Operational cut at the **controller input**: estimate **may still be
  computed** for disclosure/receipts, but is **zeroed or held** so it is **not fed into
  the budget decision**. Exact zero-vs-hold recipe free at T5 freeze.
- **Null clause 3** tightened to name zero/hold at controller input.
- New §3.4 Silencing operational lock + §3.4.1 scoring note.

### Scoring point (T11.3 fix retained)

- Primary H4 / SD comparisons use **step-grain** metrics on steps **after the evidence
  point**. Silencing does not move the scored window.

### T1 acceptance checks (gate before T2–T4)

- New §4.2.1:
  - Fixed-depth sweep 1–16, no controller → measurable easy/hard gap (H1).
  - Overthinking signature (H3): easy degrades past optimum while hard does not yet;
    no crossover / plateau-only → **FAIL / stop** (H3 untestable).
  - Stateless ACT + Convergence stubs required in the **first T1 commit**.
- T1 ticket row expanded to require §4.2.1 PASS before T2–T4 spend.

### Grid

- Disturbance grid intent unchanged in spirit: **clean + noise + shift (T8 regression)
  + silence** cells for H4; exact σ/δ at freeze. T2 harness later wired these four
  cells with parity across arms.

### Unchanged in (a)

- Status **DRAFT**; no `PREREG.sha256`.
- Affinity / C₃ / hum locks (n=2, HP every drive, amplitude never phase-gated).
- Calibration still drafted as \(\bar C = 6 \pm 0.25\) at this tip (pre–Option-1).

---

## (b) `99afbb3b…` → `8e04b801…` — Option 1 / T1 budgets (2026-09-23 PT)

Second amendment callout added. Calibration locks recalibrated to T1:

| Item | Was (`99afbb3b…` / T3 soft) | Now (`8e04b801…`) |
|------|-----------------------------|-------------------|
| \(\bar C\) target | **6 ± 0.25** | **7 ± 0.25** |
| Fixed T | **6** | **7** |
| ACT / BP centers | **4 / 8** (soft-hard-to-8) | **4 / 10** (= T1 T*) |
| Soft-hard-to-8 | present (equal-mix → 6) | **removed** |
| ACT prior_mean | 6 | **7** |
| CONV τ easy/hard | 1.335 / 2.670 (≈4/8) | 1.335 / **3.337** (≈4/10) |
| Status | DRAFT | **DRAFT** (still no freeze hash) |

Code locks live in `controllers/calibration.py`. T3/T4 (and T2 silence-prior depth)
re-smoked under the new values; receipts under `results/t3/`, `results/t4/`
(and `results/t2/` for prior=7 parity).

### Readable hunks (draft text)

```diff
- | \(\bar C\) | **6 ± 0.25** |
+ | \(\bar C\) | **7 ± 0.25** |
+ | Fixed T | **7** (matched-mean baseline) |
+ | ACT / BP centers | **easy=4 / hard=10** (= T1 T*; soft-hard-to-8 removed) |

- | Fixed T=6 | Matched-mean baseline near calibration |
+ | Fixed T=7 | Matched-mean baseline near calibration |

- | **T3** | Calibration \(\bar C=6\pm0.25\); lock ε/ACT/threshold. |
+ | **T3** | Calibration \(\bar C=7\pm0.25\); lock ε/ACT/threshold to T1 T*=4/10; Fixed T=7. |

- vs Fixed T=6 under matched \(\bar C\)
+ vs Fixed T=7 under matched \(\bar C\)
```

---

## Tip chain

```text
78091e84…  (pre-amend; bytes not in box git)
    │  (a) H4 zero/hold + T1 checks + scoring point + grid note
    ▼
99afbb3b…  (first git-stored tip)
    │  (b) C̄ 6→7; BP/ACT 4/10; Fixed T=7; soft-8 removed
    ▼
8e04b801…  (current DRAFT content sha)
```

