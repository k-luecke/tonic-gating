# CLOSING — T11 Model-Level Orchestration (routing-only dry run)

| Field | Value |
|-------|--------|
| **Date** | 2026-09-23 PT |
| **Status** | **closed / not informatively tested** on H1–H2 (generator); H3 **supported**; H5 **not run** |
| **Run class** | **Routing-only dry run** of orchestrators on sealed sequences — **not** the full T11 primary PREREG describes |
| **PREREG** | FROZEN `c62bfda8bab88df14677e47e20f28c99ba20595bc0f9273050484d4ae92e079a` |
| **HANDOFF** | `056be9628d3a07edd9fa0a2bff32b7e01ebb8fdd9a36b7b66d0ec5f7e39c8eaf` |
| **Primary receipt** | `t11/docs/T11_ROUTING_PRIMARY_RECEIPT.md` sha `ecb96c67eccb689095366e1cdef0bb8d85a41e9ca7e5c42a0e4d0ad20e24bc36` |
| **Wall** | ~0.66–0.7 s — no specialist execution |

## Verdicts (Kyle close 2026-09-23)

| H | Verdict | Why |
|--|---------|-----|
| H1 | **not informatively tested** | Hand rule M1=**1.000** on Ambiguous — family decodable from first tokens; segments were not ambiguous. Handoff named the check (first-32 classifier near chance on Ambiguous before hash) and it was skipped. |
| H2 | **not informatively tested** | Distractions without other-family cues do not tempt a stateless arm to switch; zero thrash everywhere is the **generator**, not the orchestrators. |
| H3 | **supported** | Keep this clause. Strong keyword evidence swamped the 65-token hold; predicted stubbornness did not appear. Real finding. |
| H4 | report-only / uninformative under same generator | — |
| H5 | **not run** | typed_goal_family omitted; no full SCP combo. |
| H6 | report-only on routing dry run | No specialist execution / carry under live swaps. |

**Reframe of primary receipt labels:** receipt said H1 null / H2 inconclusive_zero_thrash. Closing status is **not informatively tested** for both (Kyle). Persistence trade-off predicted pre-freeze was **not** observed — for the generator reason above, not a hum failure.

## What the dry run does say

On sequences a keyword lookup solves, integrating evidence through a 65-token hold only adds lag: hum M1≈**0.813** under hand rule **1.000**. That is expected once Ambiguous/Clean are first-token-decodable.

## What it does not say

- No specialist executed anything: no task outcomes, no swaps, no receipts, no invalid-action rate (M5).
- classifier_mlp train/validation overlap on four of five primary seeds — disclosure caught it; pooled numbers still carry the leak.
- This is **not** the T11 primary the preregistration describes. Label: **routing-only dry run**.

## Process note (second skip)

This is the **second** time a freeze went through with a check the handoff asked for left undone (T10 side-channel / oracle channel; T11 Ambiguous near-chance validation). **Fix the skipped check, not the hum.** Plumbing (sanitize gate, hum_measure import, ρ_eff / T_hold(n=3), schema) is right; cost was under a second of compute and an afternoon of wiring.

## T11.2 — four gates before hash

1. **Ambiguity validated:** hand rule **and** first-32-token classifier both **near chance** on Ambiguous segments (handoff check, enforced).
2. **Distractions carry other-family cues** (so thrash is possible for stateless arms).
3. **Classifier trained on disjoint seeds** only (no train/val overlap with primary seeds).
4. **Full harness with specialists executing** — swaps and receipts exist; typed-goal **family** active (H5 path live).

Do not freeze T11.2 until all four pass.

## Artifacts (this close)

- Primary RESULTS: `experiments/phase-routing/t11/routing/results/primary/`
- This CLOSING + sha below
- Hum / measure inventory unchanged; no retune

## Counts

PREREG + sealed batch **untouched** by this close. Relabel status only.
