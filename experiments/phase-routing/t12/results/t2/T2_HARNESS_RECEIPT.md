# T12 T2 Harness Receipt

- **Status:** **PASS**
- **PT:** 2026-09-23 19:16:09 PT
- **Smoke seeds:** [20260921, 20260922] (of draft five [20260921, 20260922, 20260923, 20260924, 20260925])
- **Init family:** `pcg64_t12_v1`
- **Cells:** clean, noise, shift, silence
- **Parity:** PASS
- **Silencing (zero/hold + disclosure):** PASS
- **Silence → prior (not live hard):** True
- **Shift lock path δ=0.3:** PASS

## Silencing

- Estimate **may compute** for disclosure.
- **Zero/hold at controller input** — not fed into budget decision.
- Live hard ACT depth=10; silenced ACT depth=7.

## Notes

- Same seed → same sequences for all arms (parity harness).
- Silence cell keeps sequences clean; gate zeros/holds regime channel.
- Affinity / C₃ / hum **untouched**. No PREREG.sha256.
