# Handoff excerpt — Oscillatory Gating vs Learned Router

**Source:** Handoff doc (2026-09-21). Condensed for local work; full text lives on Drive.

## Mission

Preregistered comparison: oscillatory gating (spatial stencil + phase pilot) vs learned top-k router on the same patches. Report selectivity, leakage, switching latency, and cost under unit mismatch, timing jitter, and shared drift.

- **H1:** Gate matches router on selectivity/leakage at lower cost when clean, and degrades more gracefully under shared drift (pilot shares drift with senders).
- **H0:** Router wins on cost, or gate advantage vanishes at realistic mismatch/jitter (5–10% freq spread, 5–10% cycle jitter).

Out of scope this sprint: consolidation, learning in the sheet, hardware ports.

## Lock-in findings (what `lockin.py` must reproduce)

Single-receiver model, two senders @ 40 Hz, fs = 4 kHz, 5-cycle symbols, rectified gain `max(0, cos φ_ref)`, threshold decision. PLL: first-order, K=800, τ=3 cycles. Modes: fixed / content-locked / pilot-locked.

1. **Slots.** Pulsed senders: BER(A) < 5% once B is ~20° away @ 10 dB, ~25° @ 0 dB, ~40° @ −6 dB → **9–18 slots/cycle**.
2. **Positive-only is load-bearing.** Signed sinusoids → U-shaped BER (180° as bad as 0°). Pulsed `max(0,cos)²` → strong anti-phase rejection.
3. **Fixed reference fails under mismatch/drift.** e.g. 0.25 Hz offset over 50 s → BER → chance.
4. **PLL must lock to pilot, not content.** Content-locked PLL follows whoever transmits → selectivity collapses. Pilot (amp 0.5, shared drift) holds BER ≈ 0 across drift and 0–8 Hz mismatch at K=800.

## Design constraints for the sheet

- Content = positive volleys only (never signed carriers).
- Each patch PLL-locks to a **control** pilot, never to the summed content.
- Pilot must share disturbances with senders for the drift advantage.
- Freeze K_p, τ before router comparison.

## Tickets T0–T6 (brief)

T0 scaffold · T1 jitter · T2 sheet.py · T3 stencil.py · T4 router.py · T5 harness · T6 PREREG freeze.
