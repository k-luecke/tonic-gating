# SMOKE_HUM_N2 — T12 easy/hard (n=2)

**Verdict: PASS**

- Created: 2026-09-23T18:28:54 PT
- Dynamics: `/workspace/scp/experiments/phase-routing/t11/hum` (frozen; sha `03ab651db94e…`)
- C n=2 sha256: `c7bf348a38fffc30217cd657ff00f2fb3ab6cf316ea4cbfc0ea01f46b87a545f`

## Locks

| Lock | Status |
|------|:------:|
| hum n=2 (easy/hard) | yes |
| high-pass every drive (HP_PERIOD=64) | yes |
| amplitude readout never phase-gated | yes |
| Affinity `__all__` / C₃ / T11 n=3 untouched | yes |

## Persistence

| Probe | T_hold |
|-------|-------:|
| Homogeneous 1/e (no HP) | 39.0000 |
| Coupled diagnostic (no HP) | 62.0000 |
| Operating + HP easy (unit 0) | 9.0000 |
| Operating + HP hard (unit 1) | 9.0000 |
| 1/λ (frozen) | 39.2457 |

Homogeneous hold matches T8 acceptance family (target ≈41, tol 5).
Operating numbers use roofing high-pass on every drive channel.
