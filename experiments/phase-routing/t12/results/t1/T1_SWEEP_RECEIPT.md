# T12 T1 Sweep Receipt

- **Status:** **PASS**
- **PT:** 2026-09-23 19:06:03 PT
- **H1:** PASS (T*_easy=4, T*_hard=10, gap=6)
- **H3:** PASS (easy_degrades=True, hard_not_yet=True, plateau_only=False)
- **Stubs ACT+Convergence:** PASS
- **NLL@T* easy/hard:** 0.0110 / 0.7011
- **ACC@T* easy/hard:** 1.0000 / 0.9432

## Depth curves (NLL)

| depth | easy NLL | easy ACC | hard NLL | hard ACC |
|------:|---------:|---------:|---------:|---------:|
| 1 | 0.4596 | 0.8673 | 1.7472 | 0.7176 |
| 2 | 0.0554 | 1.0000 | 1.5108 | 0.9295 |
| 3 | 0.0168 | 1.0000 | 1.3085 | 0.9411 |
| 4 | 0.0110 | 1.0000 | 1.1401 | 0.9432 |
| 5 | 0.0117 | 1.0000 | 1.0039 | 0.9432 |
| 6 | 0.0139 | 1.0000 | 0.8974 | 0.9432 |
| 7 | 0.0193 | 1.0000 | 0.8192 | 0.9432 |
| 8 | 0.0284 | 1.0000 | 0.7634 | 0.9432 |
| 9 | 0.0515 | 0.9986 | 0.7255 | 0.9432 |
| 10 | 0.1181 | 0.9900 | 0.7011 | 0.9432 |
| 11 | 0.4763 | 0.9413 | 0.6864 | 0.9432 |
| 12 | 0.4774 | 0.9293 | 0.6782 | 0.9432 |
| 13 | 0.4623 | 0.9176 | 0.6741 | 0.9432 |
| 14 | 0.4516 | 0.9094 | 0.6727 | 0.9432 |
| 15 | 0.4416 | 0.9010 | 0.6727 | 0.9432 |
| 16 | 0.4473 | 0.8924 | 0.6727 | 0.9432 |

## Notes

- Fixed-depth sweep loops **1–16**, **no controller**.
- Toy: prelude/recurrent/coda spirit, width 64, numpy CPU, **no train**.
- ACT + Convergence threshold wired as **stateless per-item stubs** (dry callable).
- Affinity / C₃ / hum **untouched**. No PREREG.sha256.
