# controllers/ — T12 arms

| Arm | Status |
|-----|--------|
| ACT | **T1 stub** `act_stub.py` — deepened T3 (calibration locks) |
| Convergence threshold | **T1 stub** `convergence_stub.py` — deepened T3 |
| Fixed T=7 | **T2** `fixed_stub.py` |
| Coordinator regime | **T2** `coordinator_stub.py` |
| Stateless regime | **T2** `stateless_regime_stub.py` (H4 control) |
| Silencing gate | **T2** `silencing.py` — zero/hold at controller input |
| Calibration locks | **T3** `calibration.py` — C̄=7±0.25; ACT 4/10; Fixed T=7 |

Stubs emit depth choices with the same I/O shape as future trained arms. Not trained.
Affinity / C₃ / hum **untouched**.
