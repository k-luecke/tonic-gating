# T13 Rule 19 — gate receipt

- **PT:** 2026-09-24 08:30:27 PT
- **Freeze-ready:** **True**
- **Summary:** PASS=18 FAIL=0 PENDING=0
- **Draft content sha256:** `1e3f7fe80c4763b827e1bb9cb566adf83551255c5bd4aad11073b91639ea0119`
- **PREREG.sha256 emitted:** True (`3176eb5ea3458047b820ff5511644d20a697a7a9485d5118560ff7b71e1f981d`)
- **Status:** FROZEN

## Gates

| Gate | Status | Detail |
|------|:------:|--------|
| `G1_draft_detected_blocks_emit` | PASS | draft_exists=True; is_draft=True; emit via --emit-hash promotes separate PREREG.md |
| `G2_matched_compute_formula` | PASS | markers=(True, True, True) |
| `G3_intelligibility_primary` | PASS | H_int algebra before arms |
| `G4_related_work_delta` | PASS | MoD=True; MoR=True; context_slice=True |
| `G5_hum_allocator_expected_null` | PASS | hum allocator expected-null clause present |
| `G6_affinity_frozen` | PASS | sha_match=True; no_local_affinity_copy=True; got=7dae877a3975… |
| `G7_parent_preregs_untouched` | PASS | all parent PREREG.sha256 match pins |
| `G8_scale_mythos_tiny` | PASS | mythos_tiny numeric locks present; checks={'mythos_tiny': True, 'K=4': True, 'W=64': True, 'L_max=4': True, 'S_full=64': True, 'S_e=16': True, 'vocab_256': True, 'steps_2000': True, 'seeds': True, 'C_match_numeric': True} |
| `G9_training_instability` | PASS | instability language locked; runs at train; refuse primary if FAIL |
| `G10_stopping_rule` | PASS | stopping + null/support vocabulary present |
| `G11_affinity_surface_default_off` | PASS | doc_ok=True; no_local_hum_measure=True |
| `G12_hint_controls_named` | PASS | random-tag / plain-sum / oracle / no-medium named |
| `G13_crosstalk_bound_assert` | PASS | procedure LOCKED (manipulation check; pairwise+N=36; max-dev≤0.1 not RMS); Acc PENDING post-train (does not block freeze; do not fake Acc) |
| `G14_k_phases_hashed` | PASS | draft K=4 + phases hash language present; K=True; phases=True |
| `G15_cmatch_per_arm` | PASS | §3.2 numeric C_match lines present; ref=True; shared=True |
| `G16_no_medium_inside_verdict` | PASS | no-medium inside H_int support verdict |
| `G17_tag_placement_decodability` | PASS | docs pin n=1000, p0=0.25, Acc≤0.273 (= one-sided 95% binomial UB ≈0.273); docs-assert PASS OK for freeze; probe Acc PENDING post-train (blocks primary if FAIL) |
| `G18_attention_accounting_locked` | PASS | primary=per-slice attn; secondary=full-context-equivalent; §3.0 locked |

## Notes

- Affinity / C₃ / hum **FROZEN** — affinity_inputs.py untouched.
- **G13** procedure LOCKED (manipulation check; pairwise+N=36; max-dev≤0.1 not RMS); Acc PENDING post-train — does **not** block freeze.
- **G17** docs pin n=1000 + Acc≤0.273 (= binomial UB); docs-assert PASS OK for freeze; Acc PENDING post-train.
- **G9** instability language locked; actual check runs at train; refuse primary if FAIL.
- Post-freeze sequence: train mixture → G13 Acc + G17 Acc → primary × seeds.
- Parent T11/T12/T12b PREREG.sha256 untouched.
