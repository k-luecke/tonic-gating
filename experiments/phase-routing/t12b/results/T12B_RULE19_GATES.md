# T12b Rule 19 — gate receipt

- **PT:** 2026-09-23 20:00:23 PT
- **Verdict:** **PASS**
- **Draft content sha256:** `408eaaaca59d0c210e1b1ef50d2494213662da1059b3542c38ce866651801b21`
- **PREREG.sha256 emitted:** True
- **PREREG.sha256 path exists:** True

## Gates

| Gate | Result | Detail |
|------|:------:|--------|
| `G1_gaps_zero_affinity` | PASS | gap_inject --self-test rc=0 |
| `G2_one_step_memory_present` | PASS | arms=['hum_measure', 'stateless_argmax', 'one_step_memory', 'majority_k3']; PRIMARY_ARMS=['hum_measure', 'stateless_argmax', 'one_step_memory'] |
| `G3_crossover_prediction_prerun` | PASS | cites_thold_65=True; no_primary_outputs=True |
| `G4_no_c3_hum_affinity_retune` | PASS | Affinity/C₃/hum shas match frozen baseline |
| `G5_t11_3_hash_untouched` | PASS | got=27ab4a9573ec8c01… want=27ab4a9573ec8c01… |
| `G6_t11_2_hash_untouched` | PASS | got=593762c4c19e85d1… want=593762c4c19e85d1… |
| `G7_sealed_reuse_not_rewritten` | PASS | symlink=True; target_ok=True; all_sha=646df30bbeacdc77… |
| `G8_routing_only` | PASS | forbidden_hits=[]; routing_claimed=True |
| `G9_stub_ne_counted_arm` | PASS | class=HumMeasureArm; counted=True; stub_refused=True |
| `G10_in_gap_scoreaffinity_lure` | PASS | flip_nz=True; surrounding_zero=True; mid_offset=3 |
| `G11_majority_k3_labeled_secondary` | PASS | present=True; labeled=True; k=3; not_primary=True; draft_docs=True |

## Notes

- Counted arm path = HumState (`HumMeasureArm`).
- Primary contrast = `hum_measure` vs `one_step_memory`.
- `majority_k3` = labeled secondary (G11).
- Affinity / C₃ / hum frozen. Parent T11.2 / T11.3 PREREG untouched.
- Freeze emit: see `prereg_sha256_emitted` field.
