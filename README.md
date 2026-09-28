# Phase Selects; It Does Not Amplify (Tonic Gating)

**Repository**: `driftbound-labs/tonic-gating`  
**Author**: Kyle Luecke (ORCID: [0009-0008-0202-7852](https://orcid.org/0009-0008-0202-7852)), Driftbound Labs  
**Paper Link**: [Preprint PDF](paper/Paper_2_LaTeX.pdf) · DOI: `10.5281/zenodo.pending`  
**Status**: `v1 preprint. Replication pending; results may be revised in v2.`

---

## 1. Summary

Recordings of prefrontal cortex describe a division of labor in which slow rhythms coordinate which information is active while synapses store it: beta as a spatial "stencil" of inhibition that holds a task set and is interrupted by gamma bursts carrying content. We asked whether this division of labor does useful work when built from arithmetic inside a small mixture-of-experts transformer, and we preregistered every test. Across twelve experimental stages on a synthetic rule-following task, three confirmatory findings survive:

1. **Persistent Coordination**: An always-on selection state with fixed dynamics holds an expert-selection regime after control input stops ($T_\text{hold} = 26.8$ tokens at $n=8$ and $65.0$ tokens at $n=3$, effective spectral radius $\rho_\text{eff} = 0.989$), compared to under 2 tokens for a stateless controller. Hold time is insensitive to intrinsic unit leak alone (flat at 28.5 / 26.8 / 24.2), indicating competitive survival under drive.
2. **Selective Physical Law**: When experts write into a shared complex-valued buffer tagged by phase and a reference demodulates it, selection follows the predicted cosine law: crosstalk scales as $X(\Delta \varphi) \approx \max(0, \cos \Delta \varphi)$, drifting references maintain selection, and usable phase separation follows the slot law. Coherent experts do not, however, amplify or add ($G_\text{medium} = 0.044$ vs $G_\text{sum} = 0.251$).
3. **Lookup Dominance under Continuous Evidence**: When evidence is continuously available, persistent coordination provides no benefit over lookup. In looped depth budgeting (T12), stateless ACT matched the oracle budget ($\Delta \text{ACC} = 0$), leaving the persistent coordinator nothing to add; and under mid-gap lures (T12b), held state beat one-step memory (0.6525 vs 0.2987) but fell into the preregistered smoothing class tied by $k=3$ majority lookup (0.6390). In looped architectures (T13), phase superposition provides a positive payoff as an interpretability and attribution medium ($r_\varphi = 0.87$ vs $0.80$ random tags).

---

## 2. Experimental Code & Distribution Map

All twelve stages are accounted for in this repository:

| Stage | Focus / Contribution | Runnable Code Included | Location |
|---|---|---|---|
| **Oscillator Sheet & Receiver** | Lock-in demodulation & spatial stencil | **YES** (Complete simulation + tests) | `experiments/gating/` (`sheet.py`, `lockin.py`, `stencil.py`, `run_comparison.py`) |
| **T0–T4** | Oscillator baseline, hash router, phase gate | **YES** (Complete harness + tests) | `experiments/phase-routing/t0_t4/` (`train.py`, `train_phase_gate.py`, `harness.py`, `phase_routing/`) |
| **T5 / T6** | Phase gate, task supplied | **YES (Core)** | Architecture in `t0_t4/`; data generation in `t0_t4/` |
| **T7** | Phase gate, task inferred (`LR-smoothed`) | **YES** (Frozen harness) | `experiments/phase-routing/t7/` (`t7_routing/`, `train.py`, `harness.py`, `costs.py`) |
| **T8** | Always-on state persistence ($T_\text{hold}$) | **YES** (Verified acceptance rerun harness) | `experiments/phase-routing/t11/hum/acceptance_t8.py`, `t12/scripts/run_t8_acceptance.py` |
| **T9** | Phase medium / ensemble test | Recipe, scoreboards, and summary data | `experiments/phase-routing/t9/` (Recipe and disclosures bundled; training script was hosted on external box) |
| **T10** | Coherence as classifier (Retracted) | Evaluation scoreboards & calibration | `experiments/phase-routing/t10/` (Scoreboards and diagnosis bundled; training script was on external box) |
| **T11** | Specialist execution & ambiguity | **YES** (Full harness) | `experiments/phase-routing/t11/` (`generators/`, `hum/`, `routing/`, `scripts/`) |
| **T12** | Looped depth budgeting & overthinking | **YES** (Full harness) | `experiments/phase-routing/t12/` (`controllers/`, `generators/`, `harness/`, `run_t6_primary.py`) |
| **T12b** | Evidence gaps & mid-gap lures | **YES** (Full harness) | `experiments/phase-routing/t12b/` (`scripts/arms.py`, `gap_inject.py`, `primary_run.py`) |
| **T13** | MoE-of-loops / context slices ($H_\text{int}$) | **YES** (Full primary harness) | `experiments/phase-routing/t13/` (`train_mixture.py`, `primary_run.py`, `mythos_model.py`) |

*Note on T12 Scaffolding*: In T12, controller files carry the `_stub.py` suffix (`act_stub.py`, `coordinator_stub.py`, `oracle_stub.py`). These are the actual analytical implementations evaluated in `run_t6_primary.py`. Their parameters (`easy=4`, `hard=10`) were pre-calibrated and frozen via Rule-19 Gate T5.

---

## 3. Cryptographic Pre-commitments & Blockchain Proofs

Every preregistration was locked prior to inspecting outcome data. Preregistrations, SHA-256 sidecars, and OpenTimestamps (`.ots`) proofs are located in [`prereg/`](prereg/).
* Because experimental executions took place across distributed research boxes, blockchain anchoring was completed on 2026-09-24 as an explicit backfill (`anchored_retroactively`).
* Verify an anchor:
  ```bash
  ots verify prereg/PREREG_T12.md.ots
  ```

---

## 4. License & Citation

* **Code**: [MIT License](LICENSE)
* **Documentation, Preregistrations & Results**: [Creative Commons Attribution 4.0 International](LICENSE-docs)
* **Citation**: See [`CITATION.cff`](CITATION.cff).
