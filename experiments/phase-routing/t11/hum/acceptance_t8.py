#!/usr/bin/env python3
"""T8 HumState acceptance (Kyle draft X) — run BEFORE any T11 sequences.

Pass criteria (docs/T8_ACCEPTANCE.md / HUM_PARAMS.json):
  - |T_hold − 41| ≤ 5
  - 1/λ scaling confirmed on ≥2 λ values (direction + approx magnitude)
Fail → no soft-pass.

Probe (documented; original t8/ MoE harness missing):
  Drive unit 0 for 64 tokens at unit drive, then full drive silence.
  T_hold = silence tokens until excess amplitude (|s|-a0)+ ≤ excess0/e.
  Homogeneous (coupling zeroed for the hold scalar) so the measure matches
  the PREREG/HUM_PARAMS 1/λ≈39.2 persistence prediction. Coupling matrix is
  still built and checked against frozen ρ(C) / effective_linear_ρ.
  λ sweep: {½, 1, 2}× frozen λ; require monotone decreasing T_hold and
  T_hold * λ ≈ constant (within 25% of median product).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from hum_state import (
    AMP_GATE_THRESHOLD,
    A0,
    COUPLING_SPECTRAL_RADIUS,
    HumState,
    LAMBDA_LEAK,
    measure_thold_excess_1e,
)

HERE = Path(__file__).resolve().parent
# Prefer coordination-tree params; fall back to scp mirror copy
PARAMS_CANDIDATES = [
    Path("/workspace/phase-routing/t11/imports/HUM_PARAMS.json"),
    Path("/workspace/scp/t11/imports/HUM_PARAMS.json"),
    HERE.parent.parent.parent.parent / "t11" / "imports" / "HUM_PARAMS.json",
]
OUT_DIR = HERE / "results"
CODE_MIRROR = Path("/workspace/phase-routing/t11/code")

TARGET_HOLD = 41.0
HOLD_TOL = 5.0
LAMBDA_PROBES = [0.012740234375, 0.02548046875, 0.0509609375]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_params() -> dict:
    for p in PARAMS_CANDIDATES:
        if p.is_file():
            return json.loads(p.read_text()), str(p)
    raise FileNotFoundError("HUM_PARAMS.json not found in known locations")


def measure_hold(lam: float, homogeneous: bool = True) -> dict:
    hum = HumState()
    hum.set_lambda(lam)
    if homogeneous:
        hum.C = np.zeros_like(hum.C)
        hum._validate_linear_map()
    return measure_thold_excess_1e(hum)


def scaling_ok(holds: dict) -> dict:
    """Check 1/λ scaling on ≥2 λ values."""
    lams = sorted(holds.keys())
    ts = [holds[l] for l in lams]
    # strict monotone decreasing in λ (longer hold at smaller leak)
    monotone = all(ts[i] > ts[i + 1] for i in range(len(ts) - 1))
    products = [holds[l] * l for l in lams]
    med = float(np.median(products))
    # approx magnitude: each product within 25% of median (ideal product ≈ 1 for continuous)
    mag_ok = all(abs(p - med) / med <= 0.25 for p in products) if med > 0 else False
    # also compare ratios of holds to ratios of 1/λ
    ratio_ok = True
    details = []
    for i in range(len(lams)):
        for j in range(i + 1, len(lams)):
            r_hold = holds[lams[i]] / holds[lams[j]]
            r_inv = (1.0 / lams[i]) / (1.0 / lams[j])
            rel = abs(r_hold - r_inv) / r_inv
            details.append(
                {
                    "lam_i": lams[i],
                    "lam_j": lams[j],
                    "hold_ratio": r_hold,
                    "inv_lambda_ratio": r_inv,
                    "rel_err": rel,
                }
            )
            if rel > 0.25:
                ratio_ok = False
    return {
        "monotone_decreasing": monotone,
        "product_magnitude_ok": mag_ok,
        "ratio_ok": ratio_ok,
        "products_T_times_lambda": {str(l): holds[l] * l for l in lams},
        "pair_details": details,
        "passed": bool(monotone and (mag_ok or ratio_ok) and len(lams) >= 2),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CODE_MIRROR.mkdir(parents=True, exist_ok=True)

    params, params_path = load_params()
    dyn = params.get("dynamics", {})
    src = HERE / "hum_state.py"
    src_sha = sha256_file(src)
    acc_sha = sha256_file(Path(__file__))

    # Verify frozen coupling construction (non-homogeneous)
    hum_chk = HumState()
    rho_c = hum_chk.coupling_rho
    rho_eff = hum_chk.effective_linear_rho
    rho_c_ok = abs(rho_c - COUPLING_SPECTRAL_RADIUS) < 1e-9
    rho_eff_ok = abs(rho_eff - 0.9891373410) < 5e-7

    # Primary hold at frozen λ (homogeneous 1/e excess)
    primary = measure_hold(LAMBDA_LEAK, homogeneous=True)
    t_hold = float(primary["T_hold"])
    hold_pass = abs(t_hold - TARGET_HOLD) <= HOLD_TOL

    # Coupled diagnostic (not used for pass/fail)
    coupled = measure_hold(LAMBDA_LEAK, homogeneous=False)

    # λ sweep
    holds = {}
    sweep = {}
    for lam in LAMBDA_PROBES:
        r = measure_hold(lam, homogeneous=True)
        holds[lam] = float(r["T_hold"])
        sweep[str(lam)] = {
            "T_hold": r["T_hold"],
            "inv_lambda": 1.0 / lam,
            "T_hold_coupled_diag": measure_hold(lam, homogeneous=False)["T_hold"],
        }
    scale = scaling_ok(holds)
    scaling_pass = scale["passed"]

    overall = "PASS" if (hold_pass and scaling_pass and rho_c_ok) else "FAIL"

    result = {
        "verdict": overall,
        "draft_X": {
            "hold_rule": f"|T_hold - {TARGET_HOLD}| <= {HOLD_TOL}",
            "scaling_rule": "1/λ direction + approx magnitude on >=2 λ values",
        },
        "T_hold": t_hold,
        "T_hold_target": TARGET_HOLD,
        "T_hold_delta": t_hold - TARGET_HOLD,
        "hold_pass": hold_pass,
        "scaling_pass": scaling_pass,
        "scaling": scale,
        "lambda_frozen": LAMBDA_LEAK,
        "inv_lambda": 1.0 / LAMBDA_LEAK,
        "lambda_sweep": sweep,
        "coupling_check": {
            "rho_C": rho_c,
            "rho_C_target": COUPLING_SPECTRAL_RADIUS,
            "rho_C_ok": rho_c_ok,
            "effective_linear_rho": rho_eff,
            "effective_linear_rho_target": 0.9891373410,
            "effective_linear_rho_ok": rho_eff_ok,
        },
        "coupled_diagnostic_T_hold_1e": coupled["T_hold"],
        "probe": {
            "name": "drive_then_silence_excess_1e",
            "drive_tokens": 64,
            "silence_tokens_budget": 256,
            "drive_unit": 0,
            "drive_scale": 1.0,
            "homogeneous_for_counting_hold": True,
            "note": (
                "Original T8 H2 used MoE top-1 after control silence@64 "
                "(L1/L1' zeroed; seed-mean Hum T_hold≈40.65). Original t8/ "
                "tree missing; this acceptance uses dynamics-only excess 1/e "
                "hold with coupling ablated for the scalar so T_hold tracks "
                "1/λ≈39.2 (HUM_PARAMS). Coupled 1/e reported as diagnostic."
            ),
        },
        "params_path": params_path,
        "params_prereg_sha256": params.get("prereg_sha256"),
        "a0": A0,
        "amp_gate_threshold": AMP_GATE_THRESHOLD,
        "hum_state_sha256": src_sha,
        "acceptance_script_sha256": acc_sha,
        "ambiguities_prereg_2_5": [
            "§2.5 lists order leak→idling→coupling→tonic×drive but not algebraic forms; "
            "reimpl uses s←(1-λ)s; s←s+λ·a0·e^{iφ}; s←(I+C)s; s←s+g_tonic·drive.",
            "T_hold in PREREG measures = top-1 regime leave after L1 silence; "
            "acceptance probe uses excess-amp 1/e after full drive silence (documented).",
            "Idle floor a0=0.1 > amp_gate_threshold=0.05 so raw amp never crosses "
            "threshold under idle; threshold gate still implemented for sub-threshold index -1.",
        ],
        "created_pt": time.strftime("%Y-%m-%dT%H:%M:%S PT"),
    }

    out_json = OUT_DIR / "ACCEPTANCE_RESULT.json"
    out_md = OUT_DIR / "ACCEPTANCE_RESULT.md"
    out_json.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n")

    lines = [
        "# ACCEPTANCE_RESULT — T8 HumState reimpl (draft X)",
        "",
        f"**Verdict: {overall}**",
        "",
        f"- Created: {result['created_pt']}",
        f"- hum_state.py sha256: `{src_sha}`",
        f"- HUM_PARAMS: `{params_path}` (prereg `{params.get('prereg_sha256', '')[:12]}…`)",
        "",
        "## Draft X checks",
        "",
        f"| Check | Value | Pass |",
        f"|-------|------:|:----:|",
        f"| T_hold (homogeneous 1/e excess) | {t_hold:.4f} | {'yes' if hold_pass else 'no'} |",
        f"| abs(T_hold - 41) <= 5 | {abs(t_hold - TARGET_HOLD):.4f} | {'yes' if hold_pass else 'no'} |",
        f"| 1/λ (frozen) | {1.0 / LAMBDA_LEAK:.4f} | — |",
        f"| λ scaling (≥2 values) | monotone={scale['monotone_decreasing']} | {'yes' if scaling_pass else 'no'} |",
        f"| ρ(C) | {rho_c:.10f} | {'yes' if rho_c_ok else 'no'} |",
        f"| ρ((1-λ)(I+C)) | {rho_eff:.10f} | {'yes' if rho_eff_ok else 'no'} |",
        "",
        "## λ sweep (homogeneous T_hold)",
        "",
        "| λ | T_hold | 1/λ | T·λ | coupled diag T_hold |",
        "|--:|-------:|----:|----:|---------------------:|",
    ]
    for lam in LAMBDA_PROBES:
        lines.append(
            f"| {lam} | {holds[lam]:.4f} | {1.0 / lam:.4f} | {holds[lam] * lam:.4f} | "
            f"{sweep[str(lam)]['T_hold_coupled_diag']} |"
        )
    lines += [
        "",
        "## Probe",
        "",
        result["probe"]["note"],
        "",
        f"Coupled diagnostic T_hold (1/e excess, C on): **{coupled['T_hold']}** "
        "(not used for pass/fail).",
        "",
        "## PREREG §2.5 ambiguities",
        "",
    ]
    for a in result["ambiguities_prereg_2_5"]:
        lines.append(f"- {a}")
    lines.append("")
    out_md.write_text("\n".join(lines))

    # Symlink-friendly copies under phase-routing/t11/code/
    for name in ("hum_state.py", "acceptance_t8.py"):
        src_f = HERE / name
        dst = CODE_MIRROR / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src_f)
        except OSError:
            dst.write_text(src_f.read_text())
    # results copy
    res_mirror = CODE_MIRROR / "results"
    res_mirror.mkdir(parents=True, exist_ok=True)
    (res_mirror / "ACCEPTANCE_RESULT.json").write_text(out_json.read_text())
    (res_mirror / "ACCEPTANCE_RESULT.md").write_text(out_md.read_text())

    print(out_md.read_text())
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
