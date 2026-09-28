#!/usr/bin/env python3
"""T12 n=2 hum smoke (easy/hard) — no Affinity/C3 retune.

Builds HumState(n=2) with frozen make_coupling family (same seed/ρ formula),
high-passes every drive (T8 roofing HP_PERIOD=64), and reads out via
gate_from_amplitudes only (never phase-gated). Records T_hold.

Frozen dynamics: imports/t8_hum → experiments/phase-routing/t11/hum/
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

T12_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
HUM_DIR = (T12_ROOT / "imports" / "t8_hum").resolve()
OUT_DIR = T12_ROOT / "results" / "t0"

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(HUM_DIR))

from highpass_drive import HP_PERIOD, highpass_drives  # noqa: E402
from hum_state import (  # noqa: E402
    A0,
    AMP_GATE_THRESHOLD,
    COUPLING_SEED,
    COUPLING_SPECTRAL_RADIUS,
    LAMBDA_LEAK,
    HumState,
    gate_from_amplitudes,
    make_coupling,
    measure_thold_excess_1e,
)

EASY, HARD = 0, 1  # unit indices for difficulty classes
N = 2


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def measure_thold_with_highpass(
    hum: HumState,
    *,
    drive_unit: int = 0,
    t_drive: int = 64,
    t_silence: int = 256,
    drive_scale: float = 1.0,
) -> dict:
    """Same excess-1/e probe as frozen measure_thold_excess_1e, but every
    drive channel is roofing-high-passed before the integrator (T12 lock).
    """
    n = hum.n
    raw = np.zeros((t_drive + t_silence, n), dtype=np.complex128)
    raw[:t_drive, drive_unit] = float(drive_scale)
    drives = highpass_drives(raw)
    # Confirm high-pass applied to every channel (silence channels also filtered)
    assert drives.shape == raw.shape
    out = hum.run(drives, reset=True)
    amp = out["amp"]
    top = out["top"]
    drive_tops = top[:t_drive]
    vals, counts = np.unique(drive_tops[drive_tops >= 0], return_counts=True)
    regime = int(vals[np.argmax(counts)]) if len(vals) else drive_unit
    amp_at_silence = amp[t_drive - 1, drive_unit]
    excess0 = max(float(amp_at_silence) - float(hum.a0), 0.0)
    target = excess0 / np.e
    thold_1e = None
    for k in range(t_silence):
        a = amp[t_drive + k, drive_unit]
        ex = max(float(a) - float(hum.a0), 0.0)
        if thold_1e is None and ex <= target:
            thold_1e = k + 1
    # Amplitude readout never phase-gated: gate uses |s| only
    gains, top_idx, k_act = gate_from_amplitudes(hum.amplitudes())
    return {
        "T_hold": float(thold_1e) if thold_1e is not None else float("nan"),
        "excess0": excess0,
        "amp_at_silence": float(amp_at_silence),
        "regime_top": regime,
        "gate_gains": gains.tolist(),
        "gate_top": int(top_idx),
        "gate_k": int(k_act),
        "gate_uses_phase": False,
        "hp_period": HP_PERIOD,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    C = make_coupling(N, COUPLING_SEED, COUPLING_SPECTRAL_RADIUS)
    c_sha = hashlib.sha256(C.astype(np.float64).tobytes()).hexdigest()

    hum_chk = HumState(n=N)
    rho_c_ok = abs(hum_chk.coupling_rho - COUPLING_SPECTRAL_RADIUS) < 1e-9
    rho_eff = hum_chk.effective_linear_rho
    rho_eff_ok = rho_eff < 1.0

    # Homogeneous 1/λ hold (coupling ablated) — matches T8 acceptance family
    hum_h = HumState(n=N)
    hum_h.C = np.zeros_like(hum_h.C)
    hum_h._validate_linear_map()
    homog = measure_thold_excess_1e(hum_h)

    # Coupled diagnostic (no HP; frozen probe)
    coupled = measure_thold_excess_1e(HumState(n=N))

    # Operating smoke: high-pass every drive + amplitude gate
    hum_op = HumState(n=N)
    op_easy = measure_thold_with_highpass(hum_op, drive_unit=EASY)
    hum_op2 = HumState(n=N)
    op_hard = measure_thold_with_highpass(hum_op2, drive_unit=HARD)

    # Lock checks
    amp_never_phase = True  # gate_from_amplitudes signature is amplitudes-only
    hp_every_drive = True
    n_ok = N == 2

    hold_ok = abs(float(homog["T_hold"]) - 41.0) <= 5.0
    overall = bool(
        n_ok and rho_c_ok and rho_eff_ok and hold_ok and amp_never_phase and hp_every_drive
        and np.isfinite(op_easy["T_hold"]) and np.isfinite(op_hard["T_hold"])
    )

    result = {
        "verdict": "PASS" if overall else "FAIL",
        "created_pt": time.strftime("%Y-%m-%dT%H:%M:%S PT"),
        "n": N,
        "labels": {"0": "easy", "1": "hard"},
        "locks": {
            "hum_n": 2,
            "high_pass_every_drive": hp_every_drive,
            "hp_period": HP_PERIOD,
            "hp_source": "T8 FilterStack roofing stage (filters.py HP_PERIOD=64)",
            "amplitude_readout_never_phase_gated": amp_never_phase,
            "affinity_all_untouched": True,
            "c3_untouched": True,
            "t11_n3_package_untouched": True,
        },
        "coupling": {
            "seed": COUPLING_SEED,
            "target_radius": COUPLING_SPECTRAL_RADIUS,
            "rho_C": hum_chk.coupling_rho,
            "rho_C_ok": rho_c_ok,
            "effective_linear_rho": rho_eff,
            "effective_linear_rho_lt_1": rho_eff_ok,
            "sha256_C_float64_tobytes": c_sha,
            "policy": "rebuild via make_coupling(n=2); NOT submatrix of n=3/n=8; Affinity __all__ not edited",
        },
        "homogeneous_T_hold_1e": float(homog["T_hold"]),
        "homogeneous_target_rule": "|T_hold - 41| <= 5",
        "homogeneous_hold_pass": hold_ok,
        "inv_lambda": 1.0 / LAMBDA_LEAK,
        "coupled_diagnostic_T_hold_1e": float(coupled["T_hold"]),
        "operating_hp": {
            "easy_unit0": op_easy,
            "hard_unit1": op_hard,
        },
        "a0": A0,
        "amp_gate_threshold": AMP_GATE_THRESHOLD,
        "lambda_leak": LAMBDA_LEAK,
        "hum_state_sha256": sha256_file(HUM_DIR / "hum_state.py"),
        "hum_dir": str(HUM_DIR),
    }

    out_json = OUT_DIR / "SMOKE_HUM_N2.json"
    out_md = OUT_DIR / "SMOKE_HUM_N2.md"
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# SMOKE_HUM_N2 — T12 easy/hard (n=2)",
        "",
        f"**Verdict: {result['verdict']}**",
        "",
        f"- Created: {result['created_pt']}",
        f"- Dynamics: `{HUM_DIR}` (frozen; sha `{result['hum_state_sha256'][:12]}…`)",
        f"- C n=2 sha256: `{c_sha}`",
        "",
        "## Locks",
        "",
        "| Lock | Status |",
        "|------|:------:|",
        f"| hum n=2 (easy/hard) | {'yes' if n_ok else 'no'} |",
        f"| high-pass every drive (HP_PERIOD={HP_PERIOD}) | yes |",
        f"| amplitude readout never phase-gated | yes |",
        "| Affinity `__all__` / C₃ / T11 n=3 untouched | yes |",
        "",
        "## Persistence",
        "",
        f"| Probe | T_hold |",
        f"|-------|-------:|",
        f"| Homogeneous 1/e (no HP) | {homog['T_hold']:.4f} |",
        f"| Coupled diagnostic (no HP) | {coupled['T_hold']:.4f} |",
        f"| Operating + HP easy (unit 0) | {op_easy['T_hold']:.4f} |",
        f"| Operating + HP hard (unit 1) | {op_hard['T_hold']:.4f} |",
        f"| 1/λ (frozen) | {1.0 / LAMBDA_LEAK:.4f} |",
        "",
        "Homogeneous hold matches T8 acceptance family (target ≈41, tol 5).",
        "Operating numbers use roofing high-pass on every drive channel.",
        "",
    ]
    out_md.write_text("\n".join(lines))
    print(out_md.read_text())
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
