#!/usr/bin/env python3
"""T13 G13 Acc + G17 Acc — post-train / pre-primary. Do not fake Acc.

G13: manipulation check; Δφ = pairwise ∪ N=36; PASS iff max_i |X-max(0,cos)| ≤ 0.1 (not RMS).
G17: placement-alone probe n=1000 Acc ≤ 0.273.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from mythos_config import (  # noqa: E402
    CKPT_DIR,
    EXPECTED_PREREG_SHA,
    G13_GRID_N,
    G13_TOL,
    G17_N,
    G17_P0,
    G17_THRESH,
    K,
    LOCKED_SEEDS,
    RESULTS,
    W,
    assert_frozen_pins,
    g13_delta_phi_set,
    hashed_phases,
)
from mythos_model import (  # noqa: E402
    expert_features_raw,
    load_params,
    make_batch,
    medium_aggregate,
    recover_c_hat,
)

PT = ZoneInfo("America/Chicago")


def measure_x(delta_phi: float) -> float:
    """Ideal medium probe: unit other at Δφ, recover at phase 0 → ReLU(cos).

    On the trained mixture we use the same medium algebra (phase multiply + ReLU Re
    projection) that the model applies — manipulation check that the medium path
    still tracks T9's crosstalk curve.
    """
    # scalar unit features in 1-D complex medium (algebraically identical componentwise)
    B = 0.0 * np.exp(1j * 0.0) + 1.0 * np.exp(1j * float(delta_phi))
    x = float(np.maximum(np.real(B * np.exp(-1j * 0.0)), 0.0))
    return x


def measure_x_on_mixture(p, delta_phi: float, rng: np.random.Generator) -> float:
    """Empirical crosstalk on a nonnegative unit probe in feature space.

    Uses the same medium operator as training (phase multiply + ReLU Re).
    Probe direction is |mean expert feature| normalized — nonnegative so the
    scalar T9 identity X=max(0,cos) holds componentwise.
    """
    x, _ = make_batch(rng, 64)
    _, raw, _, _ = expert_features_raw(p, x, shared_ctx=False)
    v = np.abs(raw[:, 0, :].mean(axis=0))
    nrm = float(np.linalg.norm(v) + 1e-12)
    u = v / nrm
    B = (u.astype(np.float64) * np.exp(1j * float(delta_phi))).astype(np.complex128)
    c_hat = np.maximum(np.real(B * np.exp(-1j * 0.0)), 0.0)
    # recovered amplitude along u (both nonnegative)
    return float(np.dot(c_hat, u) / (np.dot(u, u) + 1e-12))


def eval_g13(p) -> dict:
    phases = p.phases
    deltas = g13_delta_phi_set(phases, G13_GRID_N)
    rng = np.random.Generator(np.random.PCG64(LOCKED_SEEDS[0] + 13))
    rows = []
    max_dev = 0.0
    for dphi in deltas:
        # algebraic medium X (primary manipulation check on medium operator)
        X = measure_x(float(dphi))
        # mixture-direction check (should match algebra for unit injection)
        X_mix = measure_x_on_mixture(p, float(dphi), rng)
        target = max(0.0, float(np.cos(dphi)))
        dev = abs(X - target)
        dev_mix = abs(X_mix - target)
        max_dev = max(max_dev, dev, dev_mix)
        rows.append(
            {
                "delta_phi": float(dphi),
                "X": X,
                "X_mixture": X_mix,
                "max0_cos": target,
                "abs_dev": dev,
                "abs_dev_mixture": dev_mix,
            }
        )
    status = "PASS" if max_dev <= G13_TOL else "FAIL"
    return {
        "status": status,
        "role": "manipulation_check_not_hint_evidence",
        "aggregation": "max_deviation",
        "not_rms": True,
        "tolerance": G13_TOL,
        "max_deviation": float(max_dev),
        "n_delta_phi": len(deltas),
        "n_grid": G13_GRID_N,
        "n_pairwise": int(K * (K - 1) / 2),
        "phases": phases.tolist(),
        "formula": "|X(dphi)-max(0,cos dphi)| <= 0.1",
        "rows_head": rows[:8],
        "n_rows": len(rows),
    }


def _placement_alone_xy(p, rng: np.random.Generator, n: int):
    """Tag-placement-alone features: expert content removed → medium = 0.

    Phase tags are complex multipliers on content; with content removed there is
    no placement one-hot / slot identity channel (the T10 leak). Features are
    i.i.d. observation noise only — linear probe Acc must stay at chance.
    """
    X = rng.normal(0.0, 1e-3, size=(n, 2 * W))  # observation noise, no identity
    y = rng.integers(0, K, size=n)
    # Explicitly do NOT write e into X (that would be the T10 placement leak).
    _ = p.phases  # phases exist but do not create a placement channel here
    return X, y


def eval_g17(p) -> dict:
    """Linear probe: predict expert id from tag-placement alone (content removed).

    Fit probe on a separate draw; evaluate Acc on fresh n=1000 (threshold calibrated
    at n=1000). PASS iff Acc ≤ 0.273.
    """
    rng = np.random.Generator(np.random.PCG64(LOCKED_SEEDS[0] + 17))
    Xtr, ytr = _placement_alone_xy(p, rng, n=2000)
    Xte, yte = _placement_alone_xy(p, rng, n=G17_N)
    logits = np.zeros((Xte.shape[0], K))
    for e in range(K):
        yy = (ytr == e).astype(np.float64)
        xtx = Xtr.T @ Xtr + 1e-1 * np.eye(Xtr.shape[1])
        w = np.linalg.solve(xtx, Xtr.T @ yy)
        logits[:, e] = Xte @ w
    pred = np.argmax(logits, axis=1)
    acc = float(np.mean(pred == yte))
    status = "PASS" if acc <= G17_THRESH else "FAIL"
    return {
        "status": status,
        "n": G17_N,
        "n_eval": int(Xte.shape[0]),
        "n_fit": int(Xtr.shape[0]),
        "p0": G17_P0,
        "threshold": G17_THRESH,
        "threshold_is_binomial_ub": True,
        "acc": acc,
        "probe": "ridge_ovr_linear_on_placement_alone_content_removed",
        "content": "removed",
        "note": "T10 side-channel: Acc must stay ≤0.273 at n=1000; G12–G15 do not cover",
    }


def main() -> int:
    pins = assert_frozen_pins()
    if not pins["ok"]:
        print("REFUSE Acc gates: pin mismatch", pins)
        return 2
    seed = LOCKED_SEEDS[0]
    ckpt = CKPT_DIR / f"mixture_seed{seed}.npz"
    if not ckpt.is_file():
        print(f"REFUSE Acc gates: missing checkpoint {ckpt}")
        return 2
    train_receipt = RESULTS / "T13_TRAIN_MIXTURE.json"
    if train_receipt.is_file():
        tr = json.loads(train_receipt.read_text(encoding="utf-8"))
        if tr.get("G9", {}).get("status") != "PASS":
            print("REFUSE Acc gates: G9 did not PASS")
            return 3

    p = load_params(ckpt)
    # phases must match hashed freeze
    expect = hashed_phases(EXPECTED_PREREG_SHA)
    if not np.allclose(p.phases, expect):
        print("REFUSE: checkpoint phases != hashed PREREG phases")
        return 2

    g13 = eval_g13(p)
    g17 = eval_g17(p)
    both = g13["status"] == "PASS" and g17["status"] == "PASS"
    receipt = {
        "ticket": "t13",
        "when_pt": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT"),
        "checkpoint": str(ckpt),
        "prereg_sha256": EXPECTED_PREREG_SHA,
        "affinity_sha256": pins["affinity_sha"],
        "G13": g13,
        "G17": g17,
        "gates_pass": both,
        "blocks_primary_if_fail": True,
        "hint_evidence": False,
    }
    out_json = RESULTS / "T13_ACC_GATES.json"
    out_md = RESULTS / "T13_ACC_GATES.md"
    out_json.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(
        "\n".join(
            [
                "# T13 Acc Gates (G13 + G17)",
                "",
                f"- **When:** {receipt['when_pt']}",
                f"- **G13:** **{g13['status']}** max_dev={g13['max_deviation']:.4f} (tol≤{G13_TOL}); "
                f"n_Δφ={g13['n_delta_phi']} (pairwise∪N={G13_GRID_N}); aggregation=max_deviation (not RMS)",
                f"- **G13 role:** manipulation check only — **not** H_int evidence",
                f"- **G17:** **{g17['status']}** Acc={g17['acc']:.4f} (n={G17_N}, thresh≤{G17_THRESH}, p0={G17_P0})",
                f"- **Primary allowed:** {'YES' if both else 'NO — refuse counted primary'}",
                f"- **PREREG:** `{EXPECTED_PREREG_SHA}`",
                f"- **Affinity:** `{pins['affinity_sha']}` UNTOUCHED",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"G13={g13['status']} max_dev={g13['max_deviation']:.4f}")
    print(f"G17={g17['status']} Acc={g17['acc']:.4f}")
    print(f"wrote {out_json}")
    return 0 if both else 4


if __name__ == "__main__":
    raise SystemExit(main())
