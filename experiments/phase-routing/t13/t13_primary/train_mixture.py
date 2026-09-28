#!/usr/bin/env python3
"""T13 train mixture under frozen PREREG + G9 training-instability gate.

Does NOT amend PREREG. Does NOT touch Affinity / C3 / hum.
Refuse counted primary if G9 FAIL (receipt written either way).
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
    BATCH,
    CKPT_DIR,
    EXPECTED_PREREG_SHA,
    GRAD_EXPLODE,
    K,
    L_MAX,
    LOCKED_SEEDS,
    LOSS_EXPLODE,
    LR,
    RESULTS,
    S_E,
    S_FULL,
    TRAIN_STEPS,
    VOCAB,
    W,
    assert_frozen_pins,
    c_match_moe_slices,
    hashed_phases,
    phases_sha,
)
from mythos_model import (  # noqa: E402
    forward,
    init_params,
    make_batch,
    save_params,
    sgd_step,
)

PT = ZoneInfo("America/Chicago")


def main() -> int:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pins = assert_frozen_pins()
    if not pins["ok"]:
        print("REFUSE train: frozen pin mismatch", pins)
        return 2

    seed = LOCKED_SEEDS[0]
    phases = hashed_phases(EXPECTED_PREREG_SHA)
    psha = phases_sha(phases)
    p = init_params(seed, phases=phases)
    rng = np.random.Generator(np.random.PCG64(seed))

    g9_fail_reasons = []
    history = []
    max_grad = 0.0
    last_nll = float("nan")
    last_acc = float("nan")

    print(
        f"T13 train mixture mythos_tiny seed={seed} steps={TRAIN_STEPS} "
        f"K={K} W={W} L_max={L_MAX} S_full={S_FULL} S_e={S_E} vocab={VOCAB}"
    )
    print(f"phases_sha={psha[:16]}… C_match_ref={c_match_moe_slices()}")

    for step in range(1, TRAIN_STEPS + 1):
        x, y = make_batch(rng, BATCH)
        nll, acc, gnorm, finite = sgd_step(p, x, y, LR, shared_ctx=False)
        last_nll, last_acc = nll, acc
        max_grad = max(max_grad, gnorm if np.isfinite(gnorm) else GRAD_EXPLODE)
        if not finite or not np.isfinite(nll) or not np.isfinite(gnorm):
            g9_fail_reasons.append(f"step={step} non-finite nll={nll} gnorm={gnorm} finite={finite}")
            break
        if gnorm > GRAD_EXPLODE:
            g9_fail_reasons.append(f"step={step} explosive grad_norm={gnorm} > {GRAD_EXPLODE}")
            break
        if nll > LOSS_EXPLODE:
            g9_fail_reasons.append(f"step={step} explosive loss={nll} > {LOSS_EXPLODE}")
            break
        if step == 1 or step % 200 == 0 or step == TRAIN_STEPS:
            history.append({"step": step, "nll": float(nll), "acc": float(acc), "grad_norm": float(gnorm)})
            print(f"  step {step:4d}  nll={nll:.4f}  acc={acc:.3f}  |g|={gnorm:.3f}")

    g9_pass = len(g9_fail_reasons) == 0
    # final eval batch
    x_ev, y_ev = make_batch(np.random.Generator(np.random.PCG64(seed + 999)), 128)
    ev = forward(p, x_ev, y_ev, shared_ctx=False)

    ckpt_path = CKPT_DIR / f"mixture_seed{seed}.npz"
    save_params(ckpt_path, p)

    receipt = {
        "ticket": "t13",
        "arm": "moe_of_loops_slices",
        "scale": "mythos_tiny",
        "seed": seed,
        "steps": TRAIN_STEPS,
        "steps_completed": history[-1]["step"] if history else 0,
        "final_nll": float(last_nll),
        "final_acc": float(last_acc),
        "eval_nll": float(ev["nll"]),
        "eval_acc": float(ev["acc"]),
        "max_grad_norm": float(max_grad),
        "phases": phases.tolist(),
        "phases_sha256": psha,
        "prereg_sha256": EXPECTED_PREREG_SHA,
        "affinity_sha256": pins["affinity_sha"],
        "C_match": c_match_moe_slices(),
        "G9": {
            "status": "PASS" if g9_pass else "FAIL",
            "checks": ["NaN", "non-finite", "explosive_grads"],
            "grad_explode_threshold": GRAD_EXPLODE,
            "loss_explode_threshold": LOSS_EXPLODE,
            "fail_reasons": g9_fail_reasons,
            "refuse_primary_if_fail": True,
        },
        "checkpoint": str(ckpt_path),
        "history": history,
        "frozen_at_pt": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT"),
        "pins": pins,
    }
    out_json = RESULTS / "T13_TRAIN_MIXTURE.json"
    out_md = RESULTS / "T13_TRAIN_MIXTURE.md"
    out_json.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(
        "\n".join(
            [
                "# T13 Train Mixture + G9",
                "",
                f"- **When:** {receipt['frozen_at_pt']}",
                f"- **Seed:** {seed}",
                f"- **Steps:** {receipt['steps_completed']}/{TRAIN_STEPS}",
                f"- **Final NLL/ACC:** {last_nll:.4f} / {last_acc:.3f}",
                f"- **Eval NLL/ACC:** {ev['nll']:.4f} / {ev['acc']:.3f}",
                f"- **Max |g|:** {max_grad:.4f}",
                f"- **G9:** **{receipt['G9']['status']}**",
                f"- **phases_sha:** `{psha}`",
                f"- **ckpt:** `{ckpt_path.name}`",
                f"- **PREREG:** `{EXPECTED_PREREG_SHA}`",
                f"- **Affinity:** `{pins['affinity_sha']}` UNTOUCHED",
                "",
                ("Refuse counted primary." if not g9_pass else "G9 PASS — proceed to G13 Acc + G17 Acc."),
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"G9={receipt['G9']['status']} wrote {out_json}")
    return 0 if g9_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
