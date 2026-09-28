#!/usr/bin/env python3
"""T13 counted primary — H_int × locked seeds. Refuse unless G9+G13+G17 PASS.

Does not amend PREREG. Affinity / C3 / hum untouched.
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
    K,
    LOCKED_SEEDS,
    LR,
    RESULTS,
    SEED_INTERVAL,
    TRAIN_STEPS,
    W,
    assert_frozen_pins,
    hashed_phases,
)
from mythos_model import (  # noqa: E402
    clone_params,
    forward,
    init_params,
    make_batch,
    medium_aggregate,
    nll_acc,
    readout_from_medium,
    recover_c_hat,
    save_params,
    sgd_step,
)

PT = ZoneInfo("America/Chicago")
PRIMARY_DIR = RESULTS / "primary"


def gates_clear() -> tuple[bool, str]:
    train = RESULTS / "T13_TRAIN_MIXTURE.json"
    acc = RESULTS / "T13_ACC_GATES.json"
    if not train.is_file() or not acc.is_file():
        return False, "missing train or Acc gate receipts"
    tr = json.loads(train.read_text(encoding="utf-8"))
    ac = json.loads(acc.read_text(encoding="utf-8"))
    if tr.get("G9", {}).get("status") != "PASS":
        return False, "G9 FAIL"
    if ac.get("G13", {}).get("status") != "PASS":
        return False, "G13 Acc FAIL"
    if ac.get("G17", {}).get("status") != "PASS":
        return False, "G17 Acc FAIL"
    return True, "G9+G13+G17 PASS"


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.size < 3:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    da = float(np.sqrt(np.sum(a * a)))
    db = float(np.sqrt(np.sum(b * b)))
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return float(np.sum(a * b) / (da * db))


def train_arm(seed: int, shared_ctx: bool) -> object:
    phases = hashed_phases(EXPECTED_PREREG_SHA)
    p = init_params(seed, phases=phases)
    rng = np.random.Generator(np.random.PCG64(seed + (99 if shared_ctx else 0)))
    for step in range(1, TRAIN_STEPS + 1):
        x, y = make_batch(rng, BATCH)
        nll, acc, gnorm, finite = sgd_step(p, x, y, LR, shared_ctx=shared_ctx)
        if not finite:
            raise RuntimeError(f"non-finite during primary train seed={seed} shared={shared_ctx} step={step}")
    return p


def hint_scores(p, x: np.ndarray, target: np.ndarray, shared_ctx: bool = False) -> dict:
    """Pearson r over (e,x) for phase / plain / random-tag / oracle / no-medium."""
    Bsz = x.shape[0]
    base = forward(p, x, target, shared_ctx=shared_ctx)
    feats = base["feats"]
    phases = p.phases
    Bmed = base["B"]
    base_nll = base["nll_vec"]

    # actual ΔL_e: silence expert e
    dL_actual = np.zeros((Bsz, K), dtype=np.float64)
    for e in range(K):
        silenced = forward(p, x, target, shared_ctx=shared_ctx, silence=e)
        dL_actual[:, e] = silenced["nll_vec"] - base_nll

    def delta_from_chat(c_hat_e: np.ndarray, e: int) -> np.ndarray:
        # remove ĉ_e * e^{iφ} from medium; re-readout
        B2 = Bmed - c_hat_e.astype(np.float64) * np.exp(1j * float(phases[e]))
        logits = readout_from_medium(p, B2)
        nll_vec, _, _, _ = nll_acc(logits, target)
        return nll_vec - base_nll

    # (1) phase recovery
    dL_phase = np.zeros((Bsz, K), dtype=np.float64)
    for e in range(K):
        c_hat = recover_c_hat(Bmed, phases, e)
        dL_phase[:, e] = delta_from_chat(c_hat, e)

    # (2) plain sum: ĉ = aggregate scaled by gate weight (use ||f_e|| / sum)
    norms = np.linalg.norm(feats, axis=2) + 1e-12  # [B,K]
    wts = norms / norms.sum(axis=1, keepdims=True)
    B_real_mag = np.abs(Bmed)  # rough
    dL_plain = np.zeros((Bsz, K), dtype=np.float64)
    for e in range(K):
        # plain: scale |B| direction by weight — use ReLU(Re(B))*w_e as ĉ proxy without phase
        c_hat = np.maximum(np.real(Bmed), 0.0) * wts[:, e : e + 1]
        dL_plain[:, e] = delta_from_chat(c_hat, e)

    # (3) random-tag: shuffle phases
    rng = np.random.Generator(np.random.PCG64(12345))
    rand_phases = rng.uniform(0, 2 * np.pi, size=K)
    B_rand = medium_aggregate(feats, rand_phases)
    dL_rand = np.zeros((Bsz, K), dtype=np.float64)
    for e in range(K):
        c_hat = recover_c_hat(B_rand, rand_phases, e)
        B2 = B_rand - c_hat.astype(np.float64) * np.exp(1j * float(rand_phases[e]))
        logits = readout_from_medium(p, B2)
        nll_vec, _, _, _ = nll_acc(logits, target)
        dL_rand[:, e] = nll_vec - base_nll

    # (4) oracle: true f_e * e^{iφ}
    dL_oracle = np.zeros((Bsz, K), dtype=np.float64)
    for e in range(K):
        true_c = feats[:, e, :]  # real magnitude in feature space before phase
        # oracle ĉ = ReLU(Re(f_e e^{iφ} e^{-iφ})) = ReLU(f_e) approx f when positive
        c_hat = np.maximum(true_c, 0.0)
        dL_oracle[:, e] = delta_from_chat(c_hat, e)

    # (5) no-medium: activation-norm attribution
    dL_nomed = norms  # higher norm → predicted larger effect; scale to NLL units via z-score later
    # correlate raw norms against actual ΔL (same pearson)

    r_phase = pearson(dL_phase, dL_actual)
    r_plain = pearson(dL_plain, dL_actual)
    r_rand = pearson(dL_rand, dL_actual)
    r_oracle = pearson(dL_oracle, dL_actual)
    r_nomed = pearson(dL_nomed, dL_actual)

    return {
        "r_phase": r_phase,
        "r_plain": r_plain,
        "r_random": r_rand,
        "r_oracle": r_oracle,
        "r_no_medium": r_nomed,
        "n_pairs": int(Bsz * K),
    }


def verdict(r_phase: float, r_plain: float, r_random: float, r_nomed: float, interval: float) -> str:
    """§4 / §6 vocabulary."""
    c1 = (r_phase - r_plain) > interval
    c2 = r_phase >= (r_random - 0.05)
    c3 = (r_phase - r_nomed) > interval
    if c1 and c2 and c3:
        if (r_phase - r_random) > interval:
            return "support"  # includes phase_over_tags
        return "support"  # clears floor but not necessarily phase_over_tags subset
    if c1 and c2 and not c3:
        return "medium_not_beyond_attribution"
    if abs(r_phase - r_random) <= interval and c1:
        return "tagged_superposition"
    return "null"


def main() -> int:
    pins = assert_frozen_pins()
    if not pins["ok"]:
        print("REFUSE primary: pin mismatch", pins)
        return 2
    ok, reason = gates_clear()
    if not ok:
        print(f"REFUSE counted primary: {reason}")
        receipt = {
            "status": "REFUSED",
            "reason": reason,
            "when_pt": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT"),
        }
        PRIMARY_DIR.mkdir(parents=True, exist_ok=True)
        (PRIMARY_DIR / "T13_PRIMARY_REFUSED.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        return 5

    PRIMARY_DIR.mkdir(parents=True, exist_ok=True)
    per_seed = []
    print(f"T13 primary H_int × seeds {LOCKED_SEEDS}")

    for seed in LOCKED_SEEDS:
        print(f"=== seed {seed}: train slice arm ===")
        p_slice = train_arm(seed, shared_ctx=False)
        save_params(CKPT_DIR / f"primary_slice_seed{seed}.npz", p_slice)
        print(f"=== seed {seed}: train shared_ctx ablation ===")
        p_shared = train_arm(seed, shared_ctx=True)
        save_params(CKPT_DIR / f"primary_shared_seed{seed}.npz", p_shared)

        rng = np.random.Generator(np.random.PCG64(seed + 777))
        x, y = make_batch(rng, 128)
        h_slice = hint_scores(p_slice, x, y, shared_ctx=False)
        h_shared = hint_scores(p_shared, x, y, shared_ctx=True)
        v_slice = verdict(
            h_slice["r_phase"], h_slice["r_plain"], h_slice["r_random"], h_slice["r_no_medium"], SEED_INTERVAL
        )
        v_shared = verdict(
            h_shared["r_phase"], h_shared["r_plain"], h_shared["r_random"], h_shared["r_no_medium"], SEED_INTERVAL
        )
        row = {
            "seed": seed,
            "slice": h_slice,
            "shared_ctx": h_shared,
            "verdict_slice": v_slice,
            "verdict_shared_ctx": v_shared,
        }
        per_seed.append(row)
        print(
            f"  slice r_phase={h_slice['r_phase']:.3f} r_plain={h_slice['r_plain']:.3f} "
            f"r_rand={h_slice['r_random']:.3f} r_nomed={h_slice['r_no_medium']:.3f} → {v_slice}"
        )
        print(
            f"  shared r_phase={h_shared['r_phase']:.3f} → {v_shared}"
        )

    # seed-mean + interval
    def mean_r(key_arm: str, key_r: str) -> float:
        return float(np.mean([row[key_arm][key_r] for row in per_seed]))

    agg = {
        "r_phase_slice_mean": mean_r("slice", "r_phase"),
        "r_plain_slice_mean": mean_r("slice", "r_plain"),
        "r_random_slice_mean": mean_r("slice", "r_random"),
        "r_no_medium_slice_mean": mean_r("slice", "r_no_medium"),
        "r_oracle_slice_mean": mean_r("slice", "r_oracle"),
        "r_phase_shared_mean": mean_r("shared_ctx", "r_phase"),
        "seed_interval": SEED_INTERVAL,
    }
    overall = verdict(
        agg["r_phase_slice_mean"],
        agg["r_plain_slice_mean"],
        agg["r_random_slice_mean"],
        agg["r_no_medium_slice_mean"],
        SEED_INTERVAL,
    )
    # slice contrast: slice beats shared on three-clause
    slice_beats = (agg["r_phase_slice_mean"] - agg["r_phase_shared_mean"]) > SEED_INTERVAL
    if overall == "support" and not slice_beats:
        # PREREG: when slice contrast in primary, slice arm beats shared-ctx
        label = "mixed"
    else:
        label = overall

    receipt = {
        "ticket": "t13",
        "status": "COMPLETE",
        "when_pt": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT"),
        "prereg_sha256": EXPECTED_PREREG_SHA,
        "affinity_sha256": pins["affinity_sha"],
        "seeds": LOCKED_SEEDS,
        "primary_metric": "H_int",
        "aggregate": agg,
        "verdict": label,
        "slice_beats_shared_ctx": slice_beats,
        "per_seed": per_seed,
        "gates": {"G9": "PASS", "G13": "PASS", "G17": "PASS"},
    }
    out_json = PRIMARY_DIR / "T13_PRIMARY.json"
    out_md = PRIMARY_DIR / "T13_PRIMARY.md"
    out_json.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# T13 Primary — H_int",
        "",
        f"- **When:** {receipt['when_pt']}",
        f"- **Verdict:** **{label}**",
        f"- **Seeds:** {LOCKED_SEEDS}",
        f"- **r_phase (slice mean):** {agg['r_phase_slice_mean']:.4f}",
        f"- **r_plain:** {agg['r_plain_slice_mean']:.4f}",
        f"- **r_random:** {agg['r_random_slice_mean']:.4f}",
        f"- **r_no_medium:** {agg['r_no_medium_slice_mean']:.4f}",
        f"- **r_oracle:** {agg['r_oracle_slice_mean']:.4f}",
        f"- **r_phase shared_ctx mean:** {agg['r_phase_shared_mean']:.4f}",
        f"- **slice_beats_shared:** {slice_beats}",
        f"- **interval:** {SEED_INTERVAL}",
        f"- **PREREG:** `{EXPECTED_PREREG_SHA}`",
        f"- **Affinity:** `{pins['affinity_sha']}` UNTOUCHED",
        "",
        "## Per seed",
        "",
    ]
    for row in per_seed:
        lines.append(
            f"- seed {row['seed']}: slice={row['verdict_slice']} "
            f"(rφ={row['slice']['r_phase']:.3f}) shared={row['verdict_shared_ctx']} "
            f"(rφ={row['shared_ctx']['r_phase']:.3f})"
        )
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    # also top-level pointer
    (RESULTS / "T13_PRIMARY.json").write_text(out_json.read_text(encoding="utf-8"), encoding="utf-8")
    (RESULTS / "T13_PRIMARY.md").write_text(out_md.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"PRIMARY verdict={label} wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
