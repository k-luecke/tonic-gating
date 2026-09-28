#!/usr/bin/env python3
"""T12b primary — routing-only gaps/lags under frozen PREREG.

Primary contrast: hum_measure vs one_step_memory (CSR_gap).
majority_k3 = labeled secondary only.
G10 mid-gap lure active; G9 counted HumState arm.
Affinity / C₃ / hum untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np

T12B = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "t11" / "imports"))
sys.path.insert(0, str(ROOT / "experiments" / "phase-routing" / "t11"))

from arms import (  # noqa: E402
    ARMS,
    HumMeasureArm,
    HumMeasureArmStub,
    PRIMARY_ARMS,
    SECONDARY_ARMS,
    arm_factory,
    refuse_stub_as_counted,
    smoke_decide,
)
from gap_inject import (  # noqa: E402
    GAP_LENGTHS,
    LAG_LENGTHS,
    GapSpec,
    apply_lag,
    choose_gap_start,
    inject_gap_with_g10,
)
from hum_measure.regime_probe import SPEC_ORDER, feature_counts, tokenize  # noqa: E402

PT = ZoneInfo("America/Chicago")
SEALED_ALL = T12B / "generators" / "sealed" / "ALL_SEQUENCES.jsonl"
CROSSOVER = T12B / "docs" / "T12B_CROSSOVER_PREDICTION.md"
PREREG_SHA_PATH = T12B / "PREREG.sha256"
EXPECTED_T113_SHA = "27ab4a9573ec8c01ed043d72ef6c83ba47fdd4a764a373acb7453381bb9bd442"
EXPECTED_ALL_SHA = "646df30bbeacdc77432e9cb1daa366640ca0cfeb2e01216d4cb5f03a9df404c6"
EXPECTED_AFFINITY = "7dae877a3975c1ba4ca264f03f572d6749cce39a114ea3adbcffc010c7852c50"
SEEDS = [20260923, 20260924, 20260925, 20260926, 20260927]
SEED_INTERVAL = 0.05
OUT_DIR = T12B / "results"
SPEC_TO_IDX = {s: i for i, s in enumerate(SPEC_ORDER)}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_sequences() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(SEALED_ALL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prefix_score_matrix(text: str) -> Tuple[np.ndarray, List[str], str]:
    """Build (T,3) non-neg score matrix from expanding token prefixes."""
    toks = tokenize(text)
    T = len(toks)
    mat = np.zeros((T, 3), dtype=np.float64)
    for step in range(T):
        counts = feature_counts(toks[: step + 1])
        for i, s in enumerate(SPEC_ORDER):
            mat[step, i] = float(counts.get(s, 0))
    return mat, toks, text


def csr_on_mask(
    preds: List[Optional[str]], oracle: str, mask: np.ndarray
) -> Dict[str, Any]:
    idxs = [i for i, m in enumerate(mask) if m]
    if not idxs:
        return {"csr": None, "n": 0, "correct": 0}
    correct = sum(1 for i in idxs if preds[i] == oracle)
    n = len(idxs)
    return {"csr": correct / n, "n": n, "correct": correct}


def run_episode_gap(
    *,
    score_mat: np.ndarray,
    oracle: str,
    gap_len: int,
    seed: int,
    seq_id: int,
    seg_index: int,
    evidence_point: int,
) -> Optional[Dict[str, Any]]:
    T = score_mat.shape[0]
    start = choose_gap_start(
        T, gap_len, seed=seed, seq_id=seq_id, seg_index=seg_index, evidence_point=evidence_point
    )
    if start is None:
        return None
    true_idx = SPEC_TO_IDX.get(oracle)
    gap = GapSpec(start=int(start), length=int(gap_len))
    mod, mask, pert = inject_gap_with_g10(
        score_mat,
        gap,
        seed=seed,
        seq_id=seq_id,
        true_spec_index=true_idx,
        peak=1.0,
    )
    arms = arm_factory(mode="primary")
    refuse_stub_as_counted(arms["hum_measure"])
    for a in arms.values():
        a.reset()
    preds: Dict[str, List[Optional[str]]] = {k: [] for k in arms}
    for t in range(T):
        row = mod[t].tolist()
        for name, arm in arms.items():
            preds[name].append(arm.decide(row))

    # pre-gap window: same length immediately before gap
    pre_start = max(0, gap.start - gap_len)
    pre_mask = np.zeros(T, dtype=bool)
    pre_mask[pre_start:gap.start] = True

    out: Dict[str, Any] = {
        "gap_start": gap.start,
        "gap_len": gap_len,
        "t_flip": gap.start + pert.offset,
        "wrong_spec_index": int(pert.wrong_spec_index),
        "oracle": oracle,
        "csr_gap": {},
        "csr_pre": {},
        "default_rate_gap": {},
    }
    for name in arms:
        out["csr_gap"][name] = csr_on_mask(preds[name], oracle, mask)
        out["csr_pre"][name] = csr_on_mask(preds[name], oracle, pre_mask)
        gap_preds = [preds[name][i] for i, m in enumerate(mask) if m]
        out["default_rate_gap"][name] = (
            sum(1 for p in gap_preds if p is None) / len(gap_preds) if gap_preds else None
        )
    return out


def run_episode_lag(
    *,
    score_mat: np.ndarray,
    oracle: str,
    lag: int,
    evidence_point: int,
) -> Dict[str, Any]:
    lagged = apply_lag(score_mat, lag, zero_pad=True)
    arms = arm_factory(mode="primary")
    refuse_stub_as_counted(arms["hum_measure"])
    for a in arms.values():
        a.reset()
    preds: Dict[str, List[Optional[str]]] = {k: [] for k in arms}
    T = lagged.shape[0]
    for t in range(T):
        row = lagged[t].tolist()
        for name, arm in arms.items():
            preds[name].append(arm.decide(row))
    mask = np.zeros(T, dtype=bool)
    mask[max(evidence_point, 0) :] = True
    out: Dict[str, Any] = {"lag": lag, "csr_lag": {}}
    for name in arms:
        out["csr_lag"][name] = csr_on_mask(preds[name], oracle, mask)
    return out


def run_seed(seed: int, sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
    seqs = [r for r in sequences if int(r["seed"]) == int(seed)]
    gap_cells: List[Dict[str, Any]] = []
    lag_cells: List[Dict[str, Any]] = []
    for seq_id, row in enumerate(seqs):
        for seg_index, seg in enumerate(row.get("segments") or []):
            text = seg.get("text") or ""
            oracle = seg.get("oracle_specialist")
            if not text or not oracle:
                continue
            ep = int(seg.get("evidence_point", 0) or 0)
            score_mat, toks, _ = prefix_score_matrix(text)
            if score_mat.shape[0] < 8:
                continue
            for G in GAP_LENGTHS:
                ep_cell = run_episode_gap(
                    score_mat=score_mat,
                    oracle=oracle,
                    gap_len=int(G),
                    seed=seed,
                    seq_id=seq_id,
                    seg_index=seg_index,
                    evidence_point=ep,
                )
                if ep_cell is None:
                    continue
                ep_cell["sequence_type"] = row.get("sequence_type")
                ep_cell["seg_index"] = seg_index
                ep_cell["seq_id"] = seq_id
                gap_cells.append(ep_cell)
            for L in LAG_LENGTHS:
                lc = run_episode_lag(
                    score_mat=score_mat, oracle=oracle, lag=int(L), evidence_point=ep
                )
                lc["sequence_type"] = row.get("sequence_type")
                lc["seg_index"] = seg_index
                lc["seq_id"] = seq_id
                lag_cells.append(lc)

    def mean_csr(cells: List[Dict[str, Any]], key: str, arm: str) -> Optional[float]:
        vals = []
        for c in cells:
            block = c.get(key) or {}
            m = block.get(arm) or {}
            if m.get("csr") is not None and m.get("n", 0) > 0:
                vals.append(float(m["csr"]))
        return float(np.mean(vals)) if vals else None

    arms_all = list(ARMS)
    csr_gap = {a: mean_csr(gap_cells, "csr_gap", a) for a in arms_all}
    csr_pre = {a: mean_csr(gap_cells, "csr_pre", a) for a in arms_all}
    csr_lag = {a: mean_csr(lag_cells, "csr_lag", a) for a in arms_all}
    return {
        "seed": seed,
        "n_gap_cells": len(gap_cells),
        "n_lag_cells": len(lag_cells),
        "CSR_gap": csr_gap,
        "CSR_pre": csr_pre,
        "CSR_lag": csr_lag,
        "gap_cells": gap_cells,
        "lag_cells": lag_cells,
    }


def aggregate(seed_reports: List[Dict[str, Any]], prereg_sha: str) -> Dict[str, Any]:
    arms_all = list(ARMS)

    def mean_over_seeds(metric: str, arm: str) -> Optional[float]:
        vals = []
        for s in seed_reports:
            v = (s.get(metric) or {}).get(arm)
            if v is not None:
                vals.append(float(v))
        return float(np.mean(vals)) if vals else None

    csr_gap = {a: mean_over_seeds("CSR_gap", a) for a in arms_all}
    csr_pre = {a: mean_over_seeds("CSR_pre", a) for a in arms_all}
    csr_lag = {a: mean_over_seeds("CSR_lag", a) for a in arms_all}

    def gap_delta(a: str, b: str) -> Optional[float]:
        if csr_gap.get(a) is None or csr_gap.get(b) is None:
            return None
        return float(csr_gap[a] - csr_gap[b])

    primary_delta = gap_delta("hum_measure", "one_step_memory")
    vs_stateless = gap_delta("hum_measure", "stateless_argmax")
    vs_majority = gap_delta("hum_measure", "majority_k3")

    # seed-level primary deltas for interval
    seed_deltas = []
    for s in seed_reports:
        h = (s.get("CSR_gap") or {}).get("hum_measure")
        o = (s.get("CSR_gap") or {}).get("one_step_memory")
        if h is not None and o is not None:
            seed_deltas.append(float(h - o))
    seed_interval_ok = (
        abs(float(np.std(seed_deltas))) if len(seed_deltas) > 1 else 0.0
    )

    # null: hum no better than one_step within SEED_INTERVAL
    null_holds = None
    if primary_delta is not None:
        null_holds = primary_delta <= SEED_INTERVAL
    support_hint = None
    if primary_delta is not None:
        if primary_delta > SEED_INTERVAL:
            support_hint = "support_lean"  # null fail ⇒ support
        elif primary_delta >= -SEED_INTERVAL:
            support_hint = "null_lean"
        else:
            support_hint = "null_lean_hum_worse"

    return {
        "ticket": "t12b",
        "pt": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT"),
        "prereg_sha256": prereg_sha,
        "seeds": [s["seed"] for s in seed_reports],
        "primary_contrast": "hum_measure vs one_step_memory",
        "labeled_secondary": "majority_k3",
        "seed_interval_csr": SEED_INTERVAL,
        "CSR_gap": csr_gap,
        "CSR_pre": csr_pre,
        "CSR_lag": csr_lag,
        "deltas": {
            "CSR_gap(hum)-CSR_gap(one_step_memory)": primary_delta,
            "CSR_gap(hum)-CSR_gap(stateless_argmax)": vs_stateless,
            "CSR_gap(hum)-CSR_gap(majority_k3)": vs_majority,
        },
        "seed_primary_deltas": seed_deltas,
        "seed_primary_delta_std": seed_interval_ok,
        "null_holds_hum_le_one_step_plus_interval": null_holds,
        "support_null_hint": support_hint,
        "affinity_sha": EXPECTED_AFFINITY,
        "affinity_untouched": True,
        "G10_active": True,
        "G9_counted_humstate": True,
        "n_gap_cells_total": sum(s["n_gap_cells"] for s in seed_reports),
        "n_lag_cells_total": sum(s["n_lag_cells"] for s in seed_reports),
    }


def dry_run() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "mode": "dry-run",
        "ticket": "t12b",
        "pt": datetime.now(PT).strftime("%Y-%m-%d %H:%M:%S PT"),
        "status": "FROZEN" if PREREG_SHA_PATH.is_file() else "DRAFT",
        "primary": "NOT_RUN",
    }
    sealed_link = T12B / "generators" / "sealed"
    report["sealed_is_symlink"] = sealed_link.is_symlink()
    report["sealed_resolve"] = str(sealed_link.resolve())
    if not SEALED_ALL.is_file():
        raise FileNotFoundError(f"missing sealed batch: {SEALED_ALL}")
    all_sha = sha256_file(SEALED_ALL)
    report["ALL_SEQUENCES_sha256"] = all_sha
    report["ALL_SEQUENCES_sha256_match"] = all_sha == EXPECTED_ALL_SHA
    t113_sha_path = T12B.parent / "t11_3" / "PREREG.sha256"
    t113 = t113_sha_path.read_text().split()[0].strip()
    report["t11_3_prereg_sha256"] = t113
    report["t11_3_untouched"] = t113 == EXPECTED_T113_SHA
    report["crossover_doc_exists"] = CROSSOVER.is_file()
    report["arms"] = list(arm_factory(mode="primary"))
    report["arm_smoke"] = smoke_decide(mode="primary")
    report["ok"] = True
    return report


def run_primary() -> int:
    if not PREREG_SHA_PATH.is_file():
        print("REFUSE primary: missing PREREG.sha256 — freeze first", file=sys.stderr)
        return 2
    prereg_sha = PREREG_SHA_PATH.read_text().split()[0].strip()
    aff = sha256_file(ROOT / "t11" / "imports" / "hum_measure" / "affinity_inputs.py")
    if aff != EXPECTED_AFFINITY:
        print(f"REFUSE: Affinity sha drifted {aff}", file=sys.stderr)
        return 3
    sequences = load_sequences()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_reports = []
    for seed in SEEDS:
        print(f"=== T12b primary seed {seed} ===")
        rep = run_seed(seed, sequences)
        # strip heavy cell lists from per-seed file? keep them — designed receipts
        slim = {k: v for k, v in rep.items()}
        out = OUT_DIR / f"T12B_SEED_{seed}.json"
        out.write_text(json.dumps(slim, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {out} gap_cells={rep['n_gap_cells']} CSR_gap={rep['CSR_gap']}")
        seed_reports.append(rep)

    agg = aggregate(seed_reports, prereg_sha)
    # drop per-seed heavy cells from aggregate payload
    agg_path = OUT_DIR / "T12B_PRIMARY_AGGREGATE.json"
    agg_path.write_text(json.dumps(agg, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_lines = [
        "# T12b Primary AGGREGATE",
        "",
        f"- **PT:** {agg['pt']}",
        f"- **PREREG.sha256:** `{prereg_sha}`",
        f"- **Seeds:** {agg['seeds']}",
        f"- **Primary contrast:** `{agg['primary_contrast']}`",
        f"- **Labeled secondary:** `{agg['labeled_secondary']}`",
        f"- **Affinity:** `{EXPECTED_AFFINITY[:16]}…` untouched",
        f"- **G9/G10:** counted HumState; mid-gap lure active",
        "",
        "## CSR_gap (seed-mean)",
        "",
        "| Arm | CSR_gap | role |",
        "|-----|--------:|------|",
    ]
    roles = {
        "hum_measure": "primary counted",
        "one_step_memory": "primary null baseline",
        "stateless_argmax": "primary control",
        "majority_k3": "labeled secondary",
    }
    for a in ARMS:
        v = agg["CSR_gap"].get(a)
        md_lines.append(f"| `{a}` | {v if v is None else f'{v:.6f}'} | {roles.get(a, '')} |")
    md_lines += [
        "",
        "## Deltas",
        "",
        "| Contrast | Δ CSR_gap |",
        "|----------|----------:|",
    ]
    for k, v in agg["deltas"].items():
        md_lines.append(f"| {k} | {v if v is None else f'{v:.6f}'} |")
    md_lines += [
        "",
        f"- **seed_interval:** {SEED_INTERVAL}",
        f"- **null_holds (hum ≤ one_step + interval):** {agg['null_holds_hum_le_one_step_plus_interval']}",
        f"- **support/null hint:** `{agg['support_null_hint']}` (numbers-only close deferred)",
        "",
    ]
    md_path = OUT_DIR / "T12B_PRIMARY_AGGREGATE.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"wrote {agg_path}")
    print(f"wrote {md_path}")
    print(json.dumps({k: agg[k] for k in ['CSR_gap', 'deltas', 'support_null_hint']}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="T12b primary")
    ap.add_argument("--dry-run", action="store_true", help="Validate wiring; no primary")
    args = ap.parse_args()
    if args.dry_run:
        rep = dry_run()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "DRY_RUN.json"
        out.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(rep, indent=2, sort_keys=True))
        print(f"wrote {out}")
        return 0 if rep.get("ok") else 1
    return run_primary()


if __name__ == "__main__":
    raise SystemExit(main())
