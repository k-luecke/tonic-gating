#!/usr/bin/env python3
"""Calibrate T11 context-classifier MLP on a disjoint sealed-seed split.

Split (documented, freeze-ready):
  eval / smoke seed : 20260923   — NEVER used for fit or temperature
  train seeds       : 20260924, 20260925, 20260926
  val seed          : 20260927   — early-stop + temperature only

Does NOT freeze PREREG. Does NOT train on the smoke seed alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
T11_EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T11_EXP))

from routing.features import (  # noqa: E402
    LEXICON_NAMES,
    feature_dim,
    feature_names,
    featurize,
)
from routing.lexicon import first_n_tokens, refuse_match  # noqa: E402
from routing.mlp_core import (  # noqa: E402
    LABELS,
    LABEL_TO_I,
    accuracy,
    fit_temperature,
    train_mlp,
)

SEALED_ALL = T11_EXP / "generators" / "sealed" / "ALL_SEQUENCES.jsonl"
EXPECTED_ALL_SHA = "305c446f4e773bafe6c77d4224e7ba3978cb354cd7f2f9dbfa2e18156eca8da1"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
DOCS = ROOT / "t11" / "docs"
RECEIPTS = DOCS / "receipts"

EVAL_SEED = 20260923
TRAIN_SEEDS = [20260924, 20260925, 20260926]
VAL_SEED = 20260927
FIRST_N = 32
HIDDEN = 24
UNIGRAM_K = 48
TRAIN_SEED_RNG = 42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_segments(path: Path, seeds: Sequence[int] | None) -> List[Dict[str, Any]]:
    seed_set = set(seeds) if seeds is not None else None
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            seq = json.loads(line)
            if seed_set is not None and seq["seed"] not in seed_set:
                continue
            for i, seg in enumerate(seq["segments"]):
                rows.append(
                    {
                        "seed": seq["seed"],
                        "sequence_type": seq["sequence_type"],
                        "seg_i": i,
                        "text": seg["text"],
                        "oracle_specialist": seg["oracle_specialist"],
                        "family": seg.get("family"),
                    }
                )
    return rows


def build_unigram_vocab(segments: List[Dict[str, Any]], k: int) -> List[str]:
    """Top-k tokens from train first-32 windows; exclude pure digits."""
    c: Counter = Counter()
    for seg in segments:
        toks = first_n_tokens(seg["text"], FIRST_N)
        # skip refuse segments for vocab? still include — refuse is rare/absent in sealed
        c.update(toks)
    # Prefer contentful tokens; drop ultra-short stoppers lightly
    ranked = [t for t, _ in c.most_common(k * 3) if not t.isdigit() and len(t) > 1]
    return ranked[:k]


def matrixize(
    segments: List[Dict[str, Any]], unigram_vocab: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], int]:
    xs: List[List[float]] = []
    ys: List[int] = []
    kept: List[Dict[str, Any]] = []
    abstained = 0
    for seg in segments:
        toks = first_n_tokens(seg["text"], FIRST_N)
        if refuse_match(toks) is not None:
            abstained += 1
            continue
        feats, _ = featurize(seg["text"], first_n=FIRST_N, unigram_vocab=unigram_vocab)
        lab = seg["oracle_specialist"]
        if lab not in LABEL_TO_I:
            continue
        xs.append(feats)
        ys.append(LABEL_TO_I[lab])
        kept.append(seg)
    X = np.asarray(xs, dtype=np.float64) if xs else np.zeros((0, feature_dim(unigram_vocab)))
    y = np.asarray(ys, dtype=np.int64)
    return X, y, kept, abstained


def eval_split(mlp, X, y) -> Dict[str, Any]:
    if len(y) == 0:
        return {"n": 0, "acc": None, "per_label": {}}
    pred = mlp.predict(X)
    per = {}
    for i, lab in enumerate(LABELS):
        mask = y == i
        per[lab] = {
            "n": int(mask.sum()),
            "acc": float(np.mean(pred[mask] == y[mask])) if mask.any() else None,
        }
    return {"n": int(len(y)), "acc": accuracy(pred, y), "per_label": per}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--unigram-k", type=int, default=UNIGRAM_K)
    ap.add_argument("--epochs", type=int, default=500)
    args = ap.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    all_sha = sha256_file(SEALED_ALL)
    if all_sha != EXPECTED_ALL_SHA:
        print("SEAL MISMATCH", all_sha, EXPECTED_ALL_SHA)
        return 2

    train_segs = load_segments(SEALED_ALL, TRAIN_SEEDS)
    val_segs = load_segments(SEALED_ALL, [VAL_SEED])
    eval_segs = load_segments(SEALED_ALL, [EVAL_SEED])

    # Disjointness asserts
    assert EVAL_SEED not in TRAIN_SEEDS and EVAL_SEED != VAL_SEED
    assert VAL_SEED not in TRAIN_SEEDS
    train_ids = {(s["seed"], s["sequence_type"], s["seg_i"]) for s in train_segs}
    eval_ids = {(s["seed"], s["sequence_type"], s["seg_i"]) for s in eval_segs}
    assert train_ids.isdisjoint(eval_ids)

    unigram_vocab = build_unigram_vocab(train_segs, args.unigram_k)
    Xtr, ytr, kept_tr, abs_tr = matrixize(train_segs, unigram_vocab)
    Xva, yva, kept_va, abs_va = matrixize(val_segs, unigram_vocab)
    Xev, yev, kept_ev, abs_ev = matrixize(eval_segs, unigram_vocab)

    print(
        f"split sizes (scored segs): train={len(ytr)} (abs={abs_tr}) "
        f"val={len(yva)} (abs={abs_va}) eval={len(yev)} (abs={abs_ev}) "
        f"in_dim={Xtr.shape[1]}"
    )

    mlp, train_meta = train_mlp(
        Xtr,
        ytr,
        X_val=Xva,
        y_val=yva,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=0.04,
        l2=1e-3,
        batch_size=16,
        seed=TRAIN_SEED_RNG,
        patience=50,
    )
    T = fit_temperature(mlp, Xva, yva)
    mlp.temperature = T

    metrics = {
        "train": eval_split(mlp, Xtr, ytr),
        "val": eval_split(mlp, Xva, yva),
        "eval_smoke_seed": eval_split(mlp, Xev, yev),
        "temperature": T,
        "abstentions": {"train": abs_tr, "val": abs_va, "eval": abs_ev},
    }
    print(json.dumps(metrics, indent=2))

    now_pt = datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + " PT"
    blob: Dict[str, Any] = {
        "format": "t11-classifier-mlp-v1",
        "created": now_pt,
        "status": "calibrated",
        "freeze_ready_weights": True,
        "prereg_frozen": False,
        "architecture": {
            "type": "2-layer MLP",
            "activation": "ReLU",
            "in_dim": int(Xtr.shape[1]),
            "hidden": args.hidden,
            "out_dim": 3,
            "labels": list(LABELS),
            "first_n_tokens": FIRST_N,
            "temperature": T,
            "loss": "softmax cross-entropy + L2",
            "optimizer": "Adam (numpy)",
        },
        "features": {
            "lexicon_names": LEXICON_NAMES,
            "lexicon_dim": len(LEXICON_NAMES),
            "unigram_vocab": unigram_vocab,
            "unigram_k": len(unigram_vocab),
            "feature_names": feature_names(unigram_vocab),
            "note": "Lexicon dims fixed from HAND_RULE; unigram vocab fit on train seeds only.",
        },
        "split": {
            "batch": "t11-counting-sealed-v1",
            "ALL_SEQUENCES_sha256": all_sha,
            "eval_seed": EVAL_SEED,
            "train_seeds": TRAIN_SEEDS,
            "val_seed": VAL_SEED,
            "rule": (
                "Train on seeds 20260924–26; val=20260927 for early-stop + temperature; "
                "eval/smoke seed 20260923 held out entirely (never fit)."
            ),
            "n_train_segments_scored": int(len(ytr)),
            "n_val_segments_scored": int(len(yva)),
            "n_eval_segments_scored": int(len(yev)),
        },
        "train_meta": train_meta,
        "metrics": metrics,
        "weights": mlp.to_dict(),
        "disclosure": (
            "Calibrated 2-layer MLP on first-32 token features (HAND_RULE lexicon counts + "
            "train-only unigram bag). Disjoint from smoke seed 20260923. "
            "Base form stateless; classifier-buffer carry uses EMA of feature vectors."
        ),
    }

    weights_path = ARTIFACTS / "classifier_mlp_v1.json"
    raw = json.dumps(blob, indent=2, sort_keys=True).encode("utf-8")
    weights_path.write_bytes(raw)
    weights_sha = sha256_bytes(raw)

    # Compact npz for numeric reload (same arrays)
    npz_path = ARTIFACTS / "classifier_mlp_v1.npz"
    np.savez_compressed(
        npz_path,
        W1=mlp.W1,
        b1=mlp.b1,
        W2=mlp.W2,
        b2=mlp.b2,
        temperature=np.array([mlp.temperature], dtype=np.float64),
    )
    npz_sha = sha256_file(npz_path)

    split_path = ARTIFACTS / "classifier_split.json"
    split_doc = {
        "created": now_pt,
        "split": blob["split"],
        "architecture": blob["architecture"],
        "metrics": metrics,
        "weights_json": str(weights_path),
        "weights_json_sha256": weights_sha,
        "weights_npz": str(npz_path),
        "weights_npz_sha256": npz_sha,
    }
    split_bytes = json.dumps(split_doc, indent=2, sort_keys=True).encode("utf-8")
    split_path.write_bytes(split_bytes)
    split_sha = sha256_bytes(split_bytes)

    sha_list = ARTIFACTS / "CLASSIFIER_WEIGHTS.sha256"
    with open(sha_list, "w", encoding="utf-8") as f:
        f.write(f"{weights_sha}  {weights_path.name}\n")
        f.write(f"{npz_sha}  {npz_path.name}\n")
        f.write(f"{split_sha}  {split_path.name}\n")

    # Calibration receipt
    receipt_md = DOCS / "T11_CLASSIFIER_CALIBRATION_RECEIPT.md"
    lines = [
        "# T11 classifier calibration receipt",
        "",
        "| Field | Value |",
        "|-------|--------|",
        f"| **Date** | {now_pt} |",
        f"| **Format** | `t11-classifier-mlp-v1` |",
        f"| **Batch** | `t11-counting-sealed-v1` |",
        f"| **PREREG** | **not frozen** |",
        f"| **Weights sha256** | `{weights_sha}` |",
        "",
        "## Split (disjoint)",
        "",
        f"- **Train seeds:** `{TRAIN_SEEDS}` (n_scored={len(ytr)})",
        f"- **Val seed:** `{VAL_SEED}` (early-stop + temperature; n_scored={len(yva)})",
        f"- **Eval / smoke seed:** `{EVAL_SEED}` — **held out; never fit** (n_scored={len(yev)})",
        f"- Sealed `ALL_SEQUENCES.jsonl` sha256 `{all_sha}`",
        "",
        "## Architecture",
        "",
        f"- 2-layer MLP: {Xtr.shape[1]} → ReLU({args.hidden}) → 3 (Browser/Structured/General)",
        f"- Features: {len(LEXICON_NAMES)} HAND_RULE lexicon counts + {len(unigram_vocab)} train-only unigrams; first-{FIRST_N} tokens",
        f"- Temperature (val-calibrated): {T:.4f}",
        f"- Optimizer: numpy Adam; L2={train_meta['l2']}; epochs_ran={train_meta['epochs_ran']}",
        "",
        "## Metrics (segment CSR ≈ M1; abstentions excluded)",
        "",
        f"- Train acc: {metrics['train']['acc']}",
        f"- Val acc: {metrics['val']['acc']}",
        f"- Eval (smoke seed {EVAL_SEED}) acc: {metrics['eval_smoke_seed']['acc']}",
        "",
        "## Artifacts",
        "",
        f"- `{weights_path}` sha256 `{weights_sha}`",
        f"- `{npz_path}` sha256 `{npz_sha}`",
        f"- `{split_path}` sha256 `{split_sha}`",
        f"- `{sha_list}`",
        "",
        "## Notes",
        "",
        "- Stub (`classifier_stub`) remains as disclosed untrained baseline.",
        "- Calibrated arm loads hashed JSON weights via `classifier_mlp.py`.",
        "- Do not freeze PREREG.",
        "",
    ]
    receipt_md.write_text("\n".join(lines), encoding="utf-8")
    receipt_sha = sha256_file(receipt_md)

    receipt_json = RECEIPTS / "T11_CLASSIFIER_CALIBRATION_RECEIPT.json"
    rj = {
        **split_doc,
        "receipt_md": str(receipt_md),
        "receipt_md_sha256": receipt_sha,
        "ALL_SEQUENCES_sha256": all_sha,
        "prereg_frozen": False,
    }
    receipt_json.write_text(json.dumps(rj, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "weights": str(weights_path),
                "weights_sha256": weights_sha,
                "npz_sha256": npz_sha,
                "eval_acc": metrics["eval_smoke_seed"]["acc"],
                "val_acc": metrics["val"]["acc"],
                "train_acc": metrics["train"]["acc"],
                "temperature": T,
                "receipt": str(receipt_md),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
