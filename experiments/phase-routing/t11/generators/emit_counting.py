#!/usr/bin/env python3
"""Emit first sealed T11 counting-sequence batch (JSONL + MANIFEST).

Not smoke. No model calls. Does NOT freeze PREREG.
Requires T8 rerun gate lifted (caller responsibility).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ambiguous import generate_ambiguous
from clean import generate_clean
from common import (
    CARRY_MODES,
    MINIWOB_SEALED_PATH,
    PLACEMENTS,
    SEEDS,
    Sequence,
    load_miniwob_sealed,
    sha256_file,
    write_jsonl,
)
from distraction import generate_distraction
from renewal import generate_renewal
from window_break import generate_window_break

SEALED = HERE / "sealed"
HAND_RULE = Path("/workspace/scp/t11/HAND_RULE.md")
BATCH_ID = "t11-counting-sealed-v1"
BATCH_DATE = "2026-09-23"


def _count_segments(seqs: List[Sequence]) -> int:
    return sum(len(s.segments) for s in seqs)


def _browser_episodes(seqs: List[Sequence]) -> int:
    n = 0
    for s in seqs:
        for seg in s.segments:
            if seg.browser_episode is not None:
                n += 1
    return n


def main() -> None:
    SEALED.mkdir(parents=True, exist_ok=True)

    clean_seqs = [generate_clean(s) for s in SEEDS]
    amb_seqs = [generate_ambiguous(s) for s in SEEDS]
    dist_seqs = [generate_distraction(s) for s in SEEDS]
    renew_seqs = [generate_renewal(s) for s in SEEDS]
    wb_seqs: List[Sequence] = []
    for s in SEEDS:
        for carry in CARRY_MODES:
            for placement in PLACEMENTS:
                wb_seqs.append(
                    generate_window_break(s, carry_mode=carry, placement=placement)
                )

    outputs: Dict[str, List[Sequence]] = {
        "clean.jsonl": clean_seqs,
        "ambiguous.jsonl": amb_seqs,
        "distraction.jsonl": dist_seqs,
        "renewal.jsonl": renew_seqs,
        "window_break.jsonl": wb_seqs,
    }

    file_meta = []
    for name, seqs in outputs.items():
        path = SEALED / name
        write_jsonl(path, seqs)
        file_meta.append(
            {
                "path": str(path),
                "relpath": f"sealed/{name}",
                "n_sequences": len(seqs),
                "n_segments": _count_segments(seqs),
                "n_browser_episodes": _browser_episodes(seqs),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )

    miniwob = load_miniwob_sealed()
    miniwob_sha = sha256_file(MINIWOB_SEALED_PATH)
    hand_rule_sha = sha256_file(HAND_RULE) if HAND_RULE.exists() else None

    total_sequences = sum(m["n_sequences"] for m in file_meta)
    total_segments = sum(m["n_segments"] for m in file_meta)
    total_browser_eps = sum(m["n_browser_episodes"] for m in file_meta)

    # Expected geometry (locked for this batch id)
    expected = {
        "seeds": SEEDS,
        "n_seeds": len(SEEDS),
        "clean": len(SEEDS),
        "ambiguous": len(SEEDS),
        "distraction": len(SEEDS),
        "renewal": len(SEEDS),
        "window_break": len(SEEDS) * len(CARRY_MODES) * len(PLACEMENTS),
        "carry_modes": CARRY_MODES,
        "placements": PLACEMENTS,
    }
    assert expected["window_break"] == 5 * 4 * 2 == 40
    assert total_sequences == 5 + 5 + 5 + 5 + 40 == 60

    manifest = {
        "batch_id": BATCH_ID,
        "status": "sealed",
        "ready_for_routing_smoke": True,
        "prereg_frozen": False,
        "created_pt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "date": BATCH_DATE,
        "work_tree": "/workspace/scp",
        "generators_dir": str(HERE),
        "output_dir": str(SEALED),
        "seeds": SEEDS,
        "sequence_types": [
            "Clean",
            "Ambiguous",
            "Distraction",
            "Renewal",
            "Window-break",
        ],
        "geometry": expected,
        "totals": {
            "sequences": total_sequences,
            "segments": total_segments,
            "browser_episodes_linked": total_browser_eps,
        },
        "files": file_meta,
        "inputs": {
            "MINIWOB25_SEALED.json": {
                "path": str(MINIWOB_SEALED_PATH),
                "sha256": miniwob_sha,
                "num_tasks": miniwob.get("num_tasks"),
                "num_seeds": miniwob.get("num_seeds"),
                "episode_seeds": miniwob.get("seeds"),
                "note": "enter-password omitted from browser prompts (HAND_RULE refuse word)",
            },
            "HAND_RULE.md": {
                "path": str(HAND_RULE),
                "sha256": hand_rule_sha,
            },
        },
        "notes": [
            "Content-defined oracle_specialist; no planted tags in segment text.",
            "Refuse-list validated at write time (assert_no_refuse_cues).",
            "Window-break carry_mode/placement are harness metadata only.",
            "Browser segments link sealed MiniWoB task+episode seed when use_miniwob.",
            "Do NOT freeze PREREG yet — this batch enables routing smoke next.",
        ],
    }

    # Combined corpus (one JSONL, all types) for harness convenience
    all_path = SEALED / "ALL_SEQUENCES.jsonl"
    all_seqs: List[Sequence] = []
    for seqs in outputs.values():
        all_seqs.extend(seqs)
    write_jsonl(all_path, all_seqs)
    all_sha = sha256_file(all_path)
    manifest["files"].append(
        {
            "path": str(all_path),
            "relpath": "sealed/ALL_SEQUENCES.jsonl",
            "n_sequences": len(all_seqs),
            "n_segments": _count_segments(all_seqs),
            "n_browser_episodes": _browser_episodes(all_seqs),
            "sha256": all_sha,
            "bytes": all_path.stat().st_size,
        }
    )
    manifest["ALL_SEQUENCES_sha256"] = all_sha

    man_path = SEALED / "SEALED_MANIFEST.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    man_sha = sha256_file(man_path)

    # Sidecar sha list
    sha_path = SEALED / "SEALED_MANIFEST.sha256"
    sha_path.write_text(f"{man_sha}  SEALED_MANIFEST.json\n", encoding="utf-8")
    for m in file_meta + [
        {
            "relpath": "sealed/ALL_SEQUENCES.jsonl",
            "sha256": all_sha,
        }
    ]:
        # already included above for per-type; append ALL if not in file_meta loop
        pass
    with open(SEALED / "FILES.sha256", "w", encoding="utf-8") as f:
        for m in manifest["files"]:
            f.write(f"{m['sha256']}  {Path(m['relpath']).name}\n")
        f.write(f"{man_sha}  SEALED_MANIFEST.json\n")

    print(f"batch_id={BATCH_ID}")
    print(f"output_dir={SEALED}")
    print(f"sequences={total_sequences} segments={total_segments} browser_eps={total_browser_eps}")
    print(f"ALL_SEQUENCES.sha256={all_sha}")
    print(f"SEALED_MANIFEST.sha256={man_sha}")
    for m in manifest["files"]:
        print(f"  {m['relpath']}: n={m['n_sequences']} sha256={m['sha256'][:12]}…")


if __name__ == "__main__":
    main()
