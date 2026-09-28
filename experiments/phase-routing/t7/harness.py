"""T7 evaluation harness — T4 definitions adapted for inferred PG + LR-goal."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from t7_routing.costs import (
    DISTURBANCE_GRID,
    PRIMARY_ROBUSTNESS_UNITS,
    expert_compute_cost,
    routing_decision_cost,
)
from t7_routing.data import CONTENT_OFFSET, N_TASKS, VOCAB_SIZE
from t7_routing.factory import make_model

SOURCE_GLOBS = (
    "harness.py",
    "train.py",
    "t7_routing/*.py",
    "tests/*.py",
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_manifest_sha256(root: Path) -> dict[str, Any]:
    files: list[Path] = []
    for pattern in SOURCE_GLOBS:
        files.extend(sorted(root.glob(pattern)))
    files = sorted({path.resolve() for path in files if path.is_file()})
    lines: list[str] = []
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root.resolve()).as_posix()
        file_hash = sha256_file(path)
        line = f"{file_hash}  {relative}"
        lines.append(line)
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return {
        "algorithm": "sha256-manifest",
        "files": lines,
        "manifest_sha256": digest.hexdigest(),
    }


def data_manifest_hash(data_dir: Path) -> str:
    return sha256_file(data_dir / "MANIFEST.sha256")


def dense_dispatch(route: Any) -> torch.Tensor:
    dense = torch.zeros(
        *route.indices.shape[:-1], N_TASKS, device=route.gates.device, dtype=route.gates.dtype
    )
    valid = route.indices >= 0
    dense.scatter_add_(-1, route.indices.clamp_min(0), route.gates * valid)
    return dense


@torch.no_grad()
def hidden_std(model: nn.Module, array: np.ndarray, device: torch.device, batch: int) -> float:
    sums = [0.0, 0.0, 0]

    def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        del module, kwargs
        x = args[0].double()
        sums[0] += float(x.sum())
        sums[1] += float((x * x).sum())
        sums[2] += x.numel()

    handle = model.blocks[2].ffn.register_forward_pre_hook(hook, with_kwargs=True)
    model.eval()
    model.configure_eval_disturbance()
    for start in range(0, len(array), batch):
        tokens = torch.from_numpy(array[start : start + batch].astype(np.int64, copy=False)).to(device)
        model(tokens[:, :-1])
    handle.remove()
    mean = sums[0] / sums[2]
    return math.sqrt(max(0.0, sums[1] / sums[2] - mean * mean))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    array: np.ndarray,
    device: torch.device,
    batch: int,
    *,
    arm: str,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    loss_sums = np.zeros(N_TASKS)
    counts = np.zeros(N_TASKS, dtype=np.int64)
    dispatch = np.zeros(N_TASKS)
    profiles = np.zeros((N_TASKS, N_TASKS))
    profile_counts = np.zeros(N_TASKS)
    active_sum = token_count = 0
    for start in range(0, len(array), batch):
        tokens = torch.from_numpy(array[start : start + batch].astype(np.int64, copy=False)).to(device)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        logits, _ = model(inputs)
        route = model.last_router_output
        assert route is not None
        mask = (inputs >= CONTENT_OFFSET) & (targets >= CONTENT_OFFSET)
        nll = nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1), reduction="none"
        ).view_as(targets)
        tasks = model.resolve_task_ids(inputs)
        dense = dense_dispatch(route)
        valid_routes = (route.indices >= 0) & (route.gates > 0)
        active_sum += int(valid_routes.sum())
        token_count += route.indices.shape[0] * route.indices.shape[1]
        dispatch += dense.sum((0, 1)).cpu().double().numpy()
        for task in range(N_TASKS):
            chosen = mask & (tasks == task)
            n = int(chosen.sum())
            if n:
                loss_sums[task] += float(nll[chosen].sum())
                counts[task] += n
                profiles[task] += dense[chosen].sum(0).cpu().double().numpy()
                profile_counts[task] += n
    task_loss = np.divide(loss_sums, counts, out=np.full(N_TASKS, np.nan), where=counts > 0)
    loads = dispatch / max(dispatch.sum(), 1e-12)
    mean_active = active_sum / token_count
    metrics: dict[str, Any] = {
        "L": float(loss_sums.sum() / counts.sum()),
        "L_per_task": task_loss.tolist(),
        "load_balance_cv": float(loads.std() / loads.mean()),
        "load_balance_B": float(loads.std() / loads.mean()),
        "dispatch_fraction": loads.tolist(),
    }
    metrics.update(routing_decision_cost(arm))
    metrics.update(expert_compute_cost(mean_active))
    metrics["C_route_legacy_active_dispatches_per_token"] = float(mean_active)
    metrics["C_exp_legacy_experts_used_anywhere"] = int((dispatch > 0).sum())
    return metrics, profiles / np.maximum(profile_counts[:, None], 1)


def dominant_sets(profiles: np.ndarray) -> list[list[int]]:
    result = []
    for profile in profiles:
        order = np.argsort(-profile)
        cumulative = 0.0
        selected: list[int] = []
        for expert in order:
            selected.append(int(expert))
            cumulative += float(profile[expert])
            if cumulative >= 0.5:
                break
        result.append(selected)
    return result


@torch.no_grad()
def switching_time(
    model: nn.Module,
    switching: np.ndarray,
    sets: list[list[int]],
    device: torch.device,
    batch: int,
    *,
    arm: str,
) -> dict[str, Any]:
    model.eval()
    times: list[int] = []
    failures = 0
    for start in range(0, len(switching), batch):
        tokens = torch.from_numpy(
            switching[start : start + batch].astype(np.int64, copy=False)
        ).to(device)
        model(tokens[:, :-1])
        route = model.last_router_output
        assert route is not None
        dense = dense_dispatch(route)
        for row in range(tokens.shape[0]):
            markers = torch.nonzero(tokens[row] < CONTENT_OFFSET).flatten()
            switch = int(markers[1])
            new_task = int(tokens[row, switch])
            experts = sets[new_task]
            values = dense[row, switch:, experts].sum(-1)
            crossing = torch.nonzero(values > 0.5).flatten()
            if crossing.numel():
                times.append(int(crossing[0]))
            else:
                failures += 1
    task_aware = arm in {"lr", "lr_goal", "pg", "pg_inferred"}
    return {
        "T_sw": float(np.mean(times)) if times else None,
        "T_sw_median": float(np.median(times)) if times else None,
        "T_sw_success_fraction": len(times) / (len(times) + failures),
        "T_sw_index0_means_immediate_at_marker": True,
        "H2_applicable": task_aware and arm != "hr1",
        "H2_note": (
            "H2 compares PG-inferred vs LR only. LR-goal T_sw reported for confound "
            "context. HR crossings are not task-aware."
            if arm in {"hr1", "hr2"}
            else "H2 comparator is PG-inferred vs LR only; LR-goal is confound context."
        ),
        "definition": (
            "first token lag from new marker with >0.5 dispatch mass on new-task heldout "
            "dominant profile; index 0 = immediate response at the marker"
        ),
    }


def cell_name(kind: str, units: float) -> str:
    return f"{kind}_{units:g}std"


def annotate_robustness(
    metrics: dict[str, Any],
    *,
    clean_L: float,
    units: float | None,
) -> dict[str, Any]:
    delta = float(metrics["L"] - clean_L)
    metrics["delta_L_from_clean"] = delta
    metrics["robustness_exp_exploratory"] = math.exp(-delta)
    metrics["primary_robustness_cell"] = bool(
        units is not None and abs(units - PRIMARY_ROBUSTNESS_UNITS) < 1e-12
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/exploratory/smoke"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/exploratory/smoke/harness"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("lr", "hr1", "pg", "pg_inferred", "lr_goal"),
        default=("lr", "lr_goal", "hr1", "pg_inferred"),
    )
    args = parser.parse_args()

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_all(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    heldout = np.load(args.data_dir / "heldout.npy", allow_pickle=False)
    switching = np.load(args.data_dir / "switching.npy", allow_pickle=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = source_tree_manifest_sha256(args.repo_root.resolve())
    data_hash = data_manifest_hash(args.data_dir)

    all_specs = {
        "lr": ("lr", args.checkpoint_dir / f"t7_lr_lr_seed{args.seed}.pt"),
        "hr1": ("hr", args.checkpoint_dir / f"t7_hr1_hr1_seed{args.seed}.pt"),
        "pg": ("pg", args.checkpoint_dir / f"t7_pg_supplied_pg_seed{args.seed}.pt"),
        "pg_inferred": (
            "pg_inferred",
            args.checkpoint_dir / f"t7_pg_inferred_pg_inferred_seed{args.seed}.pt",
        ),
        "lr_goal": ("lr_goal", args.checkpoint_dir / f"t7_lr_goal_lr_goal_seed{args.seed}.pt"),
    }
    specs = {arm: all_specs[arm] for arm in args.arms}

    direction_generator = torch.Generator(device="cpu").manual_seed(args.seed + 101)
    direction = torch.randn(128, generator=direction_generator)
    direction = direction / direction.square().mean().sqrt()

    definition_block = {
        "C_route": (
            "native units: LR=1024 MACs; LR-goal=2048 MACs; HR1=0 MACs; "
            "PG-inferred=256 MACs (phase head Linear width→2) + 8 cosine ops; "
            "supplied PG (secondary)=8 cosine ops. Never combine units."
        ),
        "C_exp": "mean_active_experts_per_token × 65536",
        "load_balance_B": "CV of expert utilization (dispatch mass) over heldout",
        "robustness_primary": (
            "delta_L_from_clean = L(cell)-L(clean); primary cells σ_n=0.3 and δ=0.3. "
            "For PG-inferred, noise/H2 are NOT by-construction (φ_ref from noised hidden)."
        ),
        "delta_L_drop": "mean over 8 single-expert dropouts of L(drop)-L(clean)",
        "T_sw": (
            "per-task dominant expert profile on clean heldout; after switching marker, "
            "first token where >0.5 mass on new-task set; index 0 = immediate; "
            "H2 compares PG-inferred vs LR only"
        ),
        "disturbance_placement": (
            "noise enters routing-decision hidden only; shift added to every MoE-layer "
            "hidden (routing + expert input); same seeded base tensors across arms, "
            "scaled by each arm's own clean heldout MoE hidden std"
        ),
        "disturbance_grid": list(DISTURBANCE_GRID),
    }

    summary: dict[str, Any] = {
        "experiment": "T7",
        "seed": args.seed,
        "disturbance_grid": list(DISTURBANCE_GRID),
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "data_manifest_sha256": data_hash,
        "primary_arms": ["lr", "lr_goal", "hr1", "pg_inferred"],
        "secondary_arms": ["pg"],
        "arms": {},
    }

    for arm, (factory, checkpoint) in specs.items():
        if not checkpoint.exists():
            print(f"SKIP arm={arm} missing checkpoint {checkpoint}")
            continue
        model = make_model(factory).to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        checkpoint_hash = sha256_file(checkpoint)

        std = hidden_std(model, heldout, device, args.batch_size)
        model.configure_eval_disturbance()
        clean, profiles = evaluate(model, heldout, device, args.batch_size, arm=arm)
        clean = annotate_robustness(clean, clean_L=clean["L"], units=None)
        clean["delta_L_drop"] = 0.0
        clean["hidden_std_clean"] = std
        sets = dominant_sets(profiles)
        sw = switching_time(model, switching, sets, device, args.batch_size, arm=arm)
        clean.update(sw)

        switching_metrics, _ = evaluate(model, switching, device, args.batch_size, arm=arm)
        switching_metrics.update(sw)
        switching_metrics["delta_L_from_clean"] = None
        switching_metrics["robustness_exp_exploratory"] = None
        switching_metrics["delta_L_drop"] = None
        switching_metrics["hidden_std_clean"] = std
        switching_metrics["primary_robustness_cell"] = False

        cells: dict[str, dict[str, Any]] = {"clean": clean, "switching_clean": switching_metrics}

        for units in DISTURBANCE_GRID:
            noise_seed = args.seed + 10_000 + int(round(units * 1000))
            seed_all(noise_seed)
            model.configure_eval_disturbance(noise_std=units * std)
            noise_metrics, _ = evaluate(model, heldout, device, args.batch_size, arm=arm)
            noise_metrics = annotate_robustness(noise_metrics, clean_L=clean["L"], units=units)
            noise_metrics.update(
                {
                    "delta_L_drop": None,
                    "T_sw": None,
                    "hidden_std_clean": std,
                    "disturbance_units": units,
                    "disturbance_kind": "noise",
                    "noise_seed": noise_seed,
                }
            )
            cells[cell_name("noise", units)] = noise_metrics

            model.configure_eval_disturbance(shift=direction.to(device) * units * std)
            shift_metrics, _ = evaluate(model, heldout, device, args.batch_size, arm=arm)
            shift_metrics = annotate_robustness(shift_metrics, clean_L=clean["L"], units=units)
            shift_metrics.update(
                {
                    "delta_L_drop": None,
                    "T_sw": None,
                    "hidden_std_clean": std,
                    "disturbance_units": units,
                    "disturbance_kind": "shift",
                    "shift_direction_seed": args.seed + 101,
                }
            )
            cells[cell_name("shift", units)] = shift_metrics

        drops: list[float] = []
        for expert in range(N_TASKS):
            model.configure_eval_disturbance(dropped_experts=(expert,))
            drop_metrics, _ = evaluate(model, heldout, device, args.batch_size, arm=arm)
            delta = float(drop_metrics["L"] - clean["L"])
            drops.append(delta)
            drop_metrics = annotate_robustness(drop_metrics, clean_L=clean["L"], units=None)
            drop_metrics.update(
                {
                    "delta_L_drop": delta,
                    "T_sw": None,
                    "hidden_std_clean": std,
                    "dropped_expert": expert,
                    "primary_robustness_cell": False,
                }
            )
            cells[f"drop_e{expert}"] = drop_metrics
        model.configure_eval_disturbance()

        mean_drop = float(np.mean(drops))
        for cell, metrics in cells.items():
            if cell == "clean":
                metrics["delta_L_drop"] = mean_drop
                metrics["delta_L_drop_mean_over_8_experts"] = mean_drop
                metrics["delta_L_drop_max"] = float(np.max(drops))
            artifact = {
                "experiment": "T7",
                "arm": arm,
                "seed": args.seed,
                "cell": cell,
                "metrics": metrics,
                "dominant_profiles": sets if cell == "clean" else None,
                "definitions": definition_block,
                "provenance": {
                    "source_manifest_sha256": source_manifest["manifest_sha256"],
                    "source_manifest_files": source_manifest["files"],
                    "data_manifest_sha256": data_hash,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "seed": args.seed,
                    "hidden_std_clean": std,
                    "disturbance_grid": list(DISTURBANCE_GRID),
                },
            }
            (args.output_dir / f"{arm}_seed{args.seed}_{cell}.json").write_text(
                json.dumps(artifact, indent=2, sort_keys=True) + "\n"
            )

        summary["arms"][arm] = {
            "clean_L": clean["L"],
            "C_route_MACs_per_token": clean.get("C_route_MACs_per_token"),
            "C_route_ops": clean.get("C_route_ops"),
            "mean_active_experts_per_token": clean["mean_active_experts_per_token"],
            "C_exp_MACs_per_token": clean["C_exp_MACs_per_token"],
            "load_balance_cv": clean["load_balance_cv"],
            "T_sw": clean["T_sw"],
            "mean_delta_L_drop": mean_drop,
            "hidden_std_clean": std,
            "checkpoint_sha256": checkpoint_hash,
            "cells": sorted(cells),
            "n_cells": len(cells),
            "primary_noise_0.3_delta_L": cells[cell_name("noise", 0.3)]["delta_L_from_clean"],
            "primary_shift_0.3_delta_L": cells[cell_name("shift", 0.3)]["delta_L_from_clean"],
        }
        print(
            f"arm={arm} L={clean['L']:.6f} "
            f"C_route_MACs={clean.get('C_route_MACs_per_token')} "
            f"C_route_ops={clean.get('C_route_ops')} "
            f"mean_active={clean['mean_active_experts_per_token']:.3f} "
            f"T_sw={clean['T_sw']} "
            f"ΔL_noise0.3={cells[cell_name('noise', 0.3)]['delta_L_from_clean']:.6f} "
            f"ΔL_shift0.3={cells[cell_name('shift', 0.3)]['delta_L_from_clean']:.6f}"
        )

    summary["source_manifest"] = source_manifest
    (args.output_dir / f"summary_seed{args.seed}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
