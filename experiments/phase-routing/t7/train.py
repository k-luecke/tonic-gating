"""T7 single-seed trainer for LR, LR-goal, HR1, PG-inferred (and secondary PG)."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from t7_routing.data import CONTENT_OFFSET, N_TASKS, VOCAB_SIZE
from t7_routing.factory import make_model
from t7_routing.model import MoETransformer, count_parameters

AUX_ARMS = {"lr", "lr_goal"}
PRIMARY_ARMS = ("lr", "lr_goal", "hr1", "pg_inferred")
ALL_ARMS = ("lr", "lr_goal", "hr1", "pg_inferred", "pg")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def predictive_mask(tokens: torch.Tensor) -> torch.Tensor:
    return (tokens[:, :-1] >= CONTENT_OFFSET) & (tokens[:, 1:] >= CONTENT_OFFSET)


def predictive_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    targets = tokens[:, 1:]
    losses = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).view_as(targets)
    return losses[predictive_mask(tokens)].mean()


def unigram_probabilities(train: np.ndarray) -> np.ndarray:
    result = np.zeros((N_TASKS, VOCAB_SIZE), dtype=np.float64)
    for task in range(N_TASKS):
        rows = train[train[:, 0] == task]
        mask = (rows[:, :-1] >= CONTENT_OFFSET) & (rows[:, 1:] >= CONTENT_OFFSET)
        counts = np.bincount(rows[:, 1:][mask], minlength=VOCAB_SIZE).astype(float)
        counts[CONTENT_OFFSET:] += 1e-9
        result[task] = counts / counts.sum()
    return result


@torch.no_grad()
def evaluate(
    model: MoETransformer,
    array: np.ndarray,
    unigram: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    model.eval()
    loss_sum = np.zeros(N_TASKS)
    baseline_sum = np.zeros(N_TASKS)
    counts = np.zeros(N_TASKS, dtype=np.int64)
    top1_counts = np.zeros((N_TASKS, N_TASKS), dtype=np.int64)
    dispatch = np.zeros(N_TASKS, dtype=np.int64)
    active_sum = routed = 0
    for start in range(0, len(array), batch_size):
        tokens = torch.from_numpy(
            array[start : start + batch_size].astype(np.int64, copy=False)
        ).to(device)
        logits, _ = model(tokens[:, :-1])
        route = model.last_router_output
        assert route is not None
        targets = tokens[:, 1:]
        mask = predictive_mask(tokens)
        nll = nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1), reduction="none"
        ).view_as(targets)
        active = (route.indices >= 0) & (route.gates > 0)
        active_sum += int(active.sum())
        routed += active.shape[0] * active.shape[1]
        selected = route.indices[active].cpu().numpy()
        dispatch += np.bincount(selected, minlength=N_TASKS)
        top1 = route.indices[:, :, 0]
        for row in range(tokens.shape[0]):
            task = int(tokens[row, 0])
            row_mask = mask[row]
            n = int(row_mask.sum())
            values = targets[row][row_mask].cpu().numpy()
            loss_sum[task] += float(nll[row][row_mask].sum())
            counts[task] += n
            baseline_sum[task] += float(-np.log(unigram[task, values]).sum())
            chosen = top1[row][row_mask].cpu().numpy()
            # Clamp negative (inactive) indices out of bincount.
            chosen = np.clip(chosen, 0, N_TASKS - 1)
            top1_counts[task] += np.bincount(chosen, minlength=N_TASKS)
    fractions = top1_counts / np.maximum(top1_counts.sum(1, keepdims=True), 1)
    task_loss = loss_sum / np.maximum(counts, 1)
    baseline = baseline_sum / np.maximum(counts, 1)
    shares = fractions.max(1)
    return {
        "loss": float(loss_sum.sum() / max(counts.sum(), 1)),
        "task_loss": task_loss.tolist(),
        "task_unigram_loss": baseline.tolist(),
        "all_tasks_below_unigram": bool(np.all(task_loss < baseline)),
        "mean_active_experts_per_token": active_sum / max(routed, 1),
        "dispatch_fraction": (dispatch / max(dispatch.sum(), 1)).tolist(),
        "top1_utilization_by_task": fractions.tolist(),
        "dominant_expert": fractions.argmax(1).tolist(),
        "dominant_share": shares.tolist(),
        "dominant_tasks": int((shares >= 0.25).sum()),
        "specialization_acceptance": bool((shares >= 0.25).sum() >= 4),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--router", choices=ALL_ARMS, default="lr")
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--aux-weight", type=float, default=0.01)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("results/exploratory/smoke"))
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--mode", choices=("exploratory", "primary"), default="exploratory")
    args = p.parse_args()
    seed_everything(args.seed)
    threads = min(8, os.cpu_count() or 1)
    torch.set_num_threads(threads)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    train = np.load(args.data_dir / "train.npy", allow_pickle=False)
    heldout = np.load(args.data_dir / "heldout.npy", allow_pickle=False)
    unigram = unigram_probabilities(train)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train.astype(np.int64, copy=False))),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )
    factory_name = {
        "lr": "lr",
        "hr1": "hr",
        "pg": "pg",
        "pg_inferred": "pg_inferred",
        "lr_goal": "lr_goal",
    }[args.router]
    model = make_model(factory_name).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    warmup = min(50, max(1, args.steps // 10))
    iterator = iter(loader)
    started = time.perf_counter()
    final = math.nan
    print(
        f"mode={args.mode} arm={args.router} seed={args.seed} "
        f"device={device} parameters={params} steps={args.steps}"
    )
    for step in range(1, args.steps + 1):
        try:
            (tokens,) = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            (tokens,) = next(iterator)
        tokens = tokens.to(device)
        if step <= warmup:
            scale = step / warmup
        else:
            progress = (step - warmup) / max(1, args.steps - warmup)
            scale = 0.1 + 0.45 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * scale
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, aux = model(tokens[:, :-1])
        lm = predictive_loss(logits, tokens)
        weight = args.aux_weight if args.router in AUX_ARMS else 0.0
        (lm + weight * aux).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final = float(lm.detach())
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(
                f"step={step:04d} lm_loss={final:.5f} aux={float(aux.detach()):.4f} "
                f"elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    evaluation = evaluate(model, heldout, unigram, device, args.batch_size)
    metrics: dict[str, object] = {
        "label": f"T7 {args.mode} seed-run {args.router}",
        "experiment": "T7",
        "mode": args.mode,
        "arm": args.router,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "aux_weight": args.aux_weight if args.router in AUX_ARMS else 0.0,
        "device": str(device),
        "threads": threads,
        "torch_version": str(torch.__version__),
        "elapsed_seconds": elapsed,
        "parameter_count": params,
        "final_train_loss": final,
        "shared_task_embedding": (
            "most-recent marker identity broadcast to every token, common to all arms"
        ),
        "router_goal_channel": (
            "concat(hidden, router-only goal_embedding[task_id]) into router Linear"
            if args.router == "lr_goal"
            else None
        ),
        "phi_ref_source": (
            "Linear(width→2) → atan2; tokens ignored for routing"
            if args.router == "pg_inferred"
            else ("marker→task_phases lookup" if args.router == "pg" else None)
        ),
        "heldout": evaluation,
    }
    metrics.update(
        {
            k: evaluation[k]
            for k in (
                "task_loss",
                "task_unigram_loss",
                "all_tasks_below_unigram",
                "dominant_expert",
                "dominant_share",
                "dominant_tasks",
                "specialization_acceptance",
                "top1_utilization_by_task",
            )
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = {
        "lr": "t7_lr",
        "hr1": "t7_hr1",
        "pg": "t7_pg_supplied",
        "pg_inferred": "t7_pg_inferred",
        "lr_goal": "t7_lr_goal",
    }[args.router]
    metrics_path = args.output_dir / f"{prefix}_metrics_seed{args.seed}.json"
    ckpt_path = args.output_dir / f"{prefix}_{args.router}_seed{args.seed}.pt"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    torch.save(
        {"model_state_dict": model.state_dict(), "metrics": metrics, "arm": args.router},
        ckpt_path,
    )
    print(
        f"elapsed_seconds={elapsed:.3f} heldout_loss={evaluation['loss']:.6f} "
        f"active={evaluation['mean_active_experts_per_token']:.3f}"
    )
    print(
        f"dominant_tasks={evaluation['dominant_tasks']}/8 "
        f"metrics={metrics_path} checkpoint={ckpt_path}"
    )


if __name__ == "__main__":
    main()
