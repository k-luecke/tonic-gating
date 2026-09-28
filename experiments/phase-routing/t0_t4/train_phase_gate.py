"""Exploratory single-seed training for the T3 Phase Gate router."""
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

from phase_routing.data import CONTENT_OFFSET, N_TASKS, VOCAB_SIZE
from phase_routing.factory import make_model
from phase_routing.model import MoETransformer, count_parameters


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def predictive_token_mask(tokens: torch.Tensor) -> torch.Tensor:
    return (tokens[:, :-1] >= CONTENT_OFFSET) & (tokens[:, 1:] >= CONTENT_OFFSET)


def predictive_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    targets = tokens[:, 1:]
    per_token = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).view_as(targets)
    return per_token[predictive_token_mask(tokens)].mean()


@torch.no_grad()
def evaluate(
    model: MoETransformer,
    array: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    model.eval()
    loss_sum = 0.0
    token_count = 0
    active_sum = 0
    routed_token_count = 0
    dispatch_counts = np.zeros(N_TASKS, dtype=np.int64)
    gain_sums = np.zeros(N_TASKS, dtype=np.float64)
    task_loss_sums = np.zeros(N_TASKS, dtype=np.float64)
    task_token_counts = np.zeros(N_TASKS, dtype=np.int64)

    for start in range(0, len(array), batch_size):
        tokens = torch.from_numpy(array[start : start + batch_size].astype(np.int64, copy=False)).to(device)
        logits, auxiliary = model(tokens[:, :-1])
        if float(auxiliary) != 0.0:
            raise RuntimeError("Phase Gate must not produce an auxiliary loss")
        route = model.last_router_output
        assert route is not None
        targets = tokens[:, 1:]
        mask = predictive_token_mask(tokens)
        nll = nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1), reduction="none"
        ).view_as(targets)
        loss_sum += float(nll[mask].sum())
        token_count += int(mask.sum())

        active = route.gates > 0
        active_sum += int(active.sum())
        routed_token_count += active.shape[0] * active.shape[1]
        selected = route.indices[active].cpu().numpy()
        dispatch_counts += np.bincount(selected, minlength=N_TASKS)
        gain_sums += route.probabilities.sum(dim=(0, 1)).cpu().double().numpy()

        initial_tasks = tokens[:, 0].cpu().numpy()
        for row, task in enumerate(initial_tasks):
            row_count = int(mask[row].sum())
            task_loss_sums[task] += float(nll[row][mask[row]].sum())
            task_token_counts[task] += row_count

    result: dict[str, object] = {
        "loss": loss_sum / token_count,
        "mean_active_experts_per_token": active_sum / routed_token_count,
        "dispatch_fraction": (dispatch_counts / dispatch_counts.sum()).tolist(),
        "normalized_gain_fraction": (gain_sums / gain_sums.sum()).tolist(),
    }
    if np.all(task_token_counts > 0):
        result["task_loss"] = (task_loss_sums / task_token_counts).tolist()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/exploratory"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    seed_everything(args.seed)
    threads = min(8, os.cpu_count() or 1)
    torch.set_num_threads(threads)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_array = np.load(args.data_dir / "train.npy", allow_pickle=False)
    heldout = np.load(args.data_dir / "heldout.npy", allow_pickle=False)
    switching = np.load(args.data_dir / "switching.npy", allow_pickle=False)
    dataset = TensorDataset(torch.from_numpy(train_array.astype(np.int64, copy=False)))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )

    model = make_model("phase_gate").to(device)
    router = model.blocks[2].ffn.router
    initial_phases = router.task_phases.detach().clone()
    parameter_count = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    warmup = min(50, max(1, args.steps // 10))

    print(f"exploratory=true t3=true seed={args.seed} device={device} threads={threads}")
    print(f"parameters={parameter_count} steps={args.steps} auxiliary_loss=none")
    started = time.perf_counter()
    iterator = iter(loader)
    final_train_loss = math.nan
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
            scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * scale

        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, auxiliary = model(tokens[:, :-1])
        if float(auxiliary.detach()) != 0.0:
            raise RuntimeError("Phase Gate must not produce an auxiliary loss")
        loss = predictive_loss(logits, tokens)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_train_loss = float(loss.detach())
        if step == 1 or step % 50 == 0 or step == args.steps:
            print(
                f"step={step:04d} lm_loss={final_train_loss:.4f} "
                f"elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )

    elapsed_seconds = time.perf_counter() - started
    heldout_metrics = evaluate(model, heldout, device, args.batch_size)
    switching_metrics = evaluate(model, switching, device, args.batch_size)
    phase_offsets = torch.atan2(
        torch.sin(router.task_phases.detach() - initial_phases),
        torch.cos(router.task_phases.detach() - initial_phases),
    )

    metrics = {
        "label": "exploratory single-seed T3 Phase Gate",
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "threads": threads,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "elapsed_seconds": elapsed_seconds,
        "parameter_count": parameter_count,
        "final_train_loss": final_train_loss,
        "auxiliary_loss": None,
        "expert_phases": router.expert_phases.detach().cpu().tolist(),
        "learned_task_phases": router.task_phases.detach().cpu().tolist(),
        "task_phase_offsets_from_initial": phase_offsets.cpu().tolist(),
        "geometric_active_prediction_at_aligned_phase": 3.0,
        "geometric_active_prediction_generic_phase": 4.0,
        "heldout": heldout_metrics,
        "switching": switching_metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"t3_phase_gate_metrics_seed{args.seed}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    checkpoint_path = args.output_dir / f"t3_phase_gate_seed{args.seed}.pt"
    torch.save({"model_state_dict": model.state_dict(), "metrics": metrics}, checkpoint_path)

    print(f"elapsed_seconds={elapsed_seconds:.3f}")
    print(
        "heldout_loss={:.4f} active_experts_per_token={:.4f}".format(
            heldout_metrics["loss"], heldout_metrics["mean_active_experts_per_token"]
        )
    )
    print(
        "switching_loss={:.4f} active_experts_per_token={:.4f}".format(
            switching_metrics["loss"], switching_metrics["mean_active_experts_per_token"]
        )
    )
    print(f"metrics={metrics_path} checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
