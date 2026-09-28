"""T4 evaluation-only disturbance and switching harness."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from phase_routing.data import CONTENT_OFFSET, N_TASKS, VOCAB_SIZE
from phase_routing.factory import make_model


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def dense_dispatch(route: Any) -> torch.Tensor:
    dense = torch.zeros(*route.indices.shape[:-1], N_TASKS, device=route.gates.device, dtype=route.gates.dtype)
    valid = route.indices >= 0
    dense.scatter_add_(-1, route.indices.clamp_min(0), route.gates * valid)
    return dense


@torch.no_grad()
def hidden_std(model: nn.Module, array: np.ndarray, device: torch.device, batch: int) -> float:
    sums = [0.0, 0.0, 0]
    def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        del module, kwargs
        x = args[0].double(); sums[0] += float(x.sum()); sums[1] += float((x*x).sum()); sums[2] += x.numel()
    handle = model.blocks[2].ffn.register_forward_pre_hook(hook, with_kwargs=True)
    model.eval(); model.configure_eval_disturbance()
    for start in range(0, len(array), batch):
        tokens = torch.from_numpy(array[start:start+batch].astype(np.int64, copy=False)).to(device)
        model(tokens[:, :-1])
    handle.remove()
    mean = sums[0] / sums[2]
    return math.sqrt(max(0.0, sums[1] / sums[2] - mean * mean))


@torch.no_grad()
def evaluate(model: nn.Module, array: np.ndarray, device: torch.device, batch: int) -> tuple[dict[str, Any], np.ndarray]:
    model.eval(); loss_sums = np.zeros(N_TASKS); counts = np.zeros(N_TASKS, dtype=np.int64)
    dispatch = np.zeros(N_TASKS); profiles = np.zeros((N_TASKS, N_TASKS)); profile_counts = np.zeros(N_TASKS)
    active_sum = token_count = 0
    for start in range(0, len(array), batch):
        tokens = torch.from_numpy(array[start:start+batch].astype(np.int64, copy=False)).to(device)
        inputs = tokens[:, :-1]; targets = tokens[:, 1:]
        logits, _ = model(inputs); route = model.last_router_output; assert route is not None
        mask = (inputs >= CONTENT_OFFSET) & (targets >= CONTENT_OFFSET)
        nll = nn.functional.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1), reduction="none").view_as(targets)
        tasks = model.resolve_task_ids(inputs)
        dense = dense_dispatch(route)
        valid_routes = (route.indices >= 0) & (route.gates > 0)
        active_sum += int(valid_routes.sum()); token_count += route.indices.shape[0] * route.indices.shape[1]
        dispatch += dense.sum((0,1)).cpu().double().numpy()
        for task in range(N_TASKS):
            chosen = mask & (tasks == task); n = int(chosen.sum())
            if n:
                loss_sums[task] += float(nll[chosen].sum()); counts[task] += n
                profiles[task] += dense[chosen].sum(0).cpu().double().numpy(); profile_counts[task] += n
    task_loss = np.divide(loss_sums, counts, out=np.full(N_TASKS, np.nan), where=counts>0)
    loads = dispatch / max(dispatch.sum(), 1e-12)
    metrics = {
        "L": float(loss_sums.sum() / counts.sum()), "L_per_task": task_loss.tolist(),
        "C_route": active_sum / token_count,
        "C_exp": int((dispatch > 0).sum()),
        "load_balance_cv": float(loads.std() / loads.mean()),
        "dispatch_fraction": loads.tolist(),
    }
    return metrics, profiles / np.maximum(profile_counts[:, None], 1)


def dominant_sets(profiles: np.ndarray) -> list[list[int]]:
    result = []
    for profile in profiles:
        order = np.argsort(-profile); cumulative = 0.0; selected = []
        for expert in order:
            selected.append(int(expert)); cumulative += float(profile[expert])
            if cumulative >= .5: break
        result.append(selected)
    return result


@torch.no_grad()
def switching_time(model: nn.Module, switching: np.ndarray, sets: list[list[int]], device: torch.device, batch: int) -> dict[str, Any]:
    model.eval(); times: list[int] = []; failures = 0
    for start in range(0, len(switching), batch):
        tokens = torch.from_numpy(switching[start:start+batch].astype(np.int64, copy=False)).to(device)
        model(tokens[:, :-1]); route = model.last_router_output; assert route is not None
        dense = dense_dispatch(route)
        for row in range(tokens.shape[0]):
            markers = torch.nonzero(tokens[row] < CONTENT_OFFSET).flatten()
            switch = int(markers[1]); new_task = int(tokens[row, switch]); experts = sets[new_task]
            values = dense[row, switch:, experts].sum(-1)
            crossing = torch.nonzero(values > .5).flatten()
            if crossing.numel(): times.append(int(crossing[0]))
            else: failures += 1
    return {
        "T_sw": float(np.mean(times)) if times else None,
        "T_sw_median": float(np.median(times)) if times else None,
        "T_sw_success_fraction": len(times) / (len(times) + failures),
        "definition": "first token lag from new marker with >0.5 dispatch mass on new-task heldout dominant profile",
    }


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--data-dir", type=Path, default=Path("data")); p.add_argument("--checkpoint-dir", type=Path, default=Path("results/exploratory"))
    p.add_argument("--output-dir", type=Path, default=Path("results/exploratory/harness")); p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", choices=("auto","cpu","cuda"), default="auto"); args=p.parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 1)); seed_all(args.seed)
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else "cpu" if args.device=="auto" else args.device)
    heldout=np.load(args.data_dir/"heldout.npy", allow_pickle=False); switching=np.load(args.data_dir/"switching.npy", allow_pickle=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs={"lr":("lr",args.checkpoint_dir/f"t1_lr_seed{args.seed}.pt"), "hr1":("hr",args.checkpoint_dir/f"t2_hr1_hr1_seed{args.seed}.pt"),
           "hr2":("hr2",args.checkpoint_dir/f"t2_hr2_hr2_seed{args.seed}.pt"), "pg":("pg",args.checkpoint_dir/f"t3_phase_gate_pg_seed{args.seed}.pt")}
    summary={}
    for arm,(factory,checkpoint) in specs.items():
        model=make_model(factory).to(device); payload=torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(payload["model_state_dict"])
        std=hidden_std(model, heldout, device, args.batch_size)
        model.configure_eval_disturbance(); clean, profiles=evaluate(model, heldout, device, args.batch_size)
        clean["robustness"]=1.0; clean["delta_L_drop"]=0.0; clean["hidden_std_clean"]=std
        sets=dominant_sets(profiles); sw=switching_time(model, switching, sets, device, args.batch_size); clean.update(sw)
        switching_metrics,_=evaluate(model, switching, device, args.batch_size)
        switching_metrics.update(sw)
        switching_metrics.update({"robustness":None,"delta_L_drop":None,"hidden_std_clean":std})
        cells: dict[str,dict[str,Any]]={"clean":clean, "switching_clean":switching_metrics}
        generator=torch.Generator(device="cpu").manual_seed(args.seed + 101)
        direction=torch.randn(128, generator=generator); direction=direction/direction.square().mean().sqrt()
        for units in (.5,1.0):
            name=f"noise_{units:g}std"; seed_all(args.seed+int(units*1000)); model.configure_eval_disturbance(noise_std=units*std)
            m,_=evaluate(model,heldout,device,args.batch_size); m.update({"robustness":math.exp(-(m["L"]-clean["L"])),"delta_L_drop":None,"T_sw":None,"hidden_std_clean":std,"disturbance_units":units}); cells[name]=m
            name=f"shift_{units:g}std"; model.configure_eval_disturbance(shift=direction.to(device)*units*std)
            m,_=evaluate(model,heldout,device,args.batch_size); m.update({"robustness":math.exp(-(m["L"]-clean["L"])),"delta_L_drop":None,"T_sw":None,"hidden_std_clean":std,"disturbance_units":units}); cells[name]=m
        drops=[]
        for expert in range(N_TASKS):
            model.configure_eval_disturbance(dropped_experts=(expert,)); m,_=evaluate(model,heldout,device,args.batch_size)
            delta=m["L"]-clean["L"]; drops.append(delta); m.update({"robustness":math.exp(-delta),"delta_L_drop":delta,"T_sw":None,"hidden_std_clean":std,"dropped_expert":expert}); cells[f"drop_e{expert}"]=m
        model.configure_eval_disturbance()
        for cell,m in cells.items():
            artifact={"arm":arm,"seed":args.seed,"cell":cell,"metrics":m,"dominant_profiles":sets if cell=="clean" else None,
                      "definitions":{"C_route":"mean positive expert dispatches per routed token","C_exp":"experts with nonzero aggregate dispatch (of 8)","robustness":"exp(-(L_cell-L_clean))","load_balance_cv":"population std/mean of aggregate expert dispatch mass"}}
            (args.output_dir/f"{arm}_seed{args.seed}_{cell}.json").write_text(json.dumps(artifact,indent=2,sort_keys=True)+"\n")
        summary[arm]={"clean":clean,"mean_delta_L_drop":float(np.mean(drops)),"max_delta_L_drop":float(np.max(drops)),"cells":len(cells)}
        print(f"arm={arm} L={clean['L']:.6f} C_route={clean['C_route']:.3f} CV={clean['load_balance_cv']:.3f} T_sw={clean['T_sw']} mean_drop={np.mean(drops):.6f}")
    (args.output_dir/f"summary_seed{args.seed}.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")


if __name__=="__main__": main()
