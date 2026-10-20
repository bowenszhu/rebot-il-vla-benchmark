#!/usr/bin/env python3
"""Open-loop check of a trained checkpoint on recorded frames (offline; no robot I/O).

Loads `<run>/checkpoints/<step>/pretrained_model` with its saved pre/post-processors, replays one
episode's observations through `select_action` in order, and compares the predicted action with the
recorded mirror action. Reports per-joint MAE (rad) and `select_action` latency; calls that ran a new
chunk inference are the slow tail (p95/max).

  conda run -n lerobot-blackwell python -B rebot_bench/offline_infer.py \
      --checkpoint <run>/checkpoints/last/pretrained_model \
      --dataset-root data/rebot_t1_v1 --episode 0 --frames 300
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="…/checkpoints/<step>/pretrained_model")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", type=int, default=300, help="0 = whole episode")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-action-steps", type=int, help="override the saved n_action_steps")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = args.device
    if args.n_action_steps is not None:
        cfg.n_action_steps = args.n_action_steps
    policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg)
    policy.to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    root = args.dataset_root.resolve()
    dataset = LeRobotDataset(f"local/{root.name}", root=root, episodes=[args.episode])
    n = len(dataset) if args.frames <= 0 else min(args.frames, len(dataset))
    obs_keys = [k for k in dataset.meta.features if k.startswith("observation.")]

    policy.reset()
    predicted, recorded, latency = [], [], []
    for i in range(n):
        item = dataset[i]
        batch = {k: item[k].unsqueeze(0).to(args.device) for k in obs_keys}
        batch["task"] = item["task"]
        batch["robot_type"] = dataset.meta.robot_type or ""
        with torch.inference_mode():
            batch = preprocessor(batch)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            action = policy.select_action(batch)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            latency.append(time.perf_counter() - start)
            action = postprocessor(action)
        predicted.append(action.squeeze(0).float().cpu().numpy())
        recorded.append(item["action"].numpy())

    predicted_arr, recorded_arr = np.stack(predicted), np.stack(recorded)
    if not np.isfinite(predicted_arr).all():
        raise SystemExit("non-finite predicted action")
    names = dataset.meta.features["action"]["names"]
    mae = np.abs(predicted_arr - recorded_arr).mean(axis=0)
    lat_ms = np.array(latency) * 1e3
    # Chunk inference runs whenever the action queue is empty: call 0 (also CUDA warm-up, excluded)
    # and every n_action_steps calls after it.
    step = int(getattr(cfg, "n_action_steps", 1) or 1)
    chunk_calls = np.zeros(n, dtype=bool)
    chunk_calls[::step] = True
    chunk_ms = lat_ms[1:][chunk_calls[1:]]
    report = {
        "checkpoint": str(checkpoint),
        "policy": cfg.type,
        "n_action_steps": getattr(cfg, "n_action_steps", None),
        "dataset": str(root),
        "episode": args.episode,
        "frames": n,
        "mae_rad": {name: round(float(v), 4) for name, v in zip(names, mae, strict=True)},
        "mae_rad_mean": round(float(mae.mean()), 4),
        "latency_ms": {
            "p50": round(float(np.percentile(lat_ms, 50)), 2),
            "p95": round(float(np.percentile(lat_ms, 95)), 2),
            "max": round(float(lat_ms.max()), 2),
            "first_call_warmup": round(float(lat_ms[0]), 2),
            "chunk_inference_mean": round(float(chunk_ms.mean()), 2) if chunk_ms.size else None,
            "chunk_inference_calls": int(chunk_ms.size),
        },
    }
    if args.device.startswith("cuda"):
        report["cuda_max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
