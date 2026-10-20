# Training

`train_policy.sh {act|diffusion|smolvla|pi05}` wraps `lerobot-train` (LeRobot 0.4.4) with the presets
used for every reported checkpoint; `--print-command` shows the full command. Defaults:

| Preset | Steps | Batch | Notes |
| --- | --- | --- | --- |
| `act` | 60k | 8 | chunk 60 (2 s at 30 Hz); evaluated with temporal ensembling (`--override temporal_ensemble_coeff=0.01 --override n_action_steps=1` on the policy server) |
| `diffusion` | 60k | 32 | horizon 32, 16 actions per chunk in training, DDIM 10 steps, 432×576 random crop; evaluated with `n_action_steps=31` |
| `smolvla` | 20k | 64 | from `lerobot/smolvla_base`, wrist/scene → `camera1`/`camera2` |
| `pi05` | 20k | 32 | full fine-tune of `lerobot/pi05_base` (revision `9e55186`), bf16, gradient checkpointing; scene → `base_0_rgb`, wrist → `left_wrist_0_rgb`; needs `pi05_env_patch.sh` and the gated PaliGemma tokenizer |

Cluster runs use the Slurm templates. They contain no site values; pass them on the command line:

```bash
export REBOT_CLUSTER_ROOT=<work dir> LEROBOT_ENV_ACTIVATE=<file that activates the env>
sbatch --account=<ACCOUNT> --partition=<GPU_PARTITION> --output="$REBOT_CLUSTER_ROOT/logs/%x-%j.out" \
    "$REBOT_CLUSTER_ROOT/rebot-il-vla-benchmark/training/train_smolvla.sbatch"
```

Expected layout: `<work dir>/rebot-il-vla-benchmark/` (this repository), `datasets/rebot_t{1,2}_v1/`,
`runs/`, `cache/huggingface/` (compute nodes run with `HF_HUB_OFFLINE=1`, so download the base weights
first). `DATASET_ROOT` and `OUTPUT_DIR` override the defaults; `train_pi05.sbatch` requires a fixed
`OUTPUT_DIR` and resumes from it when a job is chained or requeued.
