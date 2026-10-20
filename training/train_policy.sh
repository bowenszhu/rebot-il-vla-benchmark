#!/usr/bin/env bash
# Train ACT / Diffusion / SmolVLA / pi0.5 on a reBot T1/T2 LeRobot dataset with lerobot-train (lerobot 0.4.4).
# Offline only: reads the dataset, writes checkpoints; never touches CAN, serial ports, or cameras.
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: bash training/train_policy.sh {act|diffusion|smolvla|pi05} [options] [-- extra lerobot-train args]

Options (defaults depend on the policy, see training/README.md):
  --dataset-root PATH   LeRobot v3 dataset dir (default: <repo>/data/rebot_t1_v1)
  --output-dir PATH     must not exist yet (default: <repo>/runs/<policy>_<dataset>_<time>)
  --steps N  --batch-size N  --save-freq N  --log-freq N  --num-workers N
  --device DEVICE       cuda (default) or cpu
  --print-command       print the lerobot-train command and exit

Environment:
  LEROBOT_TRAIN         launcher command (default: conda run --no-capture-output -n <env> lerobot-train, env
                        lerobot-blackwell, or lerobot-pi for pi05; the sbatch templates set it to lerobot-train
                        inside the activated cluster env)
  PI05_BASE             pi05_base snapshot dir (default: revision 9e55186 in $HF_HOME/hub)
EOF
    exit 2
}

[[ $# -ge 1 ]] || usage
POLICY=$1
shift
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONDA_ENV=lerobot-blackwell

case "$POLICY" in
    act)
        # 30 Hz: chunk 60 = 2 s, the horizon used in the ACT paper (100 steps at 50 Hz).
        STEPS=60000 BATCH=8 SAVE=10000 WORKERS=4
        POLICY_ARGS=(--policy.type=act --policy.chunk_size=60 --policy.n_action_steps=60)
        ;;
    diffusion)
        # horizon 32 (~1.07 s), execute 16 per chunk (~0.53 s at 30 Hz) so a DDIM-10 chunk (~0.2 s on the
        # local GPU) is under half of the executed span; drop_n_last_frames = horizon - n_action_steps - n_obs_steps + 1.
        # 640x480 has no resize step in lerobot 0.4.4: random 90 % crop (432x576) instead of the 84x84 default.
        STEPS=60000 BATCH=32 SAVE=10000 WORKERS=8
        POLICY_ARGS=(--policy.type=diffusion --policy.horizon=32 --policy.n_action_steps=16
            --policy.drop_n_last_frames=15 "--policy.crop_shape=[432,576]"
            --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10)
        ;;
    smolvla)
        # Fine-tune from the pretrained base; its camera keys are camera1..3.
        STEPS=20000 BATCH=64 SAVE=5000 WORKERS=8
        POLICY_ARGS=(--policy.path=lerobot/smolvla_base
            '--rename_map={"observation.images.wrist": "observation.images.camera1", "observation.images.scene": "observation.images.camera2"}')
        ;;
    pi05)
        # Full fine-tune of pi05_base in bf16 with gradient checkpointing. The base is pinned to revision 9e55186:
        # later ones add a relative_actions_processor step that lerobot 0.4.4 cannot load. Needs the
        # transformers fix/lerobot_openpi branch (pi05_env_patch.sh) and the gated PaliGemma tokenizer in $HF_HOME.
        # Camera slots as in openpi single-arm setups: scene -> base_0_rgb, wrist -> left_wrist_0_rgb;
        # right_wrist_0_rgb is never fed and stays a masked blank image.
        STEPS=20000 BATCH=32 SAVE=2500 WORKERS=8
        CONDA_ENV=lerobot-pi
        PI05_BASE=${PI05_BASE:-${HF_HOME:-$HOME/.cache/huggingface}/hub/models--lerobot--pi05_base/snapshots/9e55186ad36e66b95cda57bc47818d9e6237ae30}
        POLICY_ARGS=("--policy.path=$PI05_BASE" --policy.dtype=bfloat16 --policy.gradient_checkpointing=true
            '--rename_map={"observation.images.wrist": "observation.images.left_wrist_0_rgb", "observation.images.scene": "observation.images.base_0_rgb"}')
        ;;
    -h|--help) usage ;;
    *) echo "Unknown policy: $POLICY" >&2; usage ;;
esac

DATASET_ROOT="$REPO_ROOT/data/rebot_t1_v1"
OUTPUT_DIR=""
LOG_FREQ=100
DEVICE=cuda
PRINT_ONLY=0
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-root) DATASET_ROOT=$2; shift 2 ;;
        --output-dir) OUTPUT_DIR=$2; shift 2 ;;
        --steps) STEPS=$2; shift 2 ;;
        --batch-size) BATCH=$2; shift 2 ;;
        --save-freq) SAVE=$2; shift 2 ;;
        --log-freq) LOG_FREQ=$2; shift 2 ;;
        --num-workers) WORKERS=$2; shift 2 ;;
        --device) DEVICE=$2; shift 2 ;;
        --print-command) PRINT_ONLY=1; shift ;;
        --) shift; EXTRA=("$@"); break ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done
for value in "$STEPS" "$BATCH" "$SAVE" "$LOG_FREQ" "$WORKERS"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo "Expected a non-negative integer, got: $value" >&2; exit 2; }
done

DATASET_ROOT=$(realpath -m -- "$DATASET_ROOT")
DATASET_NAME=$(basename -- "$DATASET_ROOT")
JOB_NAME="${POLICY}_${DATASET_NAME}"
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO_ROOT/runs/${JOB_NAME}_$(date +%Y%m%d_%H%M%S)"
fi
OUTPUT_DIR=$(realpath -m -- "$OUTPUT_DIR")

if [[ $PRINT_ONLY -eq 0 ]]; then
    [[ -f "$DATASET_ROOT/meta/info.json" ]] || { echo "Not a LeRobot dataset: $DATASET_ROOT" >&2; exit 1; }
    [[ ! -e "$OUTPUT_DIR" ]] || { echo "Output dir exists (lerobot-train refuses to overwrite): $OUTPUT_DIR" >&2; exit 1; }
    mkdir -p -- "$(dirname -- "$OUTPUT_DIR")"
fi

read -r -a LAUNCHER <<< "${LEROBOT_TRAIN:-conda run --no-capture-output -n $CONDA_ENV lerobot-train}"
CMD=("${LAUNCHER[@]}"
    "--dataset.repo_id=local/$DATASET_NAME" "--dataset.root=$DATASET_ROOT"
    "${POLICY_ARGS[@]}"
    "--policy.device=$DEVICE" --policy.push_to_hub=false
    "--output_dir=$OUTPUT_DIR" "--job_name=$JOB_NAME"
    "--steps=$STEPS" "--batch_size=$BATCH" "--save_freq=$SAVE" "--log_freq=$LOG_FREQ"
    "--num_workers=$WORKERS" --wandb.enable=false
    "${EXTRA[@]}")

printf '%q ' "${CMD[@]}"
echo
[[ $PRINT_ONLY -eq 1 ]] && exit 0
exec "${CMD[@]}"
