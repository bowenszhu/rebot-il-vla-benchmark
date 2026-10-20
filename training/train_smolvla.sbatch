#!/bin/bash
# SmolVLA fine-tune from lerobot/smolvla_base on the reBot T1 dataset (template; site values come from the
# sbatch command line). Usage and placeholders: training/README.md.
# The base weights must already be in $HF_HOME (compute nodes run with HF_HUB_OFFLINE=1).
#SBATCH --job-name=rebot-t1-smolvla
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
set -euo pipefail
: "${REBOT_CLUSTER_ROOT:?export REBOT_CLUSTER_ROOT=<CLUSTER_ROOT> before sbatch}"
: "${LEROBOT_ENV_ACTIVATE:?export LEROBOT_ENV_ACTIVATE=<file that activates the lerobot 0.4.4 env>}"
DATASET_ROOT=${DATASET_ROOT:-$REBOT_CLUSTER_ROOT/datasets/rebot_t1_v1}
OUTPUT_DIR=${OUTPUT_DIR:-$REBOT_CLUSTER_ROOT/runs/smolvla_$(basename "$DATASET_ROOT")_${SLURM_JOB_ID:-manual}}

set +u
# shellcheck disable=SC1090
source "$LEROBOT_ENV_ACTIVATE"
set -u
export HF_HOME=${HF_HOME:-$REBOT_CLUSTER_ROOT/cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export LEROBOT_TRAIN=lerobot-train
nvidia-smi --query-gpu=name,memory.total --format=csv || true

# train_policy.sh adds --policy.path=lerobot/smolvla_base and the wrist/scene -> camera1/camera2 rename_map.
bash "$REBOT_CLUSTER_ROOT/rebot-il-vla-benchmark/training/train_policy.sh" smolvla \
    --dataset-root "$DATASET_ROOT" --output-dir "$OUTPUT_DIR" \
    --num-workers "$(( ${SLURM_CPUS_PER_TASK:-16} - 2 ))" "$@"
