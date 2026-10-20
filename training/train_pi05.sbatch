#!/bin/bash
# pi0.5 fine-tune from pi05_base (revision 9e55186) on a reBot dataset (template; site values come from the
# sbatch command line). Usage and placeholders: training/README.md.
# Resumable: OUTPUT_DIR is fixed per run, a later job (chained with
# --dependency=afterany:<prev> on a 6 h lane, or requeued) resumes from OUTPUT_DIR/checkpoints/last.
# LEROBOT_ENV_ACTIVATE must activate the patched env (pi05_env_patch.sh). $HF_HOME must already hold the
# pi05_base snapshot and the PaliGemma tokenizer (compute nodes run with HF_HUB_OFFLINE=1).
#   sbatch ... --export=ALL,REBOT_CLUSTER_ROOT=$P,LEROBOT_ENV_ACTIVATE=$P/activate_env_pi.sh,\
#       DATASET_ROOT=$P/datasets/rebot_t2_v1,OUTPUT_DIR=$P/runs/<name> train_pi05.sbatch [train_policy.sh args]
#SBATCH --job-name=rebot-pi05
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
set -euo pipefail
: "${REBOT_CLUSTER_ROOT:?export REBOT_CLUSTER_ROOT=<CLUSTER_ROOT> before sbatch}"
: "${LEROBOT_ENV_ACTIVATE:?export LEROBOT_ENV_ACTIVATE=<file that activates the pi05 env>}"
: "${OUTPUT_DIR:?export OUTPUT_DIR=<fixed run dir>; it must stay the same across chained/requeued jobs}"
DATASET_ROOT=${DATASET_ROOT:-$REBOT_CLUSTER_ROOT/datasets/rebot_t2_v1}

set +u
# shellcheck disable=SC1090
source "$LEROBOT_ENV_ACTIVATE"
set -u
export HF_HOME=${HF_HOME:-$REBOT_CLUSTER_ROOT/cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
# The module form runs under the patched env's python; a lerobot-train script may point at the base env.
export LEROBOT_TRAIN=${LEROBOT_TRAIN:-python -m lerobot.scripts.lerobot_train}
echo "job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} on $(hostname) at $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -c "import transformers; print('transformers', transformers.__version__)"
# GPU memory/utilisation once a minute next to the run, to size batches on other GPUs.
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 60 \
    >> "${OUTPUT_DIR%/}.gpu_${SLURM_JOB_ID:-manual}.csv" 2>/dev/null &

RESUME_CONFIG="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json"
if [[ -f "$RESUME_CONFIG" ]]; then
    echo "resuming from $(readlink -f "$OUTPUT_DIR/checkpoints/last")"
    read -r -a LAUNCHER <<< "$LEROBOT_TRAIN"
    "${LAUNCHER[@]}" "--config_path=$RESUME_CONFIG" --resume=true
    exit
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    # Killed before its first checkpoint: lerobot-train refuses an existing dir, keep it aside for inspection.
    mv -- "$OUTPUT_DIR" "$OUTPUT_DIR.incomplete_$(date +%Y%m%d_%H%M%S)"
fi
# train_policy.sh adds --policy.path=<pi05_base snapshot>, bf16, gradient checkpointing and the camera rename_map.
bash "$REBOT_CLUSTER_ROOT/rebot-il-vla-benchmark/training/train_policy.sh" pi05 \
    --dataset-root "$DATASET_ROOT" --output-dir "$OUTPUT_DIR" \
    --num-workers "$(( ${SLURM_CPUS_PER_TASK:-16} - 2 ))" "$@"
