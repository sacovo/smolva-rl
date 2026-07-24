#!/bin/bash
# Wrapper script to train the RECAP policy on Slurm.
#
# train_recap.py consumes PRE-COMPUTED advantages — it does not run the critic.
# Generate them first with scripts/compute_advantages.sh (writes
# task_advantages_<repo>.npy / task_thresholds_<repo>.json into the save dir),
# or pass --expert_mode to skip advantage conditioning entirely.
#
# Usage: ./scripts/submit_recap.sh <dataset_repo_id> [--partition <partition>] [--num-gpus <n>] [additional_args...]

DATASET=$1

if [ -z "$DATASET" ]; then
    echo "Usage: ./scripts/submit_recap.sh <dataset_repo_id> [--partition <partition>] [--num-gpus <n>] [additional_args...]"
    echo "Note: run scripts/compute_advantages.sh first (unless using --expert_mode)."
    exit 1
fi

shift 1

# Defaults
PARTITION=""
NUM_GPUS=4

# Parse optional flags before collecting remaining args
REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition)
            PARTITION="$2"; shift 2 ;;
        --num-gpus)
            NUM_GPUS="$2"; shift 2 ;;
        *)
            REMAINING_ARGS+=("$1"); shift ;;
    esac
done
EXTRA_ARGS="${REMAINING_ARGS[@]}"

# Extract clean dataset name (e.g. lerobot/pusht -> pusht)
DATASET_NAME=$(basename "$DATASET")

# Detect mode based on whether we are loading pretrained policy weights
MODE="pretrain"
if [[ "$EXTRA_ARGS" =~ "--pretrained_policy_path" ]]; then
    MODE="finetune"
fi

# Define a highly descriptive WandB run name
JOB_NAME="recap_${DATASET_NAME}_${MODE}_$(date +%m%d_%H%M%S)"

echo "Submitting Slurm job with WandB run name: $JOB_NAME"

SBATCH_ARGS=("--gres=gpu:${NUM_GPUS}")
[ -n "$PARTITION" ] && SBATCH_ARGS+=("--partition=${PARTITION}")

sbatch "${SBATCH_ARGS[@]}" scripts/train_slurm.sh \
    src/lerobot_policy_smolvla_rl/train_recap.py \
    --dataset_repo_id "$DATASET" \
    --job_name "$JOB_NAME" \
    "$EXTRA_ARGS"
