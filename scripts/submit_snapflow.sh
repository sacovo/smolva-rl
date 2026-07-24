#!/bin/bash
# Wrapper script to train the SnapFlow policy on Slurm
# Usage: ./scripts/submit_snapflow.sh <dataset_repo_id> <recap_checkpoint> [--partition <partition>] [--num-gpus <n>] [additional_args...]

DATASET=$1
RECAP_CHECKPOINT=$2

if [ -z "$DATASET" ] || [ -z "$RECAP_CHECKPOINT" ]; then
    echo "Usage: ./scripts/submit_snapflow.sh <dataset_repo_id> <recap_checkpoint> [--partition <partition>] [--num-gpus <n>] [additional_args...]"
    exit 1
fi

shift 2

# Defaults
PARTITION="h200"
NUM_GPUS=1

source "$(dirname "${BASH_SOURCE[0]}")/submit_common.sh"
parse_submit_args "$@"

# Extract clean dataset name (e.g. lerobot/pusht -> pusht)
DATASET_NAME=$(basename "$DATASET")

# Define a highly descriptive WandB run name
JOB_NAME="snapflow_${DATASET_NAME}_$(date +%m%d_%H%M%S)"

echo "Submitting Slurm job with WandB run name: $JOB_NAME"

SBATCH_ARGS=("--gres=gpu:${NUM_GPUS}")
[ -n "$PARTITION" ] && SBATCH_ARGS+=("--partition=${PARTITION}")

sbatch "${SBATCH_ARGS[@]}" scripts/train_slurm.sh \
    src/lerobot_policy_smolvla_rl/train_snapflow.py \
    --dataset_repo_id "$DATASET" \
    --recap_checkpoint "$RECAP_CHECKPOINT" \
    --job_name "$JOB_NAME" \
    "$EXTRA_ARGS"
