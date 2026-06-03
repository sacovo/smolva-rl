#!/bin/bash
# Wrapper script to train the RECAP policy on Slurm
# Usage: ./scripts/submit_recap.sh <dataset_repo_id> <critic_checkpoint> [--partition <partition>] [--num-gpus <n>] [additional_args...]

DATASET=$1
CRITIC_CHECKPOINT=$2

if [ -z "$DATASET" ] || [ -z "$CRITIC_CHECKPOINT" ]; then
    echo "Usage: ./scripts/submit_recap.sh <dataset_repo_id> <critic_checkpoint> [--partition <partition>] [--num-gpus <n>] [additional_args...]"
    exit 1
fi

shift 2

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
    --critic_checkpoint "$CRITIC_CHECKPOINT" \
    --job_name "$JOB_NAME" \
    "$EXTRA_ARGS"
