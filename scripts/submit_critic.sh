#!/bin/bash
# Wrapper script to train the critic on Slurm
# Usage: ./scripts/submit_critic.sh <dataset_repo_id> [additional_args...]

DATASET=$1
shift
EXTRA_ARGS=$@

if [ -z "$DATASET" ]; then
    echo "Usage: ./scripts/submit_critic.sh <dataset_repo_id> [additional_args...]"
    exit 1
fi

# Extract clean dataset name (e.g. lerobot/pusht -> pusht)
DATASET_NAME=$(basename "$DATASET")

# Detect mode based on whether we are loading pretrained critic weights
MODE="scratch"
if [[ "$EXTRA_ARGS" =~ "--pretrained_critic_path" ]]; then
    MODE="finetune"
fi

# Define a highly descriptive WandB run name
JOB_NAME="critic_${DATASET_NAME}_${MODE}_$(date +%m%d_%H%M%S)"

echo "Submitting Slurm job with WandB run name: $JOB_NAME"

sbatch scripts/train_slurm.sh \
    src/lerobot_policy_smolvla_rl/train_critic.py \
    --dataset_repo_id "$DATASET" \
    --job_name "$JOB_NAME" \
    $EXTRA_ARGS
