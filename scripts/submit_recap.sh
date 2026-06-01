#!/bin/bash
# Wrapper script to train the RECAP policy on Slurm
# Usage: ./scripts/submit_recap.sh <dataset_repo_id> <critic_checkpoint> [additional_args...]

DATASET=$1
CRITIC_CHECKPOINT=$2

if [ -z "$DATASET" ] || [ -z "$CRITIC_CHECKPOINT" ]; then
    echo "Usage: ./scripts/submit_recap.sh <dataset_repo_id> <critic_checkpoint> [additional_args...]"
    exit 1
fi

shift 2
EXTRA_ARGS=$@

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

sbatch scripts/train_slurm.sh \
    src/lerobot_policy_smolvla_rl/train_recap.py \
    --dataset_repo_id "$DATASET" \
    --critic_checkpoint "$CRITIC_CHECKPOINT" \
    --job_name "$JOB_NAME" \
    $EXTRA_ARGS
