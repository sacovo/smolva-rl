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

sbatch scripts/train_slurm.sh \
    src/lerobot_policy_smolvla_rl/train_critic.py \
    --dataset_repo_id "$DATASET" \
    --job_name "train_critic_$(date +%Y%m%d_%H%M%S)" \
    $EXTRA_ARGS
