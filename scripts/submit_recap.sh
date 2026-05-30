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

sbatch scripts/train_slurm.sh \
    src/lerobot_policy_smolvla_rl/train_recap.py \
    --dataset_repo_id "$DATASET" \
    --critic_checkpoint "$CRITIC_CHECKPOINT" \
    --job_name "train_recap_$(date +%Y%m%d_%H%M%S)" \
    $EXTRA_ARGS
