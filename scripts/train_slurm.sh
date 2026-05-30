#!/bin/bash
#SBATCH --job-name=smolvla-rl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Generic Slurm training script for smolvla-rl
# Usage: sbatch scripts/train_slurm.sh <train_script_path> [args...]
# Example: sbatch scripts/train_slurm.sh src/lerobot_policy_smolvla_rl/train_critic.py --dataset_repo_id fhnw/rover_test

# Create logs directory
mkdir -p logs

# Ensure uv is in PATH (adjust if necessary for your cluster)
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8

if [ -z "$1" ]; then
    echo "Usage: sbatch scripts/train_slurm.sh <train_script_path> [args...]"
    exit 1
fi

TRAIN_SCRIPT=$1
shift
TRAIN_ARGS=$@

# Detect number of GPUs assigned by Slurm
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NUM_GPUS=$SLURM_GPUS_ON_NODE
else
    NUM_GPUS=$(nvidia-smi -L | wc -l)
fi

echo "Job started on $(hostname) at $(date)"
echo "Using $NUM_GPUS GPUs"
echo "Training script: $TRAIN_SCRIPT"
echo "Training args: $TRAIN_ARGS"

# Add --resume_from auto if not already present to support 24h limit
if [[ ! "$TRAIN_ARGS" =~ "--resume_from" ]]; then
    echo "Adding --resume_from auto to arguments"
    TRAIN_ARGS="$TRAIN_ARGS --resume_from auto"
fi

# Run with accelerate launch
# We use --multi_gpu and detect the number of processes.
# SmolVLM typically benefits from bf16 on modern GPUs (A100/H100/4090).
# If your GPUs are older, you might want to switch to fp16.
uv run accelerate launch \
    --multi_gpu \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    $TRAIN_SCRIPT $TRAIN_ARGS

echo "Job finished at $(date)"
