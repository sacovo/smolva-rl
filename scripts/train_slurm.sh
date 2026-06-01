#!/bin/bash
#SBATCH --job-name=smolvla-rl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=64G
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

# Resolve NCCL Peer-to-Peer, shared memory, and InfiniBand communication failures on cluster nodes
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO

if [ -z "$1" ]; then
    echo "Usage: sbatch scripts/train_slurm.sh <train_script_path> [args...]"
    exit 1
fi

TRAIN_SCRIPT=$1
shift
TRAIN_ARGS=$@

# Detect number of GPUs assigned by Slurm (robustly extract integer count)
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    # Extract only the trailing number (e.g., gpu:1 -> 1, gpu:rtx2080:4 -> 4)
    NUM_GPUS=$(echo "$SLURM_GPUS_ON_NODE" | grep -o '[0-9]\+$')
else
    NUM_GPUS=$(nvidia-smi -L | wc -l)
fi

# Bypass nvshare for multi-GPU DistributedDataParallel (DDP) runs to prevent DDP/NCCL conflicts
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Multi-GPU run detected ($NUM_GPUS GPUs). Unsetting LD_PRELOAD to bypass nvshare and enable native DDP."
    unset LD_PRELOAD
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
