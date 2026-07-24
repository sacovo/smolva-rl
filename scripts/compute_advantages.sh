#!/bin/bash
# Script to compute advantages on Slurm
# Usage: ./scripts/compute_advantages.sh <dataset_repo_id> <critic_checkpoint_path> [--partition <partition>] [--num-gpus <n>] [extra_args...]
#
#SBATCH --job-name=compute-adv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---------------------------------------------------------------------------
# Self-resubmission wrapper: when invoked directly (not inside a Slurm job),
# parse --partition / --num-gpus and resubmit via sbatch.
# ---------------------------------------------------------------------------
if [ -z "$SLURM_JOB_ID" ]; then
    DATASET=$1
    CRITIC_CHECKPOINT=$2

    if [ -z "$DATASET" ] || [ -z "$CRITIC_CHECKPOINT" ]; then
        echo "Usage: ./scripts/compute_advantages.sh <dataset_repo_id> <critic_checkpoint_path> [--partition <partition>] [--num-gpus <n>] [extra_args...]"
        exit 1
    fi

    shift 2

    # Defaults
    PARTITION=""
    NUM_GPUS=1

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

    SBATCH_ARGS=("--gres=gpu:${NUM_GPUS}")
    [ -n "$PARTITION" ] && SBATCH_ARGS+=("--partition=${PARTITION}")

    mkdir -p logs
    echo "Submitting Slurm job: dataset=$DATASET critic=$CRITIC_CHECKPOINT gpus=$NUM_GPUS${PARTITION:+ partition=$PARTITION}"
    exec sbatch "${SBATCH_ARGS[@]}" "$0" "$DATASET" "$CRITIC_CHECKPOINT" "${REMAINING_ARGS[@]}"
fi

# ---------------------------------------------------------------------------
# Actual job logic (runs inside Slurm)
# ---------------------------------------------------------------------------

# Create logs directory
mkdir -p logs

# Ensure uv is in PATH
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8

DATASET=$1
CRITIC_CHECKPOINT=$2
shift 2
EXTRA_ARGS=$@

if [ -z "$DATASET" ] || [ -z "$CRITIC_CHECKPOINT" ]; then
    echo "Usage: sbatch scripts/compute_advantages.sh <dataset_repo_id> <critic_checkpoint_path> [extra_args...]"
    exit 1
fi

echo "Job started on $(hostname) at $(date)"
echo "Computing advantages for $DATASET using $CRITIC_CHECKPOINT"
echo "Extra arguments: $EXTRA_ARGS"

uv run python src/lerobot_policy_smolvla_rl/compute_thresholds.py \
    --dataset_repo_id "$DATASET" \
    --critic_checkpoint "$CRITIC_CHECKPOINT" \
    "$EXTRA_ARGS"

echo "Job finished at $(date)"
