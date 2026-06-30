#!/bin/bash
#SBATCH --job-name=recap_eval_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=h200
#SBATCH --array=0-47

# Create logs and outputs directories
mkdir -p logs
mkdir -p outputs/eval

# Set up Conda environment paths
export CONDA_PREFIX="/home2/sandro.covo/conda12"
export PATH="$CONDA_PREFIX/bin:$PATH"

export PYTHONUNBUFFERED=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

echo "Evaluation sweep array job started at $(date) on $(hostname)"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_ARRAY_JOB_ID"

# Run the single task from the array
PYTHONPATH=src python scripts/run_array_task.py

echo "Evaluation sweep array job finished at $(date)"
