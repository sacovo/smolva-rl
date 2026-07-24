#!/bin/bash
#SBATCH --job-name=recap_ev_aug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --mem=24G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=p4500
#SBATCH --array=0-19

# Copy of the file running on the cluster as scripts/sub_aug_eval.sh (job
# 179462). Array bound must equal len(CHECKPOINTS) x len(SUITES) x
# len(CFG_SCALES) - 1 from the rat_*.py it runs.

# Create logs and outputs directories
mkdir -p logs
mkdir -p outputs/eval

# Set up Conda environment paths
# Activate the conda12 env (provides EGL libs + libero/robosuite; PATH-only is not enough)
source /opt/conda/etc/profile.d/conda.sh
conda activate /home2/sandro.covo/conda12
export MUJOCO_GL=egl
unset LEROBOT_HOME
export HF_LEROBOT_HOME="$HOME/.cache/huggingface/lerobot"

export PYTHONUNBUFFERED=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

echo "Evaluation sweep array job started at $(date) on $(hostname)"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_ARRAY_JOB_ID"

# Run the single task from the array
PYTHONPATH=src python scripts/rat_aug.py

echo "Evaluation sweep array job finished at $(date)"
