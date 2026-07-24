#!/usr/bin/env python3
import sys
import os
from datetime import datetime

# Add directory to python path if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.submit_utils import run_submission


def job_name(training_config, dataset_name):
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    return f"snapflow_{dataset_name}_{timestamp}"


def main():
    run_submission(
        description="Submit SnapFlow Distillation Training to Slurm",
        slurm_defaults={
            "nodes": 1,
            "ntasks_per_node": 1,
            "cpus_per_task": 16,
            "gres": "gpu:1",
            "time": "24:00:00",
            "mem": "128G",
            "output": "logs/%x_%j.out",
            "error": "logs/%x_%j.err",
            "num_jobs": 1,
            "dependency_type": "afterany",
            "partition": "h200",
        },
        train_script="src/lerobot_policy_smolvla_rl/train_snapflow.py",
        job_name_fn=job_name,
    )


if __name__ == "__main__":
    main()
