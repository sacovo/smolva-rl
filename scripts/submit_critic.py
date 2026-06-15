#!/usr/bin/env python3
import sys
import os
import argparse
from datetime import datetime

# Add directory to python path if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.submit_utils import (
    parse_unknown_args,
    serialize_args,
    load_and_merge_config,
    save_resolved_config,
    generate_sbatch_script,
    submit_sbatch,
)

def main():
    parser = argparse.ArgumentParser(description="Submit Critic Training to Slurm")
    
    # Slurm Arguments
    slurm_group = parser.add_argument_group("Slurm Configuration")
    slurm_group.add_argument("--nodes", type=int, default=None)
    slurm_group.add_argument("--ntasks-per-node", type=int, default=None)
    slurm_group.add_argument("--cpus-per-task", type=int, default=None)
    slurm_group.add_argument("--gres", type=str, default=None)
    slurm_group.add_argument("--time", type=str, default=None)
    slurm_group.add_argument("--mem", type=str, default=None)
    slurm_group.add_argument("--job-name", type=str, default=None)
    slurm_group.add_argument("--output", type=str, default=None)
    slurm_group.add_argument("--error", type=str, default=None)
    
    # Submission arguments
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON/YAML file")
    parser.add_argument("--dry-run", action="store_true", help="Print SBATCH script without submitting")
    
    # Parse Slurm and submission args, leaving the rest
    parsed_args, unknown_args = parser.parse_known_args()
    
    # Extract clean Slurm dict
    cli_slurm = {
        k: v for k, v in vars(parsed_args).items() 
        if k not in ["config", "dry_run"] and v is not None
    }
    
    # Parse unknown args (training options)
    cli_training = parse_unknown_args(unknown_args)
    
    # Load and merge with config file
    slurm_config, training_config = load_and_merge_config(
        parsed_args.config,
        cli_slurm,
        cli_training
    )
    
    # Apply defaults if still not set
    slurm_config.setdefault("nodes", 1)
    slurm_config.setdefault("ntasks_per_node", 1)
    slurm_config.setdefault("cpus_per_task", 16)
    slurm_config.setdefault("gres", "gpu:4")
    slurm_config.setdefault("time", "24:00:00")
    slurm_config.setdefault("mem", "64G")
    slurm_config.setdefault("output", "logs/%x_%j.out")
    slurm_config.setdefault("error", "logs/%x_%j.err")
    
    # Resolve job name
    dataset = training_config.get("dataset_repo_id")
    dataset_name = os.path.basename(dataset) if dataset else "unknown_dataset"
    
    mode = "scratch"
    if "pretrained_critic_path" in training_config:
        mode = "finetune"
        
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    default_job_name = f"critic_{dataset_name}_{mode}_{timestamp}"
    slurm_config.setdefault("job_name", default_job_name)
    
    # Save the resolved configuration
    save_resolved_config(slurm_config["job_name"], slurm_config, training_config)
    
    # Generate and submit sbatch
    training_args_list = serialize_args(training_config)
    sbatch_script = generate_sbatch_script(
        "src/lerobot_policy_smolvla_rl/train_critic.py",
        slurm_config,
        training_args_list
    )
    
    submit_sbatch(sbatch_script, dry_run=parsed_args.dry_run)

if __name__ == "__main__":
    main()
