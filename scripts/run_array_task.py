#!/usr/bin/env python3
import os
import sys
import json
import subprocess
from pathlib import Path

# Parameter combinations (3 x 4 x 4 = 48 total)
CHECKPOINTS = [
    ("150k", "outputs/libero_recap_150000"),
    ("250k", "outputs/libero_recap_250000"),
    ("350k (final)", "outputs/libero/recap_model/migrated")
]

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
CFG_SCALES = [0.0, 1.0, 1.5, 2.0]
N_EPISODES = 5  # 5 episodes per task * 10 tasks = 50 episodes total

def find_eval_info(job_name):
    eval_dir = Path("outputs/eval")
    if not eval_dir.exists():
        return None
    for path in eval_dir.glob(f"**/[0-9]*-[0-9]*-[0-9]*_{job_name}/eval_info.json"):
        if path.exists():
            return path
    return None

def log_to_wandb(job_name, checkpoint_name, suite, cfg, success_rate, eval_info_path):
    try:
        import wandb
        
        per_task_logs = {}
        if eval_info_path and os.path.exists(eval_info_path):
            with open(eval_info_path, 'r') as f:
                data = json.load(f)
                for task_entry in data.get("per_task", []):
                    task_id = task_entry.get("task_id")
                    task_success = task_entry.get("metrics", {}).get("successes", [False])
                    task_pc_success = sum(task_success) / len(task_success) * 100.0
                    per_task_logs[f"task_{task_id}_success"] = task_pc_success

        run = wandb.init(
            project="smolvla-recap-eval",
            name=job_name,
            config={
                "checkpoint": checkpoint_name,
                "suite": suite,
                "cfg_weight": cfg,
                "n_episodes": N_EPISODES,
            },
            reinit=True
        )
        
        logs = {"overall_success_rate": success_rate}
        logs.update(per_task_logs)
        wandb.log(logs)
        run.finish()
        print(f"--> Logged metrics to WandB for job {job_name}")
    except Exception as e:
        print(f"Warning: Failed to log to WandB: {e}")

def main():
    # Programmatically set LEROBOT_HOME if not already configured
    if "LEROBOT_HOME" not in os.environ:
        os.environ["LEROBOT_HOME"] = str(Path.home() / ".cache" / "huggingface" / "lerobot")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id_str is None:
        print("Error: SLURM_ARRAY_TASK_ID environment variable not found. Run with e.g. export SLURM_ARRAY_TASK_ID=0")
        sys.exit(1)
        
    try:
        task_id = int(task_id_str)
    except ValueError:
        print(f"Error: Invalid SLURM_ARRAY_TASK_ID: {task_id_str}")
        sys.exit(1)
        
    combinations = []
    for cp_name, cp_path in CHECKPOINTS:
        for suite in SUITES:
            for cfg in CFG_SCALES:
                combinations.append((cp_name, cp_path, suite, cfg))
                
    if task_id < 0 or task_id >= len(combinations):
        print(f"Error: Task ID {task_id} out of bounds (0-{len(combinations)-1})")
        sys.exit(1)
        
    checkpoint_name, checkpoint_path, suite, cfg = combinations[task_id]
    job_name = f"recap_{checkpoint_name}_{suite}_cfg_{cfg}".replace(" ", "_").replace("(", "").replace(")", "")
    
    print(f"Task ID: {task_id}")
    print(f"Checkpoint: {checkpoint_name} ({checkpoint_path})")
    print(f"Suite: {suite}")
    print(f"CFG: {cfg}")
    print(f"Job name: {job_name}")
    print("="*60)
    
    existing_info = find_eval_info(job_name)
    if existing_info:
        try:
            with open(existing_info, 'r') as f:
                data = json.load(f)
                success_rate = data.get("overall", {}).get("pc_success", 0.0)
                print(f"Found existing result for {job_name}: {success_rate}% success")
                log_to_wandb(job_name, checkpoint_name, suite, cfg, success_rate, existing_info)
                return
        except Exception as e:
            print(f"Error reading existing result {existing_info}: {e}, re-running...")
            
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint path {checkpoint_path} does not exist. Make sure you run the export script first!")
        sys.exit(1)

    cmd = [
        "PYTHONPATH=src", "lerobot-eval",
        f"--policy.path={checkpoint_path}",
        "--env.type=libero",
        f"--env.task={suite}",
        "--eval.batch_size=1",
        f"--eval.n_episodes={N_EPISODES}",
        "--env.max_parallel_tasks=1",
        "--policy.device=cuda",
        f"--policy.cfg_weight={cfg}",
        f"--job_name={job_name}"
    ]
    
    print(f"Launching subprocess: {' '.join(cmd)}")
    try:
        subprocess.run(" ".join(cmd), shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running subprocess for {job_name}: {e}")
        sys.exit(1)
        
    new_info = find_eval_info(job_name)
    if new_info and new_info.exists():
        try:
            with open(new_info, 'r') as f:
                data = json.load(f)
                success_rate = data.get("overall", {}).get("pc_success", 0.0)
                print(f"Execution complete. Overall success: {success_rate}%")
                log_to_wandb(job_name, checkpoint_name, suite, cfg, success_rate, new_info)
        except Exception as e:
            print(f"Error parsing final results: {e}")
            sys.exit(1)
    else:
        print(f"Error: Could not find generated eval_info.json for job {job_name}")
        sys.exit(1)

if __name__ == "__main__":
    main()
