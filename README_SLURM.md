# Slurm Training Setup for smolvla-rl

This project supports parallel training on a Slurm cluster with automatic checkpointing and resuming, using reproducible Python-based submit scripts.

## Scripts

- `scripts/submit_critic.py`: Python submission script for training the critic.
- `scripts/submit_recap.py`: Python submission script for training the RECAP policy.
- `scripts/submit_utils.py`: Shared helper module containing Slurm logic, configuration merging, and script submission.
- `scripts/compute_advantages.sh`: A Slurm submission script to pre-compute advantages and thresholds on a single GPU.

---

## Reproducible Python Submission Setup

The Python submit scripts allow configuring both Slurm and training parameters entirely via CLI arguments or by passing a JSON/YAML configuration file. 

To ensure reproducibility and ease of restarting, the scripts merge the configs (CLI parameters override file configurations), dynamically generate the SBATCH script, and store the final resolved configuration file in `runs/configuration/` with timestamps.

### Slurm Configuration CLI Arguments
The submission scripts accept the following Slurm configuration parameters:
* `--nodes`: Number of nodes (default: `1`)
* `--ntasks-per-node`: Number of tasks per node (default: `1`)
* `--cpus-per-task`: CPUs per task (default: `16`)
* `--gres`: GPUs request (default: `gpu:4`)
* `--time`: Time limit (default: `24:00:00`)
* `--mem`: Memory limit (default: `64G`)
* `--job-name`: Custom job name (defaults to auto-generated descriptive name)
* `--output`: Output log path (default: `logs/%x_%j.out`)
* `--error`: Error log path (default: `logs/%x_%j.err`)
* `--num-jobs`: Number of times to submit the job sequentially with dependencies (default: `1`)
* `--dependency-type`: Slurm dependency type, e.g. `afterany`, `afterok`, `after` (default: `afterany`)
* `--config`: Path to config file (JSON or YAML)
* `--dry-run`: Dry run to print the generated sbatch script without submitting it to Slurm

Any arguments not listed above are treated as training arguments and passed directly to the training script.

### Configuration File Layout
You can specify both Slurm configuration and training parameters in a single YAML or JSON file:

**JSON Config Example (`config.json`):**
```json
{
  "slurm": {
    "mem": "128G",
    "gres": "gpu:8"
  },
  "training": {
    "dataset_repo_id": "lerobot/droid_100",
    "steps": 20000,
    "batch_size": 16,
    "accumulation_steps": 16
  }
}
```

---

## How to use

### 1. Training the Critic

Submit the critic training job via the Python script:
```bash
python scripts/submit_critic.py --dataset_repo_id lerobot/droid_100 --steps 20000 --batch_size 16 --accumulation_steps 16
```

Using a configuration file:
```bash
python scripts/submit_critic.py --config config.json
```

Overriding a configuration file parameter via CLI (CLI always takes precedence):
```bash
python scripts/submit_critic.py --config config.json --mem 64G --steps 50000
```

> [!TIP]
> **Memory Optimization**: Since 3-camera datasets like `lerobot/droid_100` have a high memory footprint, always use smaller batch sizes per GPU coupled with gradient accumulation. We recommend:
> - Critic: `--batch_size 16 --accumulation_steps 16` (effective batch size 256 per process) or `--batch_size 8 --accumulation_steps 32` for lower-end GPU memory.

### 2. Training the RECAP Policy

Once the Critic is trained, you can submit the RECAP policy training job. We support **two different execution modes** for the policy training:

#### Mode A: High-Speed Offline Advantage Mode (RECOMMENDED 🚀)
Since the Critic is frozen during policy training, the expected returns $V(s)$ and temporal advantages $A_t$ for all frames in the dataset are completely static. 

By pre-computing all advantages once on a fast GPU (like your local **RTX 3090** with local SSD I/O speed) and saving them to disk, you can **completely skip loading the Critic on the cluster GPUs** during policy training!

##### Step A1: Pre-compute Advantages Directly on the Cluster
To submit the precomputation job:
```bash
sbatch scripts/compute_advantages.sh lerobot/droid_100 outputs/checkpoints_critic/critic/checkpoint_final.pt \
    --cameras observation.images.exterior_image_1_left observation.images.wrist_image_left \
    --batch_size 32 \
    --num_workers 4 \
    --save_dir outputs/recap_phase1
```
This single-GPU job runs very quickly (usually under 3 minutes) and generates:
1. `outputs/recap_phase1/task_thresholds_lerobot_droid_100.json` (Thresholds)
2. `outputs/recap_phase1/task_advantages_lerobot_droid_100.npy` (Binary advantages array)

##### Step A2: Launch the Policy Job on SLURM
Submit your RECAP policy job. The script will automatically detect the pre-computed files, load them in a fraction of a millisecond, **bypass loading the Critic entirely**, and start training at maximum speed:
```bash
python scripts/submit_recap.py --dataset_repo_id lerobot/droid_100 --critic_checkpoint outputs/checkpoints_critic/critic/checkpoint_final.pt --steps 20000 --batch_size 8 --accumulation_steps 8 --wandb_project smolvla-recap
```

Alternatively, you can submit using a configuration file:
```bash
python scripts/submit_recap.py --config recap_config.json
```

---

#### Mode B: Standard On-the-Fly Mode
If you prefer not to pre-compute offline, you can compute advantages on the fly:
```bash
python scripts/submit_recap.py --dataset_repo_id lerobot/droid_100 --critic_checkpoint outputs/checkpoints_critic/critic/checkpoint_final.pt --steps 20000 --batch_size 4 --accumulation_steps 16 --wandb_project smolvla-recap
```

---

### Resuming Training

The scripts are designed to automatically resume from the latest checkpoint if the job is interrupted (e.g., due to the 24h limit). Simply submit the same command again, and the training script will automatically look for the latest checkpoint using the `--resume_from auto` option, which is automatically appended if not present.

### Sequential Multi-Job Submission (Chaining)

To automate sequential training across multiple time windows on a Slurm cluster, you can use the `--num-jobs` and `--dependency-type` arguments. This will submit the job multiple times, setting up dependencies between each run so they execute one after the other.

For example, to kick off 5 sequential training windows to reach a target of 100k steps (where each run will resume from where the previous one timed out or finished):
```bash
python scripts/submit_recap.py \
    --dataset_repo_id lerobot/droid_100 \
    --critic_checkpoint outputs/checkpoints_critic/critic/checkpoint_final.pt \
    --steps 100000 \
    --batch_size 8 \
    --accumulation_steps 8 \
    --time 04:00:00 \
    --num-jobs 5 \
    --dependency-type afterany
```

- `--num-jobs 5`: Submits 5 jobs to Slurm. Job 2 runs only after Job 1 completes/terminates, Job 3 after Job 2, etc.
- `--dependency-type afterany`: By default, `afterany` is used so the next job is executed regardless of how the previous job ended (useful because Slurm timeouts count as a termination but not success). If you only want subsequent jobs to run if the previous one succeeded (exited code 0), specify `afterok`.
- Checked-in and resumed configurations automatically load the latest saved checkpoint using the `--resume_from auto` argument appended to the accelerating launching command.
- If a chained job launches but the previous runs already successfully completed the target number of steps (e.g., reached 100k steps in Job 3 of 5), the subsequent jobs will load the final state, detect that they have already reached the target steps, and exit immediately and cleanly.

