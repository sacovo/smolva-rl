# Slurm Training Setup for smolvla-rl

This project supports parallel training on a Slurm cluster with automatic checkpointing and resuming.

## Scripts

- `scripts/train_slurm.sh`: A generic Slurm submission script that uses `accelerate` for multi-GPU training.
- `scripts/submit_critic.sh`: A convenience wrapper for training the critic.

## How to use

### Training the Critic

To submit a training job for the critic:

```bash
./scripts/submit_critic.sh <dataset_repo_id>
```

Example:
```bash
./scripts/submit_critic.sh fhnw/rover_test --steps 5000 --batch_size 8
```

### Resuming Training

The scripts are designed to automatically resume from the latest checkpoint if the job is interrupted (e.g., due to the 24h limit). Simply submit the same command again, and it will pick up from where it left off.

This is achieved via the `--resume_from auto` flag, which looks for `state_*.pt` directories in the output directory.

### Training Other Models (e.g., Policy)

To train a different model using the same Slurm setup:

```bash
sbatch scripts/train_slurm.sh path/to/your/train_script.py --arg1 val1 --arg2 val2
```

The `train_slurm.sh` script will:
1. Detect the number of available GPUs.
2. Launch the training using `accelerate launch`.
3. Enable `bf16` mixed precision (ideal for SmolVLM on modern GPUs).
4. Automatically append `--resume_from auto` to your arguments.

## Requirements

- `uv` must be installed on the cluster.
- The training script should support `--resume_from` and handle the step counter correctly (similar to `src/lerobot_policy_smolvla_rl/train_critic.py`).
