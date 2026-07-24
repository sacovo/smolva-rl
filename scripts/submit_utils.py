import os
import json
import datetime
import subprocess
import argparse

def parse_unknown_args(args_list):
    """
    Parses CLI training arguments list into a dictionary.
    E.g. ['--steps', '20000', '--batch_size', '16', '--flag']
    -> {'steps': 20000, 'batch_size': 16, 'flag': True}
    """
    parsed = {}
    i = 0
    while i < len(args_list):
        arg = args_list[i]
        if arg.startswith("-"):
            # strip all leading dashes
            key = arg.lstrip("-")
            # Check if there is a value next and it is not another flag
            if i + 1 < len(args_list) and not args_list[i + 1].startswith("-"):
                val = args_list[i + 1]
                # Cast to numeric if possible
                try:
                    if "." in val:
                        val = float(val)
                    else:
                        val = int(val)
                except ValueError:
                    pass
                parsed[key] = val
                i += 2
            else:
                parsed[key] = True
                i += 1
        else:
            # Positional arguments or list elements
            parsed[f"pos_{i}"] = arg
            i += 1
    return parsed


def serialize_args(args_dict):
    """
    Converts a dictionary of arguments back into a list of CLI flags.
    """
    serialized = []
    # Sort keys to maintain deterministic ordering (optional but good for reproducibility)
    for k in sorted(args_dict.keys()):
        v = args_dict[k]
        if k.startswith("pos_"):
            serialized.append(str(v))
        elif v is True:
            serialized.append(f"--{k}")
        elif v is False or v is None:
            continue
        else:
            serialized.append(f"--{k}")
            serialized.append(str(v))
    return serialized

def load_and_merge_config(config_path, cli_slurm, cli_training_dict):
    """
    Loads JSON or YAML configuration if specified, and merges it with CLI args.
    CLI arguments override config file values.
    Returns (merged_slurm_dict, merged_training_dict).
    """
    file_slurm = {}
    file_training = {}

    if config_path and os.path.exists(config_path):
        try:
            if config_path.endswith((".yaml", ".yml")):
                import yaml
                with open(config_path, "r") as f:
                    file_config = yaml.safe_load(f) or {}
            else:
                with open(config_path, "r") as f:
                    file_config = json.load(f) or {}

            # Handle both nested and flat layouts
            if "slurm" in file_config or "training" in file_config:
                file_slurm = file_config.get("slurm", {})
                file_training = file_config.get("training", {})
            else:
                # Flat layout: separate slurm keys
                slurm_keys = {
                    "nodes", "ntasks_per_node", "ntasks-per-node",
                    "cpus_per_task", "cpus-per-task", "gres", "time", "mem",
                    "job_name", "job-name", "output", "error", "partition",
                    "num_jobs", "num-jobs", "dependency_type", "dependency-type"
                }
                for k, v in file_config.items():
                    # normalize key dashes to underscores
                    normalized_k = k.replace("-", "_")
                    if k in slurm_keys or normalized_k in slurm_keys:
                        file_slurm[normalized_k] = v
                    else:
                        file_training[k] = v
        except Exception as e:
            print(f"Warning: Failed to load config file {config_path}: {e}")

    # Merge Slurm options (CLI overrides file)
    merged_slurm = {}
    all_slurm_keys = set(file_slurm.keys()) | set(cli_slurm.keys())
    for k in all_slurm_keys:
        cli_val = cli_slurm.get(k)
        file_val = file_slurm.get(k)
        if cli_val is not None:
            merged_slurm[k] = cli_val
        else:
            merged_slurm[k] = file_val

    # Merge Training options (CLI overrides file)
    merged_training = {}
    all_training_keys = set(file_training.keys()) | set(cli_training_dict.keys())
    for k in all_training_keys:
        cli_val = cli_training_dict.get(k)
        file_val = file_training.get(k)
        if cli_val is not None:
            merged_training[k] = cli_val
        else:
            merged_training[k] = file_val

    return merged_slurm, merged_training

def save_resolved_config(job_name, slurm_config, training_config):
    """
    Saves the final configuration into runs/configuration/config_<job_name>_<timestamp>.json.
    """
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    output_dir = "runs/configuration"
    os.makedirs(output_dir, exist_ok=True)
    
    resolved_config = {
        "slurm": slurm_config,
        "training": training_config
    }
    
    filename = f"config_{job_name}_{timestamp}.json"
    file_path = os.path.join(output_dir, filename)
    with open(file_path, "w") as f:
        json.dump(resolved_config, f, indent=2)
    print(f"Configuration saved to {file_path}")
    return file_path

def generate_sbatch_script(train_script, slurm_config, training_args_list):
    """
    Constructs the sbatch script content dynamically.
    """
    # Normalize slurm keys
    nodes = slurm_config.get("nodes", 1)
    ntasks_per_node = slurm_config.get("ntasks_per_node") or slurm_config.get("ntasks-per-node", 1)
    cpus_per_task = slurm_config.get("cpus_per_task") or slurm_config.get("cpus-per-task", 16)
    gres = slurm_config.get("gres", "gpu:4")
    time = slurm_config.get("time", "24:00:00")
    mem = slurm_config.get("mem", "64G")
    job_name = slurm_config.get("job_name") or slurm_config.get("job-name", "smolvla-rl")
    output_log = slurm_config.get("output", "logs/%x_%j.out")
    error_log = slurm_config.get("error", "logs/%x_%j.err")
    partition = slurm_config.get("partition")

    # Construct args string
    args_str = " ".join(training_args_list)

    partition_line = f"#SBATCH --partition={partition}\n" if partition else ""

    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gres={gres}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --output={output_log}
#SBATCH --error={error_log}
{partition_line}
mkdir -p logs

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONFAULTHANDLER=1
# Debugging flags (disabled for performance)
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=INFO
# export NCCL_SHM_DISABLE=1
# export NCCL_CUMEM_HOST_ENABLE=0

# Detect number of GPUs assigned by Slurm
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NUM_GPUS=$(echo "$SLURM_GPUS_ON_NODE" | grep -o '[0-9]\\+$')
else
    NUM_GPUS=$(nvidia-smi -L | wc -l)
fi

# Bypass nvshare for multi-GPU DistributedDataParallel (DDP) runs
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Multi-GPU run detected ($NUM_GPUS GPUs). Unsetting LD_PRELOAD."
    unset LD_PRELOAD
fi

echo "Job started on $(hostname) at $(date)"
echo "Using $NUM_GPUS GPUs"
echo "Training script: {train_script}"
echo "Training args: {args_str}"

# Automatically add --resume_from auto if not already present
if [[ ! "{args_str}" =~ "--resume_from" ]]; then
    echo "Adding --resume_from auto to arguments"
    EXTRA_LAUNCH_ARGS="--resume_from auto"
else
    EXTRA_LAUNCH_ARGS=""
fi

LAUNCH_ARGS=""
if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCH_ARGS="--multi_gpu"
fi

uv run accelerate launch \\
    $LAUNCH_ARGS \\
    --num_machines 1 \\
    --num_processes $NUM_GPUS \\
    --mixed_precision bf16 \\
    {train_script} {args_str} $EXTRA_LAUNCH_ARGS

echo "Job finished at $(date)"
"""
    return sbatch_content

def build_submit_arg_parser(description):
    """
    Builds the argparse.ArgumentParser shared by all submit_*.py entrypoints
    (Slurm resource flags + --config/--dry-run). Training-specific options are
    parsed separately via parse_unknown_args on the leftover argv.
    """
    parser = argparse.ArgumentParser(description=description)

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
    slurm_group.add_argument("--partition", type=str, default=None)
    slurm_group.add_argument("--num-jobs", type=int, default=None, help="Number of times to submit the job sequentially with dependencies")
    slurm_group.add_argument("--dependency-type", type=str, default=None, choices=["afterany", "afterok", "after", "afternotok"], help="Slurm dependency type (default: afterany)")

    parser.add_argument("--config", type=str, default=None, help="Path to config JSON/YAML file")
    parser.add_argument("--dry-run", action="store_true", help="Print SBATCH script without submitting")
    return parser


def run_submission(description, slurm_defaults, train_script, job_name_fn):
    """
    Shared submit_*.py driver: parse CLI args, merge with --config, apply
    per-script Slurm defaults, resolve a job name, save the resolved config,
    and generate + submit the sbatch script.

    slurm_defaults: dict of Slurm config defaults (nodes, gres, mem, etc.)
        specific to this training script's resource needs.
    train_script: path to the training entrypoint (e.g.
        "src/lerobot_policy_smolvla_rl/train_recap.py").
    job_name_fn: callable(training_config, dataset_name) -> str, returning
        the job name (e.g. mode-dependent naming for critic/recap, a fixed
        scheme for snapflow).
    """
    parser = build_submit_arg_parser(description)
    parsed_args, unknown_args = parser.parse_known_args()

    cli_slurm = {
        k: v for k, v in vars(parsed_args).items()
        if k not in ["config", "dry_run"] and v is not None
    }
    cli_training = parse_unknown_args(unknown_args)

    slurm_config, training_config = load_and_merge_config(
        parsed_args.config, cli_slurm, cli_training
    )

    for key, value in slurm_defaults.items():
        slurm_config.setdefault(key, value)

    # Automatically propagate Slurm time limit to training script duration limit
    slurm_time = slurm_config.get("time")
    if slurm_time and "duration" not in training_config:
        training_config["duration"] = slurm_time

    dataset = training_config.get("dataset_repo_id")
    dataset_name = os.path.basename(dataset) if dataset else "unknown_dataset"
    slurm_config.setdefault("job_name", job_name_fn(training_config, dataset_name))

    save_resolved_config(slurm_config["job_name"], slurm_config, training_config)

    training_args_list = serialize_args(training_config)
    sbatch_script = generate_sbatch_script(train_script, slurm_config, training_args_list)

    submit_sbatch(
        sbatch_script,
        num_jobs=slurm_config["num_jobs"],
        dependency_type=slurm_config["dependency_type"],
        dry_run=parsed_args.dry_run,
    )


def extract_job_id(sbatch_output):
    """
    Extracts the numeric job ID from the sbatch command output.
    Typically: "Submitted batch job 123456" -> "123456"
    """
    if not sbatch_output:
        return None
    parts = sbatch_output.strip().split()
    if parts:
        return parts[-1]
    return None

def submit_sbatch(sbatch_content, num_jobs=1, dependency_type="afterany", dry_run=False):
    """
    Submits the dynamic sbatch script to Slurm.
    Supports submitting multiple jobs sequentially with dependencies.
    """
    dependency_job_id = None
    
    for i in range(num_jobs):
        if num_jobs > 1:
            print(f"\n--- Submitting job {i+1}/{num_jobs} ---")
            
        dependency = f"{dependency_type}:{dependency_job_id}" if dependency_job_id else None
        
        # In dry_run or fallback scenarios, mock the dependency ID for display/logging
        if dry_run and i > 0 and not dependency_job_id:
            dependency = f"{dependency_type}:dry_run_job_id_{i}"
            
        if dry_run:
            print("--- DRY RUN: Generated SBATCH Script ---")
            if dependency:
                print(f"# (Submitted with dependency: --dependency={dependency})")
            print(sbatch_content)
            print("----------------------------------------")
            submit_output = f"Submitted batch job dry_run_job_id_{i+1}"
        else:
            try:
                cmd = ["sbatch"]
                if dependency:
                    cmd.append(f"--dependency={dependency}")
                
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate(input=sbatch_content)
                if process.returncode != 0:
                    raise RuntimeError(f"sbatch submission failed: {stderr}")
                
                submit_output = stdout.strip()
                print(submit_output)
            except FileNotFoundError:
                print("Warning: sbatch command not found. Is Slurm installed?")
                if dependency:
                    print(f"# (Submitted with dependency: --dependency={dependency})")
                print("Writing script to stdout instead:")
                print(sbatch_content)
                submit_output = f"Submitted batch job mock_job_id_{i+1}"
                
        dependency_job_id = extract_job_id(submit_output)
        if not dependency_job_id and num_jobs > 1:
            print("Warning: Could not extract job ID. Subsequent jobs will not have a dependency.")
