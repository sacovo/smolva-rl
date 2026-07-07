import os
import json
import sys

# Add root directory to path to find scripts folder
sys.path.append(os.getcwd())

from scripts.submit_utils import (
    parse_unknown_args,
    serialize_args,
    load_and_merge_config,
    save_resolved_config,
    generate_sbatch_script,
)

def test_parse_and_serialize_unknown_args():
    args_list = [
        "positional_value",
        "--dataset_repo_id", "lerobot/pusht",
        "--steps", "20000",
        "--batch_size", "16",
        "--accumulation_steps", "4",
        "--pretrained_critic_path", "/some/path/critic.pt",
        "--some_flag"
    ]
    
    parsed = parse_unknown_args(args_list)
    assert parsed["dataset_repo_id"] == "lerobot/pusht"
    assert parsed["steps"] == 20000
    assert parsed["batch_size"] == 16
    assert parsed["accumulation_steps"] == 4
    assert parsed["pretrained_critic_path"] == "/some/path/critic.pt"
    assert parsed["some_flag"] is True
    assert parsed["pos_0"] == "positional_value"
    
    # Serialize back
    serialized = serialize_args(parsed)
    # Checks that all original flags are present
    assert "--dataset_repo_id" in serialized
    assert "lerobot/pusht" in serialized
    assert "--steps" in serialized
    assert "--some_flag" in serialized
    assert "positional_value" in serialized

def test_load_and_merge_config_nested(tmp_path):
    config_data = {
        "slurm": {
            "mem": "32G",
            "nodes": 1,
            "gres": "gpu:2"
        },
        "training": {
            "dataset_repo_id": "lerobot/pusht",
            "steps": 10000
        }
    }
    
    config_file = tmp_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f)
        
    cli_slurm = {
        "mem": "64G"  # override
    }
    cli_training = {
        "steps": 20000,  # override
        "batch_size": 16
    }
    
    slurm_config, training_config = load_and_merge_config(
        str(config_file), cli_slurm, cli_training
    )
    
    assert slurm_config["mem"] == "64G"
    assert slurm_config["gres"] == "gpu:2"
    assert slurm_config["nodes"] == 1
    
    assert training_config["steps"] == 20000
    assert training_config["dataset_repo_id"] == "lerobot/pusht"
    assert training_config["batch_size"] == 16

def test_load_and_merge_config_flat(tmp_path):
    config_data = {
        "mem": "32G",
        "gres": "gpu:2",
        "dataset_repo_id": "lerobot/pusht",
        "steps": 10000
    }
    
    config_file = tmp_path / "config_flat.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f)
        
    slurm_config, training_config = load_and_merge_config(
        str(config_file), {}, {}
    )
    
    assert slurm_config["mem"] == "32G"
    assert slurm_config["gres"] == "gpu:2"
    assert training_config["dataset_repo_id"] == "lerobot/pusht"
    assert training_config["steps"] == 10000

def test_save_resolved_config(tmp_path):
    # Change working directory temporarily to keep things tidy
    orig_cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        slurm_config = {"nodes": 1, "gres": "gpu:4"}
        training_config = {"steps": 1000}
        
        path = save_resolved_config("test_job", slurm_config, training_config)
        assert os.path.exists(path)
        assert "runs/configuration" in path
        
        with open(path, "r") as f:
            saved = json.load(f)
        assert saved["slurm"]["nodes"] == 1
        assert saved["training"]["steps"] == 1000
    finally:
        os.chdir(orig_cwd)

def test_generate_sbatch_script():
    slurm_config = {
        "job_name": "test-critic",
        "nodes": 1,
        "gres": "gpu:2",
        "time": "12:00:00",
        "mem": "32G",
        "output": "logs/%x.out",
        "error": "logs/%x.err"
    }
    training_args_list = ["--dataset_repo_id", "lerobot/pusht", "--steps", "50"]
    
    script = generate_sbatch_script(
        "src/lerobot_policy_smolvla_rl/train_critic.py",
        slurm_config,
        training_args_list
    )
    
    assert "#SBATCH --job-name=test-critic" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --gres=gpu:2" in script
    assert "#SBATCH --time=12:00:00" in script
    assert "#SBATCH --mem=32G" in script
    assert "#SBATCH --output=logs/%x.out" in script
    assert "#SBATCH --error=logs/%x.err" in script
    
    assert "src/lerobot_policy_smolvla_rl/train_critic.py" in script
    assert "--dataset_repo_id lerobot/pusht" in script
    assert "--steps 50" in script
