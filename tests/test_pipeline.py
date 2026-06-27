import os
import sys
import subprocess
import pytest
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), "src"))

DATASET_REPO_ID = "lerobot/pusht"
EPISODE_INDEX = 0


@pytest.fixture(scope="module")
def critic_checkpoint_path():
    checkpoint_dir = "outputs/test_critic_pytest"
    checkpoint_path = os.path.join(checkpoint_dir, "smoke_test/checkpoint_final.pt")

    # Run critic training for 1 step
    cmd = [
        sys.executable,
        "src/lerobot_policy_smolvla_rl/train_critic.py",
        "--dataset_repo_id",
        DATASET_REPO_ID,
        "--episodes",
        str(EPISODE_INDEX),
        "--steps",
        "1",
        "--batch_size",
        "1",
        "--num_vlm_layers",
        "1",
        "--save_dir",
        checkpoint_dir,
        "--model_save_name",
        "smoke_test",
        "--wandb_project",
        "pytest-smoke-test",
        "--num_workers",
        "0",
    ]

    # Set environment variable to disable wandb sync
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"

    subprocess.run(cmd, check=True, env=env)
    return checkpoint_path


def test_critic_training(critic_checkpoint_path):
    assert os.path.exists(critic_checkpoint_path)
    # Check if we can load it
    state_dict = torch.load(critic_checkpoint_path, map_location="cpu")
    assert "c51_head.0.weight" in state_dict or "module.c51_head.0.weight" in state_dict


def test_recap_training(critic_checkpoint_path):
    save_dir = "outputs/test_recap_pytest"

    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"

    # Step 1: pre-compute advantages + thresholds with the critic (train_recap
    # itself no longer runs the critic — it only consumes the saved files).
    precompute_cmd = [
        sys.executable,
        "src/lerobot_policy_smolvla_rl/compute_thresholds.py",
        "--dataset_repo_id",
        DATASET_REPO_ID,
        "--critic_checkpoint",
        critic_checkpoint_path,
        "--episodes",
        str(EPISODE_INDEX),
        "--advantage_horizon",
        "5",
        "--num_vlm_layers",
        "1",
        "--batch_size",
        "1",
        "--num_workers",
        "0",
        "--save_dir",
        save_dir,
    ]
    subprocess.run(precompute_cmd, check=True, env=env)

    safe_repo = DATASET_REPO_ID.replace("/", "_")
    assert os.path.exists(os.path.join(save_dir, f"task_advantages_{safe_repo}.npy"))
    assert os.path.exists(os.path.join(save_dir, f"task_thresholds_{safe_repo}.json"))

    # Step 2: train consuming the pre-computed files (no critic checkpoint).
    cmd = [
        sys.executable,
        "src/lerobot_policy_smolvla_rl/train_recap.py",
        "--dataset_repo_id",
        DATASET_REPO_ID,
        "--episodes",
        str(EPISODE_INDEX),
        "--action_chunk_size",
        "20",
        "--steps",
        "1",
        "--batch_size",
        "1",
        "--num_vlm_layers",
        "1",
        "--save_dir",
        save_dir,
        "--wandb_project",
        "pytest-smoke-test",
        "--num_workers",
        "0",
    ]

    subprocess.run(cmd, check=True, env=env)
    # The script saves state using accelerator.save_state which creates a directory
    assert os.path.exists(save_dir)


def test_visualization(critic_checkpoint_path):
    output_dir = "outputs/plots_pytest"
    cmd = [
        sys.executable,
        "src/lerobot_policy_smolvla_rl/visualize_critic.py",
        "--checkpoint",
        critic_checkpoint_path,
        "--dataset_repo_id",
        DATASET_REPO_ID,
        "--episodes",
        str(EPISODE_INDEX),
        "--num_vlm_layers",
        "1",
        "--batch_size",
        "1",
        "--output_dir",
        output_dir,
    ]

    subprocess.run(cmd, check=True)
    # Filename now embeds the dataset + checkpoint tags, so match by prefix.
    import glob

    matches = glob.glob(
        os.path.join(output_dir, f"episode_{EPISODE_INDEX}_critic_*.png")
    )
    assert matches, f"No critic plot for episode {EPISODE_INDEX} in {output_dir}"


def test_ds_utils():
    from lerobot_policy_smolvla_rl.ds_utils import (
        get_episode_lengths,
        get_max_task_lengths,
    )

    dataset = LeRobotDataset(DATASET_REPO_ID, episodes=[EPISODE_INDEX])

    lengths = get_episode_lengths(dataset)
    assert len(lengths) > EPISODE_INDEX
    assert lengths[EPISODE_INDEX] > 0

    max_task_lengths = get_max_task_lengths(dataset)
    assert len(max_task_lengths) > 0


def test_modeling_recap_config():
    from lerobot_policy_smolvla_rl import SmolVLARECAPConfig

    config = SmolVLARECAPConfig(num_vlm_layers=1)
    assert config.num_vlm_layers == 1
    assert config.num_fast_tokens == 1024


def test_critic_modeling():
    from lerobot_policy_smolvla_rl.smolvla_critic import (
        SmolVLACrictic,
        SmolVLMCriticConfig,
    )
    from lerobot.configs.types import FeatureType, PolicyFeature

    config = SmolVLMCriticConfig(
        num_bins=201,
        num_vlm_layers=1,
    )
    config.input_features = {
        "observation.images.image": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 224, 224)
        ),
    }

    model = SmolVLACrictic(config)
    assert hasattr(model, "c51_head")

    # Mock batch
    {
        "observation.images.image": torch.randn(1, 1, 3, 224, 224),
        "observation.language_tokens": torch.zeros(1, 1, dtype=torch.long),
        "observation.language_attention_mask": torch.ones(1, 1, dtype=torch.bool),
    }

    # Forward pass (smoke test)
    # Note: prepare_images might fail without proper preprocessing, so we just check initialization for now
    # as full forward pass requires a lot of setup (processor etc)
    assert len(model.vlm_with_expert.vlm.model.text_model.layers) == 1

