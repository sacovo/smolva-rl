"""CLI to pre-compute RECAP advantages and per-task thresholds offline.

Thin wrapper around ``advantage_utils.precompute_advantages_and_thresholds`` so
the advantage formula lives in exactly one place (see that module's docstring
for the N-step TD definition from the paper).
"""

import argparse
import logging
import os

import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_policy_smolvla_rl.advantage_utils import (
    precompute_advantages_and_thresholds,
    save_advantages_and_thresholds,
)
from lerobot_policy_smolvla_rl.dataloader_utils import add_dataloader_args
from lerobot_policy_smolvla_rl.smolvla_critic import SmolVLACrictic, SmolVLMCriticConfig

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-compute Critic Advantages and Thresholds"
    )
    parser.add_argument(
        "--dataset_repo_id", type=str, required=True, help="HF dataset repo id"
    )
    parser.add_argument(
        "--critic_checkpoint",
        type=str,
        required=True,
        help="Path to trained critic checkpoint",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of episodes to evaluate (default: whole dataset)",
    )
    parser.add_argument(
        "--advantage_horizon",
        type=int,
        default=50,
        help="N-step TD lookahead used for the advantage (paper uses N=50 for "
        "post-training). Use a value >= the longest episode for the "
        "Monte-Carlo (pre-training) regime.",
    )
    parser.add_argument(
        "--save_dir", type=str, default="outputs/recap_phase1", help="Output directory"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_vlm_layers",
        type=int,
        default=8,
        help="Number of VLM layers used in critic",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=None,
        help="List of camera keys to use. If None, all cameras in the dataset are used.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional delay in seconds between evaluation batches (GPU cooling).",
    )
    add_dataloader_args(parser)
    parser.set_defaults(skip_bad_samples=True)
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        import torch.multiprocessing as mp  # pylint: disable=import-outside-toplevel

        mp.set_sharing_strategy("file_system")
    except Exception as e:
        print(f"Warning: could not set sharing strategy: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    safe_repo_name = args.dataset_repo_id.replace("/", "_")
    thresholds_path = os.path.join(
        args.save_dir, f"task_thresholds_{safe_repo_name}.json"
    )
    advantages_path = os.path.join(
        args.save_dir, f"task_advantages_{safe_repo_name}.npy"
    )

    print(f"Loading dataset: {args.dataset_repo_id}")
    dataset = LeRobotDataset(args.dataset_repo_id, episodes=args.episodes)

    features = dataset_to_policy_features(dataset.meta.features)
    output_features = {
        k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    if args.cameras:
        all_cameras = [k for k in input_features if k.startswith("observation.images.")]
        for cam in all_cameras:
            if cam not in args.cameras:
                print(f"Filtering out camera: {cam}")
                del input_features[cam]

    print("Initializing critic...")
    critic_config = SmolVLMCriticConfig(
        num_bins=201,
        num_vlm_layers=args.num_vlm_layers,
        input_features=input_features,
    )
    critic = SmolVLACrictic(critic_config).to(device)
    critic.load_state_dict(torch.load(args.critic_checkpoint, map_location=device))

    print(f"Computing N-step TD advantages (N={args.advantage_horizon})...")
    advantages, task_thresholds = precompute_advantages_and_thresholds(
        critic,
        dataset,
        advantage_horizon=args.advantage_horizon,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        skip_bad_samples=args.skip_bad_samples,
        delay=args.delay,
    )

    save_advantages_and_thresholds(
        advantages, task_thresholds, advantages_path, thresholds_path
    )
    print(f"Saved pre-computed advantages to {advantages_path}")
    print(f"Saved task thresholds to {thresholds_path}")
    print("Done! Both files successfully saved on disk.")


if __name__ == "__main__":
    main()
