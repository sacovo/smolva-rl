from lerobot.envs.utils import FeatureType
from lerobot.policies.factory import dataset_to_policy_features
from lerobot_policy_smolvla_rl.ds_utils import (
    get_max_task_lengths,
    get_episode_lengths,
    calculate_returns,
)
import argparse
import os
import sys

# Add src to sys.path to allow importing from lerobot_policy_smolvla_rl
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_policy_smolvla_rl.smolvla_critic import SmolVLACrictic, SmolVLMCriticConfig 


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SmolVLA Critic")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        required=True,
        help="Hugging Face repo id of the dataset",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        required=True,
        help="List of episode indices to visualize",
    )
    parser.add_argument(
        "--model_id", type=str, default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    )
    parser.add_argument(
        "--num_vlm_layers",
        type=int,
        default=2,
        help="Number of layers to keep in the VLM backbone",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output_dir", type=str, default="outputs/plots")
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Max episode length used for normalization during training. If not provided, it will be calculated from the full dataset metadata.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading checkpoint from {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=args.device)

    # Auto-detect num_vlm_layers from state_dict
    layer_indices = []
    for key in state_dict.keys():
        if key.startswith("vlm.model.text_model.layers."):
            parts = key.split(".")
            if len(parts) > 4:
                try:
                    layer_indices.append(int(parts[4]))
                except ValueError:
                    pass
    if layer_indices:
        inferred_layers = max(layer_indices) + 1
        if inferred_layers != args.num_vlm_layers:
            print(
                f"Note: Inferred {inferred_layers} VLM layers from checkpoint (command line arg was {args.num_vlm_layers})."
            )
            args.num_vlm_layers = inferred_layers

    print(f"Loading critic model from {args.model_id} (layers: {args.num_vlm_layers})")
    dataset = LeRobotDataset(args.dataset_repo_id)
    config = SmolVLMCriticConfig(
        num_bins=201,
        num_vlm_layers=args.num_vlm_layers,
    )

    features = dataset_to_policy_features(dataset.meta.features)

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    config.input_features = input_features
    model = SmolVLACrictic(
        config
    ).to(args.device)

    model.load_state_dict(state_dict)
    model.eval()

    # Load full ds first for getting the correct max length
    max_lengths = get_max_task_lengths(dataset)
    episode_lengths = get_episode_lengths(dataset)

    print(f"Loading dataset: {args.dataset_repo_id}")

    # Reload if we only use the smaller ds
    if args.episodes:
        dataset = LeRobotDataset(args.dataset_repo_id, episodes=args.episodes)

    # Identify camera keys in the dataset
    camera_keys = [k for k in dataset.features if k.startswith("observation.images.")]
    print(f"Detected camera keys: {camera_keys}")

    support = torch.linspace(model.config.vmin, model.config.vmax, model.config.num_bins, device=args.device)

    # Collect results for each episode
    results_by_episode = {
        ep_idx: {
            "probs": np.zeros((episode_lengths[ep_idx], model.config.num_bins)),
            "expected_values": np.zeros(episode_lengths[ep_idx]),
            "gt_values": np.zeros(episode_lengths[ep_idx]),
            "frame_indices": np.zeros(episode_lengths[ep_idx]),
        }
        for ep_idx in args.episodes
    }

    pre = model.get_pre_processor(dataset)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):

            logits, probs = model(pre(batch))
            expected_value = (probs * support).sum(dim=-1)  # (B,)

            batch_eps = batch["episode_index"].cpu().numpy()
            batch_frames = batch["frame_index"].cpu().numpy()

            p_cpu = probs.cpu()
            ev_cpu = expected_value.cpu()

            # Calculate normalized steps remaining [-1, 0]
            returns = calculate_returns(
                episode_lengths,
                max_lengths,
                batch["task_index"],
                batch["episode_index"],
                batch["frame_index"],
            )

            for i in range(len(batch_eps)):
                frame_idx = batch["frame_index"][i].item()
                ep_idx = batch["episode_index"][i].item()

                results_by_episode[ep_idx]["probs"][frame_idx, :] = p_cpu[i]
                results_by_episode[ep_idx]["expected_values"][frame_idx] = ev_cpu[i]
                results_by_episode[ep_idx]["frame_indices"][frame_idx] = batch_frames[i]
                results_by_episode[ep_idx]["gt_values"][frame_idx] = returns[i].cpu().item()


    for ep_idx in args.episodes:
        print(f"Plotting episode {ep_idx}...")
        res = results_by_episode[ep_idx]
        if not len(res["probs"]) > 0:
            print(f"No data for episode {ep_idx}, skipping.")
            continue

        # Sort by frame index just in case
        sorted_indices = np.argsort(res["frame_indices"])
        all_probs = (
            torch.stack([torch.tensor(res["probs"][i]) for i in sorted_indices]).float().numpy()
        )
        all_expected_values = np.array(
            [res["expected_values"][i] for i in sorted_indices]
        )
        all_gt_values = np.array([res["gt_values"][i] for i in sorted_indices])

        # Plotting
        plt.figure(figsize=(15, 8))

        # Heatmap
        # We want the y-axis to be the support values.
        # sns.heatmap expects rows to be y-axis and columns to be x-axis.
        # Our all_probs is (T, num_bins). We want (num_bins, T).
        # We also want to invert the y-axis so -1.0 is at the bottom.
        # But for heatmaps, index 0 is at the top.
        # So if we want -1.0 at the bottom and 0.0 at the top:
        # Index 0 (top) -> 0.0, Index 50 (bottom) -> -1.0.
        # The support is linspace(-1.0, 0.0, 51), so support[0] = -1.0, support[50] = 0.0.
        # We flip the rows to have 0.0 at the top.

        ax = sns.heatmap(
            np.flipud(all_probs.T),
            cmap="viridis",
            cbar_kws={"label": "Probability"},
        )

        # Set ticks for Y axis
        num_ticks = 11
        ytick_indices = np.linspace(0, model.config.num_bins - 1, num_ticks)
        ytick_labels = np.linspace(model.config.vmax, model.config.vmin, num_ticks)
        ax.set_yticks(ytick_indices + 0.5)
        ax.set_yticklabels([f"{v:.1f}" for v in ytick_labels])

        # Set ticks for X axis
        T = all_probs.shape[0]
        xtick_step = max(1, T // 10)
        ax.set_xticks(np.arange(0, T, xtick_step) + 0.5)
        ax.set_xticklabels(np.arange(0, T, xtick_step))

        # Plot Expected Value and Ground Truth
        # We need to map the values to the heatmap y-coordinates.
        # value v -> y index in [0, 50]
        # y = (vmax - v) / (vmax - vmin) * (num_bins - 1)
        def val_to_y(v):
            return (model.config.vmax - v) / (model.config.vmax - model.config.vmin) * (
                model.config.num_bins - 1
            ) + 0.5

        plt.plot(
            np.arange(T) + 0.5,
            val_to_y(all_expected_values),
            color="white",
            linewidth=2,
            label="Expected Value",
        )
        plt.plot(
            np.arange(T) + 0.5,
            val_to_y(all_gt_values),
            color="red",
            linestyle="--",
            linewidth=2,
            label="Ground Truth",
        )

        plt.title(f"Critic Output - Episode {ep_idx}")
        plt.xlabel("Step")
        plt.ylabel("Time-to-Completion (steps)")
        plt.legend()

        output_path = os.path.join(args.output_dir, f"episode_{ep_idx}_critic.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
