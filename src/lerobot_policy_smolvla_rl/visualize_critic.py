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

# pylint: disable=wrong-import-position
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
        "--advantage_horizon",
        type=int,
        default=50,
        help="N-step TD lookahead (N) for the advantage (paper uses N=50, must "
        "match compute_thresholds.py). Use a value >= the longest episode for "
        "the Monte-Carlo (pre-training) regime.",
    )
    parser.add_argument(
        "--epsilon_l",
        type=float,
        default=30.0,
        help="Percentile (0-100) for calculating the advantage threshold (epsilon_l) per task",
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
    parser.add_argument(
        "--drop_n_last_frames",
        type=int,
        default=0,
        help="Drop/Shift the last N frames (should match training)",
    )
    return parser.parse_args()


# pylint: disable=too-many-branches,too-many-statements,too-many-locals
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

    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }

    config.input_features = input_features
    model = SmolVLACrictic(config).to(args.device)

    model.load_state_dict(state_dict)
    model.eval()

    # Load full ds first for getting the correct max length
    max_lengths = get_max_task_lengths(dataset)
    episode_lengths = get_episode_lengths(dataset)

    # Check for episode success metadata in full dataset
    episode_successes = {}
    if "success" in dataset.meta.episodes.column_names:
        success_list = dataset.meta.episodes["success"]
        if hasattr(success_list, "to_pylist"):
            success_list = success_list.to_pylist()
        episode_indices = dataset.meta.episodes["episode_index"]
        if hasattr(episode_indices, "to_pylist"):
            episode_indices = episode_indices.to_pylist()
        episode_successes = {idx: s for idx, s in zip(episode_indices, success_list)}

    print(f"Loading dataset: {args.dataset_repo_id}")

    # Reload if we only use the smaller ds
    if args.episodes:
        dataset = LeRobotDataset(args.dataset_repo_id, episodes=args.episodes)

    # Identify camera keys in the dataset
    camera_keys = [k for k in dataset.meta.features if k.startswith("observation.images.")]
    print(f"Detected camera keys: {camera_keys}")

    support = torch.linspace(
        model.config.vmin, model.config.vmax, model.config.num_bins, device=args.device
    )

    # Collect results for each episode
    results_by_episode = {
        ep_idx: {
            "probs": np.zeros((episode_lengths[ep_idx], model.config.num_bins)),
            "expected_values": np.zeros(episode_lengths[ep_idx]),
            "gt_values": np.zeros(episode_lengths[ep_idx]),
            "frame_indices": np.zeros(episode_lengths[ep_idx]),
            "task_index": None,
        }
        for ep_idx in args.episodes
    }

    pre = model.get_pre_processor(dataset)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            _, probs = model(pre(batch))
            expected_value = (probs * support).sum(dim=-1)  # (B,)

            def to_numpy(val):
                if hasattr(val, "cpu"):
                    val = val.cpu()
                if hasattr(val, "numpy"):
                    return val.numpy()
                return np.array(val)

            batch_eps = to_numpy(batch["episode_index"])
            batch_frames = to_numpy(batch["frame_index"])
            batch_tasks = to_numpy(batch["task_index"])

            p_cpu = probs.cpu()
            ev_cpu = expected_value.cpu()

            # Calculate normalized steps remaining [-1, 0]
            returns = calculate_returns(
                episode_lengths,
                max_lengths,
                batch["task_index"],
                batch["episode_index"],
                batch["frame_index"],
                post_goal_buffer=args.drop_n_last_frames,
            )

            for i in range(len(batch_eps)):
                frame_idx = int(batch_frames[i])
                ep_idx = int(batch_eps[i])

                results_by_episode[ep_idx]["probs"][frame_idx, :] = p_cpu[i]
                results_by_episode[ep_idx]["expected_values"][frame_idx] = ev_cpu[i]
                results_by_episode[ep_idx]["frame_indices"][frame_idx] = batch_frames[i]
                results_by_episode[ep_idx]["gt_values"][frame_idx] = (
                    returns[i].cpu().item()
                )
                results_by_episode[ep_idx]["task_index"] = int(batch_tasks[i])

    # Calculate temporal advantages for all episodes to compute task-specific thresholds
    task_advantages = {}

    for ep_idx in args.episodes:
        res = results_by_episode[ep_idx]
        if res["task_index"] is None:
            continue

        sorted_indices = np.argsort(res["frame_indices"])
        all_expected_values = np.array(
            [res["expected_values"][i] for i in sorted_indices]
        )
        all_gt_values = np.array(
            [res["gt_values"][i] for i in sorted_indices]
        )
        T = len(all_expected_values)

        # N-step TD advantage (paper Appendix A-F, same formula as
        # advantage_utils.nstep_td_advantages):
        #   A_t = (R_t - V_t) - (R_{t+N} - V_{t+N})
        # Degenerates to the Monte-Carlo advantage R_t - V_t once t+N reaches
        # the episode end (the future terms vanish).
        mc_error = all_gt_values - all_expected_values  # R_t - V_t
        advantages = mc_error.copy()
        N = args.advantage_horizon
        if N < T:
            advantages[:-N] = mc_error[:-N] - mc_error[N:]
            # advantages[-N:] stay Monte-Carlo (future frame past episode end)
        res["advantages"] = advantages

        task_idx = res["task_index"]
        is_success = episode_successes.get(ep_idx, True)

        # Only use successful episodes for calculating threshold percentiles
        if is_success:
            if task_idx not in task_advantages:
                task_advantages[task_idx] = []
            task_advantages[task_idx].extend(advantages)

    # Compute thresholds (epsilon_l) per task (using successful episodes only)
    task_thresholds = {}
    for task_idx, advs in task_advantages.items():
        if len(advs) == 0:
            task_thresholds[task_idx] = 0.0
        else:
            task_thresholds[task_idx] = np.percentile(advs, args.epsilon_l)
        print(
            f"Task {task_idx} threshold (epsilon_l={args.epsilon_l}% from successes): {task_thresholds[task_idx]:.4f}"
        )

    for ep_idx in args.episodes:
        print(f"Plotting episode {ep_idx}...")
        res = results_by_episode[ep_idx]
        if not len(res["probs"]) > 0:
            print(f"No data for episode {ep_idx}, skipping.")
            continue

        # Sort by frame index just in case
        sorted_indices = np.argsort(res["frame_indices"])
        all_probs = (
            torch.stack([torch.tensor(res["probs"][i]) for i in sorted_indices])
            .float()
            .numpy()
        )
        all_expected_values = np.array(
            [res["expected_values"][i] for i in sorted_indices]
        )
        all_gt_values = np.array([res["gt_values"][i] for i in sorted_indices])

        T = all_probs.shape[0]
        all_advantages = res["advantages"]
        task_idx = res["task_index"]
        threshold = task_thresholds.get(task_idx, 0.0)

        all_advantage_bools = all_advantages > threshold

        # Plotting
        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(
            3,
            2,
            width_ratios=[1, 0.03],
            height_ratios=[2, 0.3, 6],
            wspace=0.01,
            hspace=0.05,
        )
        ax_adv = fig.add_subplot(gs[0, 0])
        ax_bool = fig.add_subplot(gs[1, 0], sharex=ax_adv)
        ax_hm = fig.add_subplot(gs[2, 0], sharex=ax_adv)
        cax = fig.add_subplot(gs[2, 1])

        plt.setp(ax_adv.get_xticklabels(), visible=False)
        plt.setp(ax_bool.get_xticklabels(), visible=False)

        # Plot Advantage bar
        colors = ["green" if a > threshold else "red" for a in all_advantages]
        ax_adv.bar(np.arange(T) + 0.5, all_advantages, color=colors, width=1.0)
        ax_adv.set_ylabel("Advantage")
        
        # Color and label title by success status if available
        title_color = "black"
        success_status = ""
        if ep_idx in episode_successes:
            is_success = episode_successes[ep_idx]
            success_status = " (SUCCESS)" if is_success else " (FAILURE)"
            title_color = "green" if is_success else "red"
        ax_adv.set_title(f"Critic Output - Episode {ep_idx}{success_status}", color=title_color, fontweight="bold")
        
        ax_adv.axhline(0, color="black", linewidth=1)
        ax_adv.axhline(
            threshold,
            color="blue",
            linestyle="--",
            linewidth=1.5,
            label=f"Thr ({threshold:.3f})",
        )
        ax_adv.set_xlim(0, T)
        ax_adv.legend(loc="upper right")

        # Plot Boolean bar
        cmap_bool = ListedColormap(["red", "green"])
        # We need to map boolean True/False to 1/0 for the colormap (0=red, 1=green)
        bool_array = all_advantage_bools.astype(int)[None, :]
        ax_bool.imshow(bool_array, aspect="auto", cmap=cmap_bool, extent=[0, T, 0, 1])
        ax_bool.set_yticks([])
        ax_bool.set_ylabel("Adv>Thr", rotation=0, labelpad=25, va="center")

        # Heatmap using standard matplotlib to ensure axes align
        img = ax_hm.imshow(
            all_probs.T,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[0, T, model.config.vmin, model.config.vmax],
        )
        # Add colorbar for the heatmap
        fig.colorbar(img, cax=cax, label="Probability")

        ax_hm.plot(
            np.arange(T) + 0.5,
            all_expected_values,
            color="white",
            linewidth=2,
            label="Expected Value",
        )
        ax_hm.plot(
            np.arange(T) + 0.5,
            all_gt_values,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Ground Truth",
        )

        ax_hm.set_xlabel("Step")
        ax_hm.set_ylabel("Time-to-Completion (steps)")
        ax_hm.legend(loc="lower left")

        status_suffix = ""
        if ep_idx in episode_successes:
            status_suffix = "_success" if episode_successes[ep_idx] else "_failure"
        output_path = os.path.join(args.output_dir, f"episode_{ep_idx}_critic{status_suffix}.png")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
