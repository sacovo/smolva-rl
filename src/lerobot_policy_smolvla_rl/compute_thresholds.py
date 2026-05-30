import argparse
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.types import FeatureType
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot_policy_smolvla_rl.smolvla_critic import SmolVLACrictic, SmolVLMCriticConfig
from lerobot_policy_smolvla_rl.ds_utils import get_episode_lengths, get_max_task_lengths


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-compute Critic Advantages and Thresholds")
    parser.add_argument("--dataset_repo_id", type=str, required=True, help="HF dataset repo id")
    parser.add_argument("--critic_checkpoint", type=str, required=True, help="Path to trained critic checkpoint")
    parser.add_argument("--action_chunk_size", type=int, default=1, help="Chunk size for advantage calculation")
    parser.add_argument("--save_dir", type=str, default="outputs/recap_phase1", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--num_vlm_layers", type=int, default=8, help="Number of VLM layers used in critic")
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=None,
        help="List of camera keys to use (e.g. observation.images.exterior_image_1_left observation.images.wrist_image_left). If None, all cameras in the dataset are used.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    safe_repo_name = args.dataset_repo_id.replace("/", "_")
    thresholds_path = os.path.join(args.save_dir, f"task_thresholds_{safe_repo_name}.json")
    advantages_path = os.path.join(args.save_dir, f"task_advantages_{safe_repo_name}.npy")

    print(f"Loading dataset: {args.dataset_repo_id}")
    dataset = LeRobotDataset(args.dataset_repo_id)
    episode_lengths = get_episode_lengths(dataset).numpy()

    features = dataset_to_policy_features(dataset.meta.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
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
    critic.eval()

    support = torch.linspace(critic.config.vmin, critic.config.vmax, critic.config.num_bins, device=device)
    pre_critic = critic.get_pre_processor(dataset)

    print("Step 1/3: Predicting V(s_t) for all frames...")
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    all_vs_list = []
    all_tasks_list = []
    all_episodes_list = []
    all_frames_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Critic"):
            critic_batch = pre_critic(batch)
            # Move to device
            for k, v in critic_batch.items():
                if isinstance(v, torch.Tensor):
                    critic_batch[k] = v.to(device)
                elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                    critic_batch[k] = [t.to(device) for t in v]

            _, probs = critic(critic_batch)
            v_s = (probs * support).sum(dim=-1).cpu().numpy()

            all_vs_list.append(v_s)
            all_tasks_list.append(batch["task_index"].numpy())
            all_episodes_list.append(batch["episode_index"].numpy())
            all_frames_list.append(batch["frame_index"].numpy())

    all_vs = np.concatenate(all_vs_list)
    all_tasks = np.concatenate(all_tasks_list)
    all_episodes = np.concatenate(all_episodes_list)
    all_frames = np.concatenate(all_frames_list)

    print("Step 2/3: Computing temporal advantages...")
    advantages = np.zeros(len(dataset), dtype=np.float32)
    for i in range(len(dataset)):
        ep_idx = all_episodes[i]
        frame_idx = all_frames[i]
        ep_len = episode_lengths[ep_idx]

        v_current = all_vs[i]
        future_frame = frame_idx + args.action_chunk_size

        if future_frame >= ep_len:
            v_future = 0.0
        else:
            # Episodes are stored contiguously in LeRobotDataset
            v_future = all_vs[i + args.action_chunk_size]

        advantages[i] = v_future - v_current

    # Save advantages to binary .npy
    np.save(advantages_path, advantages)
    print(f"Saved pre-computed advantages to {advantages_path}")

    print("Step 3/3: Computing 30th percentile task thresholds...")
    task_thresholds = {}
    unique_tasks = np.unique(all_tasks)
    for t in unique_tasks:
        task_advs = advantages[all_tasks == t]
        threshold = np.percentile(task_advs, 30)
        task_thresholds[int(t)] = float(threshold)

    # Save thresholds to JSON
    with open(thresholds_path, "w") as f:
        json.dump(task_thresholds, f, indent=2)
    print(f"Saved task thresholds to {thresholds_path}")

    print("Done! Both files successfully saved on disk.")


if __name__ == "__main__":
    main()
