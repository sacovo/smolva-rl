import argparse
import os

import torch
from accelerate import Accelerator
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.types import FeatureType
from lerobot.datasets.feature_utils import dataset_to_policy_features
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot_policy_smolvla_rl import SmolVLARECAP, SmolVLARECAPConfig
from lerobot_policy_smolvla_rl.smolvla_critic import SmolVLACrictic, SmolVLMCriticConfig
from lerobot_policy_smolvla_rl.advantage_utils import (
    FutureFrameWrapper,
    extract_future_batch,
    compute_temporal_advantage,
    get_task_thresholds,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SmolVLA RECAP (Phase 1)")
    parser.add_argument("--dataset_repo_id", type=str, required=True)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="List of episodes to load for testing",
    )
    parser.add_argument(
        "--critic_checkpoint", type=str, help="Path to trained critic checkpoint"
    )
    parser.add_argument(
        "--action_chunk_size",
        type=int,
        default=1,
        help="Number of frames to predict (chunk size) to compute advantage",
    )
    parser.add_argument(
        "--thresholds_path",
        type=str,
        default=None,
        help="Path to save/load epsilon_l thresholds",
    )
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_vlm_layers", type=int, default=-1)
    parser.add_argument("--critic_num_vlm_layers", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="outputs/recap_phase1")
    parser.add_argument("--wandb_project", type=str, default="smolvla-recap")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(project_name=args.wandb_project, config=vars(args))
    device = accelerator.device

    # 1. Load Dataset
    dataset = LeRobotDataset(args.dataset_repo_id, episodes=args.episodes)

    # 2. Load Critic (for advantage conditioning)
    critic = None
    if not args.critic_checkpoint:
        raise ValueError("Critic checkpoint is needed for RECAP training")

    print(f"Loading critic from {args.critic_checkpoint}")

    features = dataset_to_policy_features(dataset.meta.features)
    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }

    # Initialize config for critic
    critic_config = SmolVLMCriticConfig(
        num_bins=201,
        num_vlm_layers=args.critic_num_vlm_layers,
        input_features=input_features,
    )
    critic = SmolVLACrictic(critic_config).to(device)
    critic.load_state_dict(torch.load(args.critic_checkpoint, map_location=device))
    critic.eval()

    support = torch.linspace(
        critic.config.vmin, critic.config.vmax, critic.config.num_bins, device=device
    )
    pre_critic = critic.get_pre_processor(dataset)

    thresholds_save_path = args.thresholds_path
    if not thresholds_save_path:
        thresholds_save_path = os.path.join(
            args.save_dir,
            f"task_thresholds_{args.dataset_repo_id}.json",
        )

    # Calculate or load epsilon_l per task based on the raw dataset
    task_thresholds = get_task_thresholds(
        critic,
        dataset,
        support,
        args.action_chunk_size,
        thresholds_save_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=8,
        num_workers=args.num_workers,
    )

    # Wrap dataset to include future frames on the fly
    wrapped_dataset = FutureFrameWrapper(dataset, args.action_chunk_size)
    dataloader = DataLoader(
        wrapped_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # 3. Initialize RECAP Model
    action_dim = features["action"].shape[0]
    recap_config = SmolVLARECAPConfig(
        num_vlm_layers=args.num_vlm_layers,
        max_action_dim=action_dim,
        input_features=input_features,
        action_stats=dataset.meta.stats.get("action"),
    )
    model = SmolVLARECAP(recap_config).to(device)
    pre_recap = model.get_pre_processor(dataset)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # 4. Training Loop
    step = 0
    progress_bar = tqdm(total=args.steps, initial=step, desc="RECAP Phase 1")

    while step < args.steps:
        for batch in dataloader:
            if step >= args.steps:
                break

            # Label advantage
            with torch.no_grad():
                future_batch = extract_future_batch(batch)
                has_future = batch["has_future"].to(device)

                # Compute temporal advantage on the fly
                advantage, _, _ = compute_temporal_advantage(
                    critic, pre_critic, batch, future_batch, support, has_future
                )

                # Compare advantage against task-specific epsilon_l
                task_indices = batch["task_index"].cpu().numpy()
                batch_thresholds = torch.tensor(
                    [task_thresholds[int(t)] for t in task_indices],
                    dtype=torch.float32,
                    device=device,
                )

                advantage_bool = (advantage > batch_thresholds).tolist()

            # Prepare batch for RECAP (tokenization, normalization)
            recap_batch = pre_recap(batch)

            total_loss, ar_loss, flow_loss = model.compute_loss(
                recap_batch, advantage=advantage_bool
            )

            accelerator.backward(total_loss)
            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.log(
                    {
                        "total_loss": total_loss.item(),
                        "ar_loss": ar_loss.item(),
                        "flow_loss": flow_loss.item(),
                        "step": step,
                    }
                )
                progress_bar.set_postfix({"loss": f"{total_loss.item():.4f}"})

            if step % 1000 == 0:
                accelerator.save_state(os.path.join(args.save_dir, f"step_{step}"))

            step += 1
            progress_bar.update(1)

    accelerator.end_training()


if __name__ == "__main__":
    main()
