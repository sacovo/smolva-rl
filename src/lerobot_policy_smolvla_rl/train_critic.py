from lerobot.configs.types import FeatureType
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot_policy_smolvla_rl.ds_utils import (
    get_max_task_lengths,
    get_episode_lengths,
    calculate_returns,
)
import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from accelerate import Accelerator

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_policy_smolvla_rl.smolvla_critic import (
    SmolVLACrictic,
    SmolVLMCriticConfig,
    compute_c51_target_distribution,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SmolVLA Critic")
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
        default=None,
        help="List of episodes to load for testing",
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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000, help="Total training steps")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--wandb_project", type=str, default="smolvla-critic")
    parser.add_argument("--job_name", type=str, default="train_critic")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="outputs/checkpoints_critic")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to state to resume from",
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=8,
        help="Number of steps to accumulate gradients for",
    )
    parser.add_argument(
        "--model_save_name",
        type=str,
        default="critic_final.pt",
        help="Filename for the final saved model (appended to save_dir)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator(log_with='wandb')

    # Initialize Weights & Biases
    accelerator.init_trackers(
        project_name=args.wandb_project,
        config=vars(args),
        init_kwargs={"wandb": {"entity": args.wandb_entity, "name": args.job_name}}
    )
    device = accelerator.device

    print(f"Loading dataset: {args.dataset_repo_id}")
    dataset = LeRobotDataset(args.dataset_repo_id, episodes=args.episodes)

    # Compute maximum episode length for normalization
    episode_lengths = get_episode_lengths(dataset)

    max_lengths = get_max_task_lengths(dataset)

    episode_lengths = episode_lengths.to(accelerator.device)
    max_lengths = max_lengths.to(accelerator.device)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(
        f"Initializing critic model from {args.model_id} (layers: {args.num_vlm_layers})"
    )
    config = SmolVLMCriticConfig(
        num_bins=51,
        freeze_vision_encoder=True,
        num_vlm_layers=args.num_vlm_layers,
        input_features=None,
        device=accelerator.device,
    )
    features = dataset_to_policy_features(dataset.meta.features)

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    config.input_features = input_features
    model = SmolVLACrictic(
        config
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Identify camera keys in the dataset
    camera_keys = [k for k in dataset.features if k.startswith("observation.images.")]
    print(f"Detected camera keys: {camera_keys}")

    model.train()
    step = 0

    pre = model.get_pre_processor(dataset)

    output_dir = os.path.join(args.save_dir, args.model_save_name)

    os.makedirs(output_dir, exist_ok=True)

    progress_bar = tqdm(total=args.steps, desc="Training")

    accumulation_steps = args.accumulation_steps
    optimizer.zero_grad()

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    if args.resume_from:
        accelerator.load_state(args.resume_from)
        print(f"Resumed training state from {args.resume_from}")


    while step < args.steps:
        for batch in dataloader:
            if step >= args.steps:
                break

            returns = calculate_returns(
                episode_lengths,
                max_lengths,
                batch["task_index"],
                batch["episode_index"],
                batch["frame_index"],
            )

            # Compute C51 target distribution
            target_dist = compute_c51_target_distribution(
                returns, num_bins=config.num_bins, vmin=config.vmin, vmax=config.vmax
            ).to(device)

            # Forward pass
            logits, predicted_dist = model(pre(batch))
            # Critic loss (cross entropy over C51 distribution)
            # loss = F.cross_entropy(logits, target_dist)
            loss = F.binary_cross_entropy_with_logits(
                logits, target_dist.unsqueeze(dim=1)
            )
            loss = loss / accumulation_steps

            # loss.backward()
            accelerator.backward(loss)

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step % args.log_freq == 0:
                accelerator.log({"loss": loss.item(), "step": step})
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            if (step + 1) % args.save_freq == 0:
                save_path = os.path.join(
                    output_dir, f"state_{step + 1}.pt"
                )
                # torch.save(model.state_dict(), save_path)
                accelerator.wait_for_everyone()
                accelerator.save_state(save_path)

            step += 1
            progress_bar.update(1)

    progress_bar.close()

    # Final save
    save_path = os.path.join(output_dir, "checkpoint_final.pt")

    accelerator.wait_for_everyone()
    model = accelerator.unwrap_model(model)
    torch.save(model.state_dict(), save_path)

    print(f"Training completed. Final model saved to {save_path}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
