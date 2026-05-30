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
from accelerate.utils import DistributedDataParallelKwargs
from diffusers.optimization import get_scheduler

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_policy_smolvla_rl.smolvla_critic import (
    SmolVLACrictic,
    SmolVLMCriticConfig,
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
        default=8,
        help="Number of layers to keep in the VLM backbone",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5000, help="Total training steps")
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for the VLM backbone"
    )
    parser.add_argument(
        "--lr_head",
        type=float,
        default=1e-3,
        help="Learning rate for the critic head (c51_head)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=2.5e-6,
        help="Minimum learning rate for cosine decay",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=100, help="Number of warmup steps"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)

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
        help="Path to state to resume from, or 'auto' to find the latest in save_dir",
    )
    parser.add_argument(
        "--model_save_name",
        type=str,
        default="critic",
        help="Name of the experiment/model (used for output directory)",
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=16,
        help="Accumulate over multiple steps",
    )
    parser.add_argument(
        "--state_dropout",
        type=float,
        default=0.0,
        help="Probability of zeroing out the entire state during training",
    )
    parser.add_argument(
        "--end_weight",
        type=float,
        default=1.0,
        help="Loss weight multiplier for frames near the end of an episode",
    )
    parser.add_argument(
        "--end_threshold",
        type=float,
        default=-0.2,
        help="Return threshold above which end_weight is applied (e.g., -0.2 means last 20%%)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=args.accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )

    # Initialize Weights & Biases
    accelerator.init_trackers(
        project_name=args.wandb_project,
        config=vars(args),
        init_kwargs={"wandb": {"entity": args.wandb_entity, "name": args.job_name}},
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
        num_bins=201,
        freeze_vision_encoder=True,
        num_vlm_layers=args.num_vlm_layers,
        input_features=None,
        device=accelerator.device,
        state_dropout=args.state_dropout,
    )
    features = dataset_to_policy_features(dataset.meta.features)

    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }

    config.input_features = input_features
    model = SmolVLACrictic(config).to(device)

    # Group parameters for differential learning rates
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if "c51_head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer_grouped_parameters = [
        {"params": backbone_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr_head},
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )

    # Use cosine scheduler with warmup
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.steps,
    )

    # Identify camera keys in the dataset
    camera_keys = [k for k in dataset.features if k.startswith("observation.images.")]
    print(f"Detected camera keys: {camera_keys}")

    model.train()
    step = 0

    pre = model.get_pre_processor(dataset)

    output_dir = os.path.join(args.save_dir, args.model_save_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.resume_from == "auto":
        if os.path.exists(output_dir):
            checkpoints = [
                d
                for d in os.listdir(output_dir)
                if d.startswith("state_") and os.path.isdir(os.path.join(output_dir, d))
            ]
            if checkpoints:
                # Sort by step number
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[1]))
                args.resume_from = os.path.join(output_dir, latest_checkpoint)
            else:
                args.resume_from = None
        else:
            args.resume_from = None

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    if args.resume_from:
        accelerator.load_state(args.resume_from)
        # restore step from path if possible
        try:
            step = int(os.path.basename(args.resume_from).split("_")[1])
            print(f"Resumed training state from {args.resume_from} at step {step}")
        except (ValueError, IndexError):
            print(
                f"Resumed training state from {args.resume_from}, but could not determine step. Starting from 0."
            )

    progress_bar = tqdm(total=args.steps, initial=step, desc="Training")

    while step < args.steps:
        for batch in dataloader:
            with accelerator.accumulate(model):
                if step >= args.steps:
                    break

                returns = calculate_returns(
                    episode_lengths,
                    max_lengths,
                    batch["task_index"],
                    batch["episode_index"],
                    batch["frame_index"],
                )

                # Map normalized returns [-1.0, 0.0] to bin indices [0, 200]
                # Formula: index = (val - vmin) / (vmax - vmin) * (num_bins - 1)
                normalized_indices = (
                    (returns - config.vmin) / (config.vmax - config.vmin)
                ) * (config.num_bins - 1)
                time_to_completion = (
                    torch.clamp(normalized_indices, 0, config.num_bins - 1)
                    .long()
                    .to(device)
                )

                # Apply end-of-episode weighting
                weights = torch.ones_like(returns)
                if args.end_weight != 1.0:
                    weights = torch.where(
                        returns > args.end_threshold, args.end_weight, 1.0
                    )

                logits, _ = model(pre(batch))
                targets = torch.clamp(time_to_completion, 0, config.num_bins - 1)

                if args.end_weight != 1.0:
                    loss = F.cross_entropy(logits, targets, reduction="none")
                    loss = (loss * weights).mean()
                else:
                    loss = F.cross_entropy(logits, targets)

                # loss.backward()
                accelerator.backward(loss)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if step % args.log_freq == 0:
                    accelerator.log({"loss": loss.item(), "step": step})
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                if (step + 1) % args.save_freq == 0:
                    save_path = os.path.join(output_dir, f"state_{step + 1}.pt")
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
