import argparse
import os

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from diffusers.optimization import get_scheduler
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
    parser.add_argument(
        "--min_lr",
        type=float,
        default=2.5e-7,
        help="Minimum learning rate for cosine decay",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=100, help="Number of warmup steps"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--num_vlm_layers", type=int, default=-1)
    parser.add_argument("--critic_num_vlm_layers", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="outputs/recap_phase1")
    parser.add_argument(
        "--model_save_name",
        type=str,
        default="recap_model",
        help="Name of the experiment/model (used for output directory)",
    )
    parser.add_argument("--job_name", type=str, default="train_recap")
    parser.add_argument("--wandb_project", type=str, default="smolvla-recap")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=4,
        help="Accumulate over multiple steps",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to state to resume from, or 'auto' to find the latest in save_dir",
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

    # 1. Load Dataset
    print(f"Loading dataset: {args.dataset_repo_id}")
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
        safe_repo_name = args.dataset_repo_id.replace("/", "_")
        thresholds_save_path = os.path.join(
            args.save_dir,
            f"task_thresholds_{safe_repo_name}.json",
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
        chunk_size=1,
        n_action_steps=1,
    )
    model = SmolVLARECAP(recap_config).to(device)
    pre_recap = model.get_pre_processor(dataset)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )

    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.steps,
    )

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

    step = 0
    if args.resume_from:
        accelerator.load_state(args.resume_from)
        try:
            step = int(os.path.basename(args.resume_from).split("_")[1])
            print(f"Resumed training state from {args.resume_from} at step {step}")
        except (ValueError, IndexError):
            print(
                f"Resumed training state from {args.resume_from}, but could not determine step. Starting from 0."
            )

    # 4. Training Loop
    progress_bar = tqdm(total=args.steps, initial=step, desc="RECAP Phase 1")

    while step < args.steps:
        for batch in dataloader:
            if step >= args.steps:
                break

            with accelerator.accumulate(model):
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

                total_loss, ar_loss, flow_loss = model(
                    recap_batch, advantage=advantage_bool
                )

                accelerator.backward(total_loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if step % args.log_freq == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    accelerator.log(
                        {
                            "total_loss": total_loss.item(),
                            "ar_loss": ar_loss.item(),
                            "flow_loss": flow_loss.item(),
                            "lr": lr,
                            "step": step,
                        }
                    )
                    progress_bar.set_postfix({"loss": f"{total_loss.item():.4f}", "lr": f"{lr:.2e}"})

                if (step + 1) % args.save_freq == 0:
                    save_path = os.path.join(output_dir, f"state_{step + 1}.pt")
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
