import accelerate.commands.config.sagemaker
from collections import defaultdict
from accelerate import Accelerator
import argparse
import os
import math
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_policy_smolvla_rl.smolvla_critic import (
    SmolVLMWithCriticModel,
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


def process_batch(batch, model, camera_keys, device):
    processor = model.processor
    batch_size = batch["observation.state"].shape[0]

    pixel_values_list = []
    for key in camera_keys:
        imgs = batch[key]  # Expected shape [B, C, H, W]
        # The processor expects a list of images
        imgs_list = [imgs[i].cpu() for i in range(batch_size)]

        # Use the processor to get the correctly shaped pixel_values for the vision model
        # images are already in [0, 1] range from LeRobotDataset, so we set do_rescale=False
        inputs = processor(images=imgs_list, return_tensors="pt", do_rescale=False)
        pixel_values_list.append(inputs["pixel_values"].to(device))

    tasks = batch["task"]
    # Tokenize the tasks
    text_inputs = processor.tokenizer(
        tasks, return_tensors="pt", padding=True, truncation=True
    )
    lang_tokens = text_inputs["input_ids"].to(device)
    lang_masks = text_inputs["attention_mask"].to(device)

    img_masks = [
        torch.ones(batch_size, dtype=torch.bool, device=device) for _ in camera_keys
    ]

    state = batch["observation.state"].to(device)
    # Ensure state matches max_state_dim
    if state.shape[1] < model.max_state_dim:
        pad_size = model.max_state_dim - state.shape[1]
        state = F.pad(state, (0, pad_size))
    elif state.shape[1] > model.max_state_dim:
        state = state[:, : model.max_state_dim]

    return pixel_values_list, img_masks, lang_tokens, lang_masks, state


def get_max_task_lengths(dataset: LeRobotDataset):
    """For each distinct task determine the maximum length and return the dictionary mapping task to max length

    Args:
        dataset (LeRobotDataset): The dataset to analyze
    """
    task_to_index = dataset.meta.tasks.to_dict()["task_index"]
    print(f"Unique tasks in dataset: {task_to_index}")

    task_max_lengths = defaultdict(int)
    for task, length in zip(
        dataset.meta.episodes["tasks"], dataset.meta.episodes["length"]
    ):
        task = task[0]
        task_index = task_to_index[task]

        task_max_lengths[task_index] = max(task_max_lengths[task_index], length)

    return torch.tensor(
        [v for k, v in sorted(task_max_lengths.items())], dtype=torch.float32
    )


def main():
    args = parse_args()
    accelerator = Accelerator()

    # Initialize Weights & Biases
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        name=args.job_name,
    )

    print(f"Loading dataset: {args.dataset_repo_id}")
    dataset = LeRobotDataset(args.dataset_repo_id, episodes=args.episodes)

    # Compute maximum episode length for normalization
    episode_lengths = torch.tensor(dataset.meta.episodes["length"])

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
    model = SmolVLMWithCriticModel(
        model_id=args.model_id, num_vlm_layers=args.num_vlm_layers
    ).to(accelerator.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Identify camera keys in the dataset
    camera_keys = [k for k in dataset.features if k.startswith("observation.images.")]
    print(f"Detected camera keys: {camera_keys}")

    model.train()
    step = 0

    os.makedirs(args.save_dir, exist_ok=True)

    progress_bar = tqdm(total=args.steps, desc="Training")

    accumulation_steps = args.accumulation_steps
    optimizer.zero_grad()

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    while step < args.steps:
        for batch in dataloader:
            if step >= args.steps:
                break

            # Preprocess the batch
            images, img_masks, lang_tokens, lang_masks, state = process_batch(
                batch, model, camera_keys, accelerator.device
            )

            episode_idx = batch["episode_index"]
            frame_idx = batch["frame_index"]
            task_idx = batch["task_index"]

            T = episode_lengths[episode_idx]  # Shape: [batch_size]
            rem_steps = torch.clamp(
                T - frame_idx - 1, min=0
            )  # Remaining steps until goal

            max_lens = max_lengths[task_idx]  # Max lengths for the tasks in the batch
            returns = -(rem_steps / max_lens)  # Normalize values between (-1,

            # Compute C51 target distribution
            target_dist = compute_c51_target_distribution(
                returns, num_bins=model.num_bins, vmin=model.vmin, vmax=model.vmax
            ).to(accelerator.device)

            # Forward pass
            logits = model(images, img_masks, lang_tokens, lang_masks, state)
            # Critic loss (cross entropy over C51 distribution)
            loss = F.cross_entropy(logits, target_dist)
            loss = loss / accumulation_steps

            accelerator.backward(loss)

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step % args.log_freq == 0:
                wandb.log({"loss": loss.item(), "step": step})
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            if (step + 1) % args.save_freq == 0:
                save_path = os.path.join(
                    args.save_dir, f"{args.model_save_name}_{step+1}.pt"
                )
                torch.save(model.state_dict(), save_path)

            step += 1
            progress_bar.update(1)

    progress_bar.close()

    # Final save
    save_path = os.path.join(args.save_dir, args.model_save_name + "_final.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Training completed. Final model saved to {save_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
