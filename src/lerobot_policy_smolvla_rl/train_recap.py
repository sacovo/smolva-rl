import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_policy_smolvla_rl import SmolVLARECAP, SmolVLARECAPConfig
from lerobot_policy_smolvla_rl.smolvla_critic import SmolVLACrictic, SmolVLMCriticConfig
from lerobot_policy_smolvla_rl.ds_utils import get_episode_lengths, get_max_task_lengths, calculate_returns

def parse_args():
    parser = argparse.ArgumentParser(description="Train SmolVLA RECAP (Phase 1)")
    parser.add_argument("--dataset_repo_id", type=str, required=True)
    parser.add_argument("--critic_checkpoint", type=str, help="Path to trained critic checkpoint")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_vlm_layers", type=int, default=-1)
    parser.add_argument("--save_dir", type=str, default="outputs/recap_phase1")
    parser.add_argument("--wandb_project", type=str, default="smolvla-recap")
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(project_name=args.wandb_project, config=vars(args))
    device = accelerator.device

    # 1. Load Dataset
    dataset = LeRobotDataset(args.dataset_repo_id)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Pre-calculate returns and thresholds if possible, or do it on the fly
    episode_lengths = get_episode_lengths(dataset).to(device)
    max_lengths = get_max_task_lengths(dataset).to(device)

    # 2. Load Critic (for advantage conditioning)
    critic = None
    if not args.critic_checkpoint:
        raise ValueError("Critic checkpoint is needed for RECAP training")

    print(f"Loading critic from {args.critic_checkpoint}")
    # Initialize config for critic
    critic_config = SmolVLMCriticConfig(
        num_bins=201, 
        num_vlm_layers=8,
        image_features=[k for k in dataset.features if k.startswith("observation.images.")],
    )
    # Note: In a real scenario, we'd need to match the dataset features
    critic = SmolVLACrictic(critic_config).to(device)
    critic.load_state_dict(torch.load(args.critic_checkpoint, map_location=device))
    critic.eval()
    
    # Support for expected value calculation
    support = torch.linspace(critic.config.vmin, critic.config.vmax, critic.config.num_bins, device=device)
    pre_critic = critic.get_pre_processor(dataset)
    
    # Advantage threshold (in raw steps). Positive means doing better than average.
    advantage_threshold = 0.0 

    # 3. Initialize RECAP Model
    action_dim = dataset.features["action"].shape[1]
    recap_config = SmolVLARECAPConfig(
        num_vlm_layers=args.num_vlm_layers,
        max_action_dim=action_dim,
        image_features=[k for k in dataset.features if k.startswith("observation.images.")],
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
            advantage_bool = [True] * batch["action"].shape[0] # Default to positive for demo data
            if critic:
                with torch.no_grad():
                    # Calculate ground truth returns
                    actual_return = calculate_returns(
                        episode_lengths, max_lengths, 
                        batch["task_index"], batch["episode_index"], batch["frame_index"]
                    ).to(device)
                    
                    # Get expected return V(s) from critic
                    critic_batch = pre_critic(batch)
                    _, probs = critic(critic_batch)
                    
                    # Expected return V(s) in [-1, 0]
                    v_s = (probs * support).sum(dim=-1)
                    
                    # Advantage = V(s) - Actual_Return
                    advantage = v_s - actual_return
                    advantage_bool = (advantage > advantage_threshold).tolist()

            # Prepare batch for RECAP (tokenization, normalization)
            recap_batch = pre_recap(batch)

            # Compute Loss
            total_loss, ar_loss, flow_loss = model.compute_loss(recap_batch, advantage=advantage_bool)
            
            accelerator.backward(total_loss)
            optimizer.step()
            optimizer.zero_grad()

            
            if step % 10 == 0:
                accelerator.log({
                    "total_loss": total_loss.item(),
                    "ar_loss": ar_loss.item(),
                    "flow_loss": flow_loss.item(),
                    "step": step
                })
                progress_bar.set_postfix({"loss": f"{total_loss.item():.4f}"})
                
            if step % 1000 == 0:
                accelerator.save_state(os.path.join(args.save_dir, f"step_{step}"))
                
            step += 1
            progress_bar.update(1)

    accelerator.end_training()

if __name__ == "__main__":
    main()
