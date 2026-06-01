import argparse
import os
import json
import numpy as np

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
        "--critic_checkpoint", type=str, default=None, help="Path to trained critic checkpoint"
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
    parser.add_argument(
        "--precomputed_advantages",
        type=str,
        default=None,
        help="Path to precomputed advantages (.npy) array file to run in offline mode",
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
    parser.add_argument(
        "--pretrained_policy_path",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint (.pt) or exported directory containing model.safetensors to finetune from",
    )
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

    # Configure PyTorch multiprocessing sharing strategy to prevent
    # "RuntimeError: received 0 items of ancdata" from open file descriptor/shared memory limits
    try:
        import torch.multiprocessing as mp
        mp.set_sharing_strategy("file_system")
    except Exception as e:
        print(f"Warning: could not set sharing strategy: {e}")
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
    try:
        with accelerator.local_main_process_first():
            dataset = LeRobotDataset(args.dataset_repo_id, episodes=args.episodes)
    except Exception as e:
        print(f"Warning: local_main_process_first failed during dataset load, falling back to direct load: {e}")
        dataset = LeRobotDataset(args.dataset_repo_id, episodes=args.episodes)

    # 2. Check for precomputed advantages (.npy)
    precomputed_advantages = None
    safe_repo_name = args.dataset_repo_id.replace("/", "_")
    advantages_path = args.precomputed_advantages
    if not advantages_path:
        advantages_path = os.path.join(args.save_dir, f"task_advantages_{safe_repo_name}.npy")
    
    if os.path.exists(advantages_path):
        print(f"Loading pre-computed advantages from {advantages_path}")
        precomputed_advantages = np.load(advantages_path)
    
    # 3. Initialize/Load Critic (only if pre-computed advantages do not exist)
    critic = None
    features = dataset_to_policy_features(dataset.meta.features)
    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }

    if args.cameras:
        all_cameras = [k for k in input_features if k.startswith("observation.images.")]
        for cam in all_cameras:
            if cam not in args.cameras:
                print(f"Filtering out camera: {cam}")
                del input_features[cam]

    thresholds_save_path = args.thresholds_path
    if not thresholds_save_path:
        thresholds_save_path = os.path.join(
            args.save_dir,
            f"task_thresholds_{safe_repo_name}.json",
        )

    if precomputed_advantages is None:
        if not args.critic_checkpoint:
            raise ValueError("Critic checkpoint is needed for RECAP training unless pre-computed advantages are provided")

        print(f"Loading critic from {args.critic_checkpoint}")
        # Initialize config for critic
        critic_config = SmolVLMCriticConfig(
            num_bins=201,
            num_vlm_layers=args.critic_num_vlm_layers,
            input_features=input_features,
        )
        try:
            with accelerator.local_main_process_first():
                critic = SmolVLACrictic(critic_config).to(device)
                critic.load_state_dict(torch.load(args.critic_checkpoint, map_location=device))
        except Exception as e:
            print(f"Warning: local_main_process_first failed during critic load, falling back to direct load: {e}")
            critic = SmolVLACrictic(critic_config).to(device)
            critic.load_state_dict(torch.load(args.critic_checkpoint, map_location=device))
        
        critic.eval()

        support = torch.linspace(
            critic.config.vmin, critic.config.vmax, critic.config.num_bins, device=device
        )
        pre_critic = critic.get_pre_processor(dataset)

        # Calculate epsilon_l per task based on the raw dataset (only on main process to avoid multi-GPU I/O bottlenecks)
        if accelerator.is_main_process:
            task_thresholds = get_task_thresholds(
                critic,
                dataset,
                support,
                args.action_chunk_size,
                thresholds_save_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=8,
                num_workers=0,  # use 0 workers to prevent multiprocessing shm/forking pickler crashes on HPC nodes
            )
        
        accelerator.wait_for_everyone()

        if not accelerator.is_main_process:
            with open(thresholds_save_path, "r") as f:
                str_keys = json.load(f)
                task_thresholds = {int(k): v for k, v in str_keys.items()}
    else:
        # Load thresholds directly from JSON
        print(f"Loading pre-computed advantage thresholds from {thresholds_save_path}")
        with open(thresholds_save_path, "r") as f:
            str_keys = json.load(f)
            task_thresholds = {int(k): v for k, v in str_keys.items()}

    # 4. Wrap/Load Dataloader
    if precomputed_advantages is None:
        # Wrap dataset to include future frames on the fly for on-the-fly calculation
        wrapped_dataset = FutureFrameWrapper(dataset, args.action_chunk_size)
    else:
        # High-Speed Offline Mode: use the raw dataset directly (no future wrappers needed!)
        wrapped_dataset = dataset

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
    try:
        with accelerator.local_main_process_first():
            model = SmolVLARECAP(recap_config).to(device)
    except Exception as e:
        print(f"Warning: local_main_process_first failed during model load, falling back to direct load: {e}")
        model = SmolVLARECAP(recap_config).to(device)

    # Load pretrained policy weights if provided
    if args.pretrained_policy_path:
        print(f"Loading pretrained policy weights from {args.pretrained_policy_path}")
        if os.path.isdir(args.pretrained_policy_path):
            from safetensors.torch import load_file
            safetensors_path = os.path.join(args.pretrained_policy_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path)
            else:
                pt_path = os.path.join(args.pretrained_policy_path, "checkpoint_final.pt")
                if os.path.exists(pt_path):
                    state_dict = torch.load(pt_path, map_location=device)
                else:
                    raise FileNotFoundError(f"Could not find model.safetensors or checkpoint_final.pt in {args.pretrained_policy_path}")
        else:
            state_dict = torch.load(args.pretrained_policy_path, map_location=device)

        # Clean state dict to remove "model." prefix if it comes from an exported LeRobot Policy
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                clean_state_dict[k[6:]] = v
            else:
                clean_state_dict[k] = v

        # Load weights into SmolVLARECAP model
        missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)
        print(f"Loaded pretrained weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys: {unexpected_keys[:10]}")



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
                # Sort by step number (stripping any file extension like .pt)
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))
                args.resume_from = os.path.join(output_dir, latest_checkpoint)
            else:
                args.resume_from = None
        else:
            args.resume_from = None

    # Verify input_features and model parameters consistency across all ranks
    if accelerator.use_distributed:
        print(f"[Rank {accelerator.process_index}] Verifying model parameters and input features consistency...")
        local_cams = sorted(list(input_features.keys()))
        local_params = []
        for name, param in model.named_parameters():
            local_params.append((name, list(param.shape), str(param.dtype), param.requires_grad))
        
        gathered_cams = [None] * accelerator.num_processes
        gathered_params = [None] * accelerator.num_processes
        
        try:
            torch.distributed.all_gather_object(gathered_cams, local_cams)
            torch.distributed.all_gather_object(gathered_params, local_params)
            
            if accelerator.is_main_process:
                mismatch_found = False
                ref_cams = gathered_cams[0]
                ref_params = gathered_params[0]
                ref_dict = {item[0]: item for item in ref_params}
                
                # Check cameras consistency
                for rank_idx in range(1, accelerator.num_processes):
                    rank_cams = gathered_cams[rank_idx]
                    if ref_cams != rank_cams:
                        print(f"CRITICAL ERROR: Camera key list mismatch! Rank 0 has {ref_cams}, but Rank {rank_idx} has {rank_cams}")
                        mismatch_found = True
                        
                # Check parameters consistency
                for rank_idx in range(1, accelerator.num_processes):
                    rank_params = gathered_params[rank_idx]
                    rank_dict = {item[0]: item for item in rank_params}
                    
                    if len(ref_params) != len(rank_params):
                        print(f"CRITICAL ERROR: Parameter count mismatch! Rank 0 has {len(ref_params)} params, but Rank {rank_idx} has {len(rank_params)} params.")
                        mismatch_found = True
                        
                    for name, shape, dtype, req_grad in ref_params:
                        if name not in rank_dict:
                            print(f"CRITICAL ERROR: Parameter '{name}' in Rank 0 is missing in Rank {rank_idx}!")
                            mismatch_found = True
                        else:
                            r_name, r_shape, r_dtype, r_req_grad = rank_dict[name]
                            if shape != r_shape:
                                print(f"CRITICAL ERROR: Parameter '{name}' shape mismatch! Rank 0: {shape}, Rank {rank_idx}: {r_shape}")
                                mismatch_found = True
                            if dtype != r_dtype:
                                print(f"CRITICAL ERROR: Parameter '{name}' dtype mismatch! Rank 0: {dtype}, Rank {rank_idx}: {r_dtype}")
                                mismatch_found = True
                            if req_grad != r_req_grad:
                                print(f"CRITICAL ERROR: Parameter '{name}' requires_grad mismatch! Rank 0: {req_grad}, Rank {rank_idx}: {r_req_grad}")
                                mismatch_found = True
                                
                    for name in rank_dict:
                        if name not in ref_dict:
                            print(f"CRITICAL ERROR: Parameter '{name}' in Rank {rank_idx} is missing in Rank 0!")
                            mismatch_found = True
                            
                if mismatch_found:
                    print("CRITICAL: PARAMETER OR FEATURE STRUCTURAL MISMATCH DETECTED ACROSS RANKS! Raising error to avoid DDP hang.")
                    raise RuntimeError("Parameter/feature structural mismatch detected across ranks before DDP preparation.")
                else:
                    print("SUCCESS: All model parameters and input features are perfectly consistent across all ranks!")
        except Exception as check_err:
            print(f"WARNING: Rank consistency check encountered an issue: {check_err}")

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    step = 0
    if args.resume_from:
        accelerator.load_state(args.resume_from)
        try:
            step = int(os.path.basename(args.resume_from).split("_")[1].split(".")[0])
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
                    if precomputed_advantages is not None:
                        # High-Speed Offline Mode: direct array lookup using absolute index
                        batch_indices = batch["index"].cpu().numpy()
                        advantage = torch.tensor(precomputed_advantages[batch_indices], device=device)
                    else:
                        # On-the-fly Mode: compute using Critic forward passes
                        future_batch = extract_future_batch(batch)
                        has_future = batch["has_future"].to(device)
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
