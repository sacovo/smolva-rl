import argparse
import logging
import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from diffusers.optimization import get_scheduler
from lerobot.configs.types import FeatureType
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from lerobot_policy_smolvla_rl.checkpoint_utils import (
    resolve_checkpoints,
    load_checkpoint,
)
from lerobot_policy_smolvla_rl.dataloader_utils import (
    add_dataloader_args,
    build_dataloader,
)
from lerobot_policy_smolvla_rl.ds_utils import (
    calculate_returns,
    get_episode_lengths,
    get_max_task_lengths,
)
from lerobot_policy_smolvla_rl.smolvla_critic import SmolVLACrictic, SmolVLMCriticConfig

logger = logging.getLogger(__name__)


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
    add_dataloader_args(parser)
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
        "--limit_episodes",
        type=int,
        default=None,
        help="Limit the dataset to load only the first N episodes (useful to save memory or speed up testing)",
    )

    parser.add_argument(
        "--pretrained_critic_path",
        type=str,
        default=None,
        help="Path to pretrained critic checkpoint (.pt file) to finetune from",
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
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=None,
        help=(
            "List of camera keys to use "
            "(e.g. observation.images.exterior_image_1_left "
            "observation.images.wrist_image_left). "
            "If None, all cameras in the dataset are used."
        ),
    )
    parser.add_argument(
        "--tolerance_s",
        type=float,
        default=0.0001,
        help="Tolerance in seconds for timestamp matching when loading video frames",
    )
    parser.add_argument(
        "--video_backend",
        type=str,
        default="torchcodec",
        help="Backend for loading videos (either torchcodec or pyav)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Configure PyTorch multiprocessing sharing strategy to prevent
    # "RuntimeError: received 0 items of ancdata" from open file descriptor/shared memory limits
    try:
        import torch.multiprocessing as mp  # pylint: disable=import-outside-toplevel

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
    # Determine episodes to load
    episodes_to_load = args.episodes
    if episodes_to_load is None and args.limit_episodes is not None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

            meta = LeRobotDatasetMetadata(args.dataset_repo_id)
            total = meta.total_episodes
            episodes_to_load = list(range(min(args.limit_episodes, total)))
            print(
                f"Limiting dataset load to first {len(episodes_to_load)} episodes (out of {total} total episodes)"
            )
        except Exception as e:
            print(f"Warning: could not load dataset metadata to limit episodes: {e}")

    dataset_class = LeRobotDataset
    print(f"Loading dataset: {args.dataset_repo_id}")
    dataset_kwargs = {
        "episodes": episodes_to_load,
        "tolerance_s": args.tolerance_s,
        "video_backend": args.video_backend,
    }

    try:
        with accelerator.local_main_process_first():
            dataset = dataset_class(
                args.dataset_repo_id,
                **dataset_kwargs,
            )
    except Exception as e:
        print(
            f"Warning: local_main_process_first failed during dataset load, falling back to direct load: {e}"
        )
        dataset = dataset_class(
            args.dataset_repo_id,
            **dataset_kwargs,
        )

    # Compute maximum episode length for normalization
    episode_lengths = get_episode_lengths(dataset)
    max_lengths = get_max_task_lengths(dataset)
    episode_lengths = episode_lengths.to(accelerator.device)
    max_lengths = max_lengths.to(accelerator.device)

    dataloader = build_dataloader(
        dataset,
        args,
        shuffle=True,
        device=device,
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
    camera_map = {}
    features = dataset_to_policy_features(dataset.meta.features)
    if args.cameras:
        dataset_cameras = sorted(
            [k for k in features if k.startswith("observation.images.")]
        )
        if any(cam not in features for cam in args.cameras):
            if len(dataset_cameras) == len(args.cameras):
                for src, dst in zip(dataset_cameras, args.cameras):
                    print(f"Mapping dataset camera '{src}' -> policy expected '{dst}'")
                    camera_map[src] = dst
            else:
                raise ValueError(
                    f"Cannot remap cameras: dataset has {len(dataset_cameras)} cameras ({dataset_cameras}), "
                    f"but requested {len(args.cameras)} cameras ({args.cameras})"
                )

    if camera_map:
        mapped_features = {}
        for k, v in features.items():
            if k in camera_map:
                mapped_features[camera_map[k]] = v
            else:
                mapped_features[k] = v
        features = mapped_features

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

    config.input_features = input_features
    try:
        with accelerator.local_main_process_first():
            model = SmolVLACrictic(config).to(device)
    except Exception as e:
        print(
            f"Warning: local_main_process_first failed during model load, falling back to direct load: {e}"
        )
        model = SmolVLACrictic(config).to(device)

    # Load pretrained critic weights if provided
    if args.pretrained_critic_path:
        print(f"Loading pretrained critic weights from {args.pretrained_critic_path}")
        state_dict = torch.load(args.pretrained_critic_path, map_location="cpu")

        # Clean state dict (strip DDP/wrapper prefixes)
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                clean_state_dict[k[7:]] = v
            elif k.startswith("_orig_mod."):
                clean_state_dict[k[10:]] = v
            else:
                clean_state_dict[k] = v

        # Load weights into SmolVLACrictic model
        missing_keys, unexpected_keys = model.load_state_dict(
            clean_state_dict, strict=False
        )
        print(
            f"Loaded pretrained critic weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}"
        )

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
    camera_keys = [
        k for k in dataset.meta.features if k.startswith("observation.images.")
    ]
    print(f"Detected camera keys: {camera_keys}")

    model.train()
    step = 0

    pre = model.get_pre_processor(dataset)

    output_dir = os.path.join(args.save_dir, args.model_save_name)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_to_try, fallback_checkpoint = resolve_checkpoints(
        args.resume_from, output_dir
    )

    # Verify input_features and model parameters consistency across all ranks
    if accelerator.use_distributed:
        print(
            f"[Rank {accelerator.process_index}] Verifying model parameters and input features consistency..."
        )
        local_cams = sorted(list(input_features.keys()))
        local_params = []
        for name, param in model.named_parameters():
            local_params.append(
                (name, list(param.shape), str(param.dtype), param.requires_grad)
            )

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
                        print(
                            f"CRITICAL ERROR: Camera key list mismatch! Rank 0 has {ref_cams}, but Rank {rank_idx} has {rank_cams}"
                        )
                        mismatch_found = True

                # Check parameters consistency
                for rank_idx in range(1, accelerator.num_processes):
                    rank_params = gathered_params[rank_idx]
                    rank_dict = {item[0]: item for item in rank_params}

                    if len(ref_params) != len(rank_params):
                        print(
                            f"CRITICAL ERROR: Parameter count mismatch! Rank 0 has {len(ref_params)} params, but Rank {rank_idx} has {len(rank_params)} params."
                        )
                        mismatch_found = True

                    for name, shape, dtype, req_grad in ref_params:
                        if name not in rank_dict:
                            print(
                                f"CRITICAL ERROR: Parameter '{name}' in Rank 0 is missing in Rank {rank_idx}!"
                            )
                            mismatch_found = True
                        else:
                            _, r_shape, r_dtype, r_req_grad = rank_dict[name]
                            if shape != r_shape:
                                print(
                                    f"CRITICAL ERROR: Parameter '{name}' shape mismatch! Rank 0: {shape}, Rank {rank_idx}: {r_shape}"
                                )
                                mismatch_found = True
                            if dtype != r_dtype:
                                print(
                                    f"CRITICAL ERROR: Parameter '{name}' dtype mismatch! Rank 0: {dtype}, Rank {rank_idx}: {r_dtype}"
                                )
                                mismatch_found = True
                            if req_grad != r_req_grad:
                                print(
                                    f"CRITICAL ERROR: Parameter '{name}' requires_grad mismatch! Rank 0: {req_grad}, Rank {rank_idx}: {r_req_grad}"
                                )
                                mismatch_found = True

                    for name in rank_dict:
                        if name not in ref_dict:
                            print(
                                f"CRITICAL ERROR: Parameter '{name}' in Rank {rank_idx} is missing in Rank 0!"
                            )
                            mismatch_found = True

                if mismatch_found:
                    print(
                        "CRITICAL: PARAMETER OR FEATURE STRUCTURAL MISMATCH DETECTED ACROSS RANKS! Raising error to avoid DDP hang."
                    )
                    raise RuntimeError(
                        "Parameter/feature structural mismatch detected across ranks before DDP preparation."
                    )
                else:
                    print(
                        "SUCCESS: All model parameters and input features are perfectly consistent across all ranks!"
                    )
        except Exception as check_err:
            print(f"WARNING: Rank consistency check encountered an issue: {check_err}")

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    if checkpoint_to_try:
        args.resume_from, step = load_checkpoint(
            accelerator, checkpoint_to_try, fallback_checkpoint
        )

    progress_bar = tqdm(total=args.steps, initial=step, desc="Training")

    while step < args.steps:
        for batch in dataloader:
            if camera_map:
                batch = {camera_map.get(k, k): v for k, v in batch.items()}
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
