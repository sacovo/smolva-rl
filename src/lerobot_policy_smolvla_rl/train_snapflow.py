import argparse
import logging
import os
import json
import numpy as np
import sys

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.types import FeatureType
from lerobot.datasets.feature_utils import dataset_to_policy_features
from tqdm import tqdm

from lerobot_policy_smolvla_rl import SmolVLARECAP, SmolVLARECAPConfig
from lerobot_policy_smolvla_rl.dataloader_utils import (
    add_dataloader_args,
    build_dataloader,
    patch_lerobot_dataset_reader,
)
from lerobot_policy_smolvla_rl.checkpoint_utils import (
    resolve_checkpoints,
    load_checkpoint,
    parse_duration_to_seconds,
)
from lerobot_policy_smolvla_rl.train_common import build_warmup_cosine_scheduler

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SmolVLA SnapFlow Distillation (Phase 2)")
    parser.add_argument("--recap_checkpoint", type=str, required=True,
                        help="Path to converged RECAP checkpoint (.pt file or exported directory)")
    parser.add_argument("--dataset_repo_id", type=str, required=True,
                        help="Same dataset repo id used for RECAP training")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--min_lr", type=float, default=2.5e-7,
                        help="Minimum learning rate for cosine decay")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="FM/consistency mixing ratio")
    parser.add_argument("--lambda_consistency", type=float, default=0.1,
                        help="Consistency loss weight")
    parser.add_argument("--clamp", type=float, default=20.0,
                        help="Velocity prediction clamp range")
    parser.add_argument("--save_dir", type=str, default="outputs/snapflow")
    parser.add_argument("--model_save_name", type=str, default="snapflow_model")
    parser.add_argument("--job_name", type=str, default="train_snapflow")
    parser.add_argument("--wandb_project", type=str, default="smolvla-snapflow")
    parser.add_argument("--wandb_entity", type=str, default=None)
    add_dataloader_args(parser)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--accumulation_steps", type=int, default=4,
                        help="Accumulate over multiple steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping (0 to disable)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to state to resume from, or 'auto' to find the latest in save_dir")
    parser.add_argument("--limit_episodes", type=int, default=None,
                        help="Limit the dataset to load only the first N episodes")
    parser.add_argument("--episodes", type=int, nargs="+", default=None)
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=5)
    parser.add_argument("--cameras", type=str, nargs="+", default=None)
    parser.add_argument("--tolerance_s", type=float, default=0.0001)
    parser.add_argument("--duration", type=str, default=None)
    parser.add_argument("--duration_buffer", type=int, default=600)
    parser.add_argument("--action_chunk_size", type=int, default=None,
                        help="Action chunk size (None to auto-detect from recap checkpoint)")
    return parser.parse_args()


def main():
    import time
    start_time = time.time()
    patch_lerobot_dataset_reader()
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

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

    # Resolve episodes to load
    episodes_to_load = args.episodes
    if episodes_to_load is None and args.limit_episodes is not None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
            meta = LeRobotDatasetMetadata(args.dataset_repo_id)
            total = meta.total_episodes
            episodes_to_load = list(range(min(args.limit_episodes, total)))
            print(f"Limiting dataset load to first {len(episodes_to_load)} episodes (out of {total} total episodes)")
        except Exception as e:
            print(f"Warning: could not load dataset metadata to limit episodes: {e}")

    # Load Dataset
    print(f"Loading dataset: {args.dataset_repo_id}")
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    
    ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id)

    # Load config fields and chunk_size from config.json if available
    config_json_path = None
    if os.path.isdir(args.recap_checkpoint):
        config_json_path = os.path.join(args.recap_checkpoint, "config.json")
    elif os.path.isfile(args.recap_checkpoint):
        config_json_path = os.path.join(os.path.dirname(args.recap_checkpoint), "config.json")

    chunk_size = args.action_chunk_size
    config_kwargs = {}
    if config_json_path and os.path.exists(config_json_path):
        try:
            with open(config_json_path, "r") as f:
                cfg_data = json.load(f)
                if chunk_size is None:
                    chunk_size = cfg_data.get("chunk_size", 20)
                    print(f"Auto-detected chunk_size = {chunk_size} from {config_json_path}")
                for field in ["num_vlm_layers", "num_fast_tokens", "model_id", "use_advantage_conditioning", "adv_dropout_rate"]:
                    if field in cfg_data:
                        config_kwargs[field] = cfg_data[field]
                        print(f"Loaded config field: {field} = {cfg_data[field]}")
        except Exception as e:
            print(f"Warning: Failed to load configuration details: {e}")

    if chunk_size is None:
        chunk_size = 20
        print(f"Defaulting chunk_size = {chunk_size}")

    # Instantiate dummy config to resolve delta_timestamps
    dummy_config = SmolVLARECAPConfig(chunk_size=chunk_size, n_action_steps=1)
    delta_timestamps = resolve_delta_timestamps(dummy_config, ds_meta)

    with accelerator.local_main_process_first():
        dataset = LeRobotDataset(
            args.dataset_repo_id,
            episodes=episodes_to_load,
            tolerance_s=args.tolerance_s,
            delta_timestamps=delta_timestamps,
        )

    # Map cameras
    camera_map = {}
    features = dataset_to_policy_features(dataset.meta.features)
    if args.cameras:
        dataset_cameras = sorted([k for k in features if k.startswith("observation.images.")])
        if any(cam not in features for cam in args.cameras):
            if len(dataset_cameras) == len(args.cameras):
                for src, dst in zip(dataset_cameras, args.cameras):
                    print(f"Mapping dataset camera '{src}' -> policy expected '{dst}'")
                    camera_map[src] = dst
            else:
                raise ValueError(
                    f"Cannot remap cameras: dataset has {len(dataset_cameras)} cameras, "
                    f"but requested {len(args.cameras)} cameras."
                )

    if camera_map:
        mapped_features = {}
        for k, v in features.items():
            if k in camera_map:
                mapped_features[camera_map[k]] = v
            else:
                mapped_features[k] = v
        features = mapped_features

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    if args.cameras:
        all_cameras = [k for k in input_features if k.startswith("observation.images.")]
        for cam in all_cameras:
            if cam not in args.cameras:
                print(f"Filtering out camera: {cam}")
                del input_features[cam]

    dataloader = build_dataloader(dataset, args, shuffle=True, device=None)

    # Initialize SmolVLARECAP Model
    action_dim = features["action"].shape[0]
    recap_config = SmolVLARECAPConfig(
        max_action_dim=action_dim,
        input_features=input_features,
        action_stats=dataset.meta.stats.get("action"),
        chunk_size=chunk_size,
        n_action_steps=1,
        snapflow_enabled=False,  # Keep False during training, we only set True at export!
        snapflow_alpha=args.alpha,
        snapflow_lambda=args.lambda_consistency,
        snapflow_clamp=args.clamp,
        **config_kwargs
    )
    if recap_config.pruned_layers is not None or recap_config.visual_tokens_keep is not None:
        raise ValueError("Pruning configuration must be disabled during co-training/distillation.")
    with accelerator.local_main_process_first():
        model = SmolVLARECAP(recap_config).to(device)

    # Load pretrained weights
    print(f"Loading weights from {args.recap_checkpoint}")
    if os.path.isdir(args.recap_checkpoint):
        from safetensors.torch import load_file
        safetensors_path = os.path.join(args.recap_checkpoint, "model.safetensors")
        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path)
        else:
            pt_path = os.path.join(args.recap_checkpoint, "checkpoint_final.pt")
            if os.path.exists(pt_path):
                state_dict = torch.load(pt_path, map_location="cpu")
            else:
                raise FileNotFoundError(f"Could not find model.safetensors or checkpoint_final.pt in {args.recap_checkpoint}")
    else:
        state_dict = torch.load(args.recap_checkpoint, map_location="cpu")

    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            clean_state_dict[k[6:]] = v
        else:
            clean_state_dict[k] = v

    missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)
    print(f"Loaded weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

    pre_recap = model.get_pre_processor(dataset)

    # Freeze VLM backbone entirely, only action expert and target_time_mlp are trainable
    for p in model.parameters():
        p.requires_grad = False

    trainable_modules = [
        model.target_time_mlp,
        model.vlm_with_expert.lm_expert,
        model.action_in_proj,
        model.action_out_proj,
        model.action_time_mlp_in,
        model.action_time_mlp_out,
    ]
    for module in trainable_modules:
        for p in module.parameters():
            p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Optimizer: {len(trainable_params)} trainable parameters")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )

    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
        base_lr=args.lr,
        min_lr=args.min_lr,
    )

    output_dir = os.path.join(args.save_dir, args.model_save_name)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_to_try, fallback_checkpoint = resolve_checkpoints(args.resume_from, output_dir)

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    step = 0
    if checkpoint_to_try:
        args.resume_from, step = load_checkpoint(accelerator, checkpoint_to_try, fallback_checkpoint)

    max_duration_seconds = parse_duration_to_seconds(args.duration)
    progress_bar = tqdm(total=args.steps, initial=step, desc="SnapFlow Distillation")

    while step < args.steps:
        for batch in dataloader:
            if max_duration_seconds is not None:
                elapsed = time.time() - start_time
                if elapsed >= max_duration_seconds - args.duration_buffer:
                    print(f"Approaching duration limit. Saving state at step {step} and exiting...")
                    save_path = os.path.join(output_dir, f"state_{step}.pt")
                    accelerator.wait_for_everyone()
                    accelerator.save_state(save_path)
                    progress_bar.close()
                    accelerator.end_training()
                    return

            if camera_map:
                batch = {camera_map.get(k, k): v for k, v in batch.items()}
            if step >= args.steps:
                break

            with accelerator.accumulate(model):
                recap_batch = pre_recap(batch)

                # Forward pass in SnapFlow mode
                loss, fm_loss, consistency_loss = model(
                    recap_batch,
                    mode="snapflow",
                    alpha=args.alpha,
                    lambda_c=args.lambda_consistency,
                    clamp=args.clamp,
                )

                accelerator.backward(loss)
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if step % args.log_freq == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    accelerator.log(
                        {
                            "snapflow/combined_loss": loss.item(),
                            "snapflow/fm_loss": fm_loss.item(),
                            "snapflow/consistency_loss": consistency_loss.item(),
                            "lr": lr,
                            "step": step,
                        }
                    )
                    progress_bar.set_postfix(
                        {"loss": f"{loss.item():.4f}", "fm": f"{fm_loss.item():.4f}", "consistency": f"{consistency_loss.item():.4f}"}
                    )

                if (step + 1) % args.save_freq == 0:
                    save_path = os.path.join(output_dir, f"state_{step + 1}.pt")
                    accelerator.wait_for_everyone()
                    accelerator.save_state(save_path)

                    if args.keep_last_n_checkpoints > 0 and accelerator.is_main_process:
                        existing = [
                            d for d in os.listdir(output_dir)
                            if d.startswith("state_") and os.path.isdir(os.path.join(output_dir, d))
                        ]
                        existing.sort(key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True)
                        for old_ckpt in existing[args.keep_last_n_checkpoints:]:
                            old_path = os.path.join(output_dir, old_ckpt)
                            import shutil
                            shutil.rmtree(old_path, ignore_errors=True)

                step += 1
                progress_bar.update(1)

    progress_bar.close()

    # Final save
    save_path = os.path.join(output_dir, "checkpoint_final.pt")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.state_dict(), save_path)
    print(f"Training completed. Final model saved to {save_path}")

    # Export to LeRobot format
    if accelerator.is_main_process:
        print("Converting and migrating policy to LeRobot format with SnapFlow active...")
        try:
            exported_dir = os.path.join(output_dir, "exported")
            os.makedirs(exported_dir, exist_ok=True)
            
            from lerobot_policy_smolvla_rl import SmolVLARECAPPolicy
            
            action_stats = dataset.meta.stats.get("action")
            if action_stats is not None:
                action_stats = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in action_stats.items()
                }

            config = SmolVLARECAPConfig(
                max_action_dim=action_dim,
                input_features=input_features,
                action_stats=action_stats,
                chunk_size=chunk_size,
                n_action_steps=1,
                device="cpu",
                snapflow_enabled=True,  # Set snapflow_enabled=True in exported config!
                snapflow_alpha=args.alpha,
                snapflow_lambda=args.lambda_consistency,
                snapflow_clamp=args.clamp,
                **config_kwargs
            )
            config.output_features = output_features
            
            policy = SmolVLARECAPPolicy(config)
            
            # Map weights to policy format
            clean_state_dict = {}
            for k, v in unwrapped_model.state_dict().items():
                if k.startswith("module."):
                    k = k[7:]
                if k.startswith("_orig_mod."):
                    k = k[10:]
                if not k.startswith("model."):
                    k = "model." + k
                clean_state_dict[k] = v
                
            policy.load_state_dict(clean_state_dict, strict=False)
            policy.save_pretrained(exported_dir)
            print(f"Base policy exported to {exported_dir}")
            
            # Run migration script
            import subprocess
            migrated_dir = os.path.join(output_dir, "migrated")
            cmd = [
                sys.executable,
                "-c",
                "import lerobot_policy_smolvla_rl; from lerobot.processor.migrate_policy_normalization import main; main()",
                "--pretrained-path", exported_dir,
                "--output-dir", migrated_dir
            ]
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print(f"Policy successfully migrated to: {migrated_dir}")

            # Populating migrated processors with statistics
            print("Populating migrated processors with dataset statistics...")
            stats_tensors = {}
            if dataset.meta.stats is not None:
                for feature_name, stat_dict in dataset.meta.stats.items():
                    stats_tensors[feature_name] = {}
                    for stat_name, val in stat_dict.items():
                        stats_tensors[feature_name][stat_name] = torch.tensor(val)
            
            from lerobot.policies.factory import make_pre_post_processors
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=config,
                dataset_stats=stats_tensors,
            )
            preprocessor.save_pretrained(migrated_dir)
            postprocessor.save_pretrained(migrated_dir)
            print(f"Successfully populated migrated processors with statistics in: {migrated_dir}")
            
        except Exception as export_err:
            print(f"Warning: Failed to automatically export and migrate policy: {export_err}")
            import traceback
            traceback.print_exc()

    accelerator.end_training()


if __name__ == "__main__":
    main()
