import argparse
import logging
import os

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from lerobot.configs.types import FeatureType
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from lerobot_policy_smolvla_rl import SmolVLARECAP, SmolVLARECAPConfig
from lerobot_policy_smolvla_rl.advantage_utils import load_thresholds
from lerobot_policy_smolvla_rl.checkpoint_utils import (
    load_checkpoint,
    parse_duration_to_seconds,
    resolve_checkpoints,
)
from lerobot_policy_smolvla_rl.dataloader_utils import (
    add_augmentation_args,
    add_dataloader_args,
    build_dataloader,
    build_image_transforms,
    patch_lerobot_dataset_reader,
)
from lerobot_policy_smolvla_rl.ds_utils import load_success_flags
from lerobot_policy_smolvla_rl.train_common import build_warmup_cosine_scheduler

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SmolVLA RECAP (Phase 2: Supervised/Imitation Finetuning or Phase 3: Rollout and Policy Enhancement)"
    )
    parser.add_argument(
        "--expert_mode",
        action="store_true",
        help="Enable Phase 2 Supervised/Imitation Finetuning. Bypasses the critic and treats all expert demonstration frames as advantageous.",
    )
    parser.add_argument(
        "--no_advantage_conditioning",
        action="store_false",
        dest="use_advantage_conditioning",
        default=True,
        help="Disable advantage conditioning tokens entirely (plain SmolVLA-style baseline; "
        "typically combined with --expert_mode and --ar_loss_weight 0)",
    )
    parser.add_argument(
        "--adv_dropout_rate",
        type=float,
        default=0.3,
        help="Probability of dropping the advantage token per batch so the unconditional "
        "predictor is learned (required for CFG). Set 0 with --expert_mode for a constant "
        "always-positive token, i.e. a SmolVLA-style baseline without CFG machinery.",
    )
    parser.add_argument("--dataset_repo_id", type=str, required=True)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="List of episodes to load for testing",
    )
    parser.add_argument(
        "--action_chunk_size",
        type=int,
        default=1,
        help="Action chunk size (H) predicted by the flow-matching expert",
    )
    parser.add_argument(
        "--thresholds_path",
        type=str,
        default=None,
        help="Path to pre-computed epsilon_l thresholds (.json). Defaults to "
        "<save_dir>/task_thresholds_<repo>.json. Generate with compute_thresholds.py.",
    )
    parser.add_argument(
        "--precomputed_advantages",
        type=str,
        default=None,
        help="Path to pre-computed advantages (.npy). Defaults to "
        "<save_dir>/task_advantages_<repo>.npy. Generate with compute_thresholds.py.",
    )
    parser.add_argument(
        "--demo_dataset_repo_id",
        type=str,
        default=None,
        help="Optional expert-demo dataset for RECAP co-training. Frames drawn from "
        "it are ALWAYS labeled positive/high-advantage (guaranteed-good anchor for "
        "the positive conditional branch).",
    )
    parser.add_argument(
        "--demo_mix_ratio",
        type=float,
        default=0.5,
        help="Probability that a training step draws its batch from the demo dataset "
        "(labeled positive) instead of the main dataset. Only used with "
        "--demo_dataset_repo_id.",
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
    parser.add_argument(
        "--ar_loss_weight",
        type=float,
        default=1.0,
        help="Weight for autoregressive loss",
    )
    parser.add_argument(
        "--fm_loss_weight",
        type=float,
        default=1.0,
        help="Weight for flow matching loss",
    )
    parser.add_argument(
        "--disable_ki",
        action="store_false",
        dest="knowledge_insulation",
        default=True,
        help="Disable Knowledge Insulation so the flow-matching loss trains the "
        "VLM backbone directly (SmolVLA paper-style; combine with "
        "--ar_loss_weight 0). Vision encoder stays frozen either way.",
    )
    parser.add_argument(
        "--loss_n_action_steps",
        type=int,
        default=0,
        help="Number of action steps to include in flow matching loss (0 = full chunk_size)",
    )
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
    add_dataloader_args(parser)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=4,
        help="Accumulate over multiple steps",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping (0 to disable)",
    )
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
        "--pretrained_policy_path",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint (.pt) or exported directory containing model.safetensors to finetune from",
    )
    parser.add_argument(
        "--keep_last_n_checkpoints",
        type=int,
        default=5,
        help="Keep only the last N checkpoints (0 to keep all)",
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
    add_augmentation_args(parser)
    parser.add_argument(
        "--tolerance_s",
        type=float,
        default=0.0001,
        help="Tolerance in seconds for timestamp matching when loading video frames",
    )
    parser.add_argument(
        "--duration",
        type=str,
        default=None,
        help="Maximum training duration (in hours or HH:MM:SS format). As soon as the elapsed time approaches this duration, the script saves its state and exits.",
    )
    parser.add_argument(
        "--duration_buffer",
        type=int,
        default=10 * 60,  # 10 minutes
        help="Buffer time in seconds to subtract from duration to ensure clean state saving before timeout",
    )
    return parser.parse_args()


# pylint: disable=too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks
def main():
    import time

    start_time = time.time()
    patch_lerobot_dataset_reader()
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
            from lerobot.datasets.lerobot_dataset import (
                LeRobotDatasetMetadata,
            )  # pylint: disable=import-outside-toplevel

            meta = LeRobotDatasetMetadata(args.dataset_repo_id)
            total = meta.total_episodes
            episodes_to_load = list(range(min(args.limit_episodes, total)))
            print(
                f"Limiting dataset load to first {len(episodes_to_load)} episodes (out of {total} total episodes)"
            )
        except Exception as e:
            print(f"Warning: could not load dataset metadata to limit episodes: {e}")

    # 1. Load Dataset
    print(f"Loading dataset: {args.dataset_repo_id}")
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id)
    # Instantiate dummy config to resolve delta_timestamps
    dummy_config = SmolVLARECAPConfig(
        chunk_size=args.action_chunk_size, n_action_steps=1
    )
    delta_timestamps = resolve_delta_timestamps(dummy_config, ds_meta)

    image_transforms = build_image_transforms(args.augmentation, args.augmentation_geom)

    with accelerator.local_main_process_first():
        dataset = LeRobotDataset(
            args.dataset_repo_id,
            episodes=episodes_to_load,
            tolerance_s=args.tolerance_s,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
        )

    # Load per-episode success flags (written by the annotation server / bag converter).
    # Defaults to all-True when the dataset has no success column so that datasets
    # produced without success annotations still work unchanged.
    success_flags = load_success_flags(dataset, default_all_success=True)
    if success_flags is not None:
        n_ok = success_flags.sum().item()
        print(f"Loaded episode success flags: {n_ok}/{len(success_flags)} successful.")

    # 2. Resolve input/output features (with optional camera remapping)
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

    # 3. Load pre-computed advantages + per-task thresholds.
    #    These MUST be generated beforehand with compute_thresholds.py — the
    #    trainer never runs the critic itself.  The advantage formula (N-step TD,
    #    paper Appendix A-F) lives entirely in advantage_utils / compute_thresholds.
    precomputed_advantages = None
    task_thresholds = None
    if not args.expert_mode:
        safe_repo_name = args.dataset_repo_id.replace("/", "_")
        advantages_path = args.precomputed_advantages or os.path.join(
            args.save_dir, f"task_advantages_{safe_repo_name}.npy"
        )
        thresholds_path = args.thresholds_path or os.path.join(
            args.save_dir, f"task_thresholds_{safe_repo_name}.json"
        )

        missing = [
            p for p in (advantages_path, thresholds_path) if not os.path.exists(p)
        ]
        if missing:
            raise FileNotFoundError(
                "Pre-computed advantage files are missing: "
                f"{missing}. Generate them first with:\n"
                f"    python src/lerobot_policy_smolvla_rl/compute_thresholds.py "
                f"--dataset_repo_id {args.dataset_repo_id} "
                f"--critic_checkpoint <path> --save_dir {args.save_dir}\n"
                "Or pass --expert_mode to skip advantage conditioning entirely."
            )

        print(f"Loading pre-computed advantages from {advantages_path}")
        precomputed_advantages = np.load(advantages_path)
        print(f"Loading pre-computed advantage thresholds from {thresholds_path}")
        task_thresholds = load_thresholds(thresholds_path)

    # 4. Build DataLoader over the raw dataset; advantages are looked up per
    #    frame via batch["index"].  No CudaPrefetcher — Accelerate handles
    #    device placement, and wrapping before accelerator.prepare() would
    #    conflict with DDP's DistributedSampler.
    dataloader = build_dataloader(
        dataset,
        args,
        shuffle=True,
        device=None,  # Disable CudaPrefetcher; Accelerate manages devices
    )

    # 4b. Optional expert-demo co-training loader (frames always labeled positive).
    demo_dataloader = None
    if args.demo_dataset_repo_id:
        print(
            f"Loading demo co-training dataset (labeled positive): {args.demo_dataset_repo_id}"
        )
        demo_meta = LeRobotDatasetMetadata(args.demo_dataset_repo_id)
        demo_delta = resolve_delta_timestamps(dummy_config, demo_meta)
        with accelerator.local_main_process_first():
            demo_dataset = LeRobotDataset(
                args.demo_dataset_repo_id,
                tolerance_s=args.tolerance_s,
                delta_timestamps=demo_delta,
            )
        demo_dataloader = build_dataloader(
            demo_dataset, args, shuffle=True, device=None
        )

    # 3. Initialize RECAP Model
    action_dim = features["action"].shape[0]
    recap_config = SmolVLARECAPConfig(
        num_vlm_layers=args.num_vlm_layers,
        max_action_dim=action_dim,
        input_features=input_features,
        action_stats=dataset.meta.stats.get("action"),
        chunk_size=args.action_chunk_size,
        n_action_steps=1,
        ar_loss_weight=args.ar_loss_weight,
        fm_loss_weight=args.fm_loss_weight,
        loss_n_action_steps=args.loss_n_action_steps,
        use_advantage_conditioning=args.use_advantage_conditioning,
        adv_dropout_rate=args.adv_dropout_rate,
        knowledge_insulation=args.knowledge_insulation,
    )
    if (
        recap_config.pruned_layers is not None
        or recap_config.visual_tokens_keep is not None
    ):
        raise ValueError(
            "Pruning configuration must be disabled during co-training/distillation."
        )
    with accelerator.local_main_process_first():
        model = SmolVLARECAP(recap_config).to(device)

    # Load pretrained policy weights if provided
    if args.pretrained_policy_path:
        print(f"Loading pretrained policy weights from {args.pretrained_policy_path}")
        if os.path.isdir(args.pretrained_policy_path):
            from safetensors.torch import (
                load_file,
            )  # pylint: disable=import-outside-toplevel

            safetensors_path = os.path.join(
                args.pretrained_policy_path, "model.safetensors"
            )
            if os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path)
            else:
                pt_path = os.path.join(
                    args.pretrained_policy_path, "checkpoint_final.pt"
                )
                if os.path.exists(pt_path):
                    state_dict = torch.load(pt_path, map_location="cpu")
                else:
                    raise FileNotFoundError(
                        f"Could not find model.safetensors or checkpoint_final.pt in {args.pretrained_policy_path}"
                    )
        else:
            state_dict = torch.load(args.pretrained_policy_path, map_location="cpu")

        # Clean state dict to remove "model." prefix if it comes from an exported LeRobot Policy
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                clean_state_dict[k[6:]] = v
            else:
                clean_state_dict[k] = v

        # Load weights into SmolVLARECAP model
        missing_keys, unexpected_keys = model.load_state_dict(
            clean_state_dict, strict=False
        )
        print(
            f"Loaded pretrained weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}"
        )
        if len(unexpected_keys) > 0:
            print(f"Unexpected keys: {unexpected_keys[:10]}")

    pre_recap = model.get_pre_processor(dataset)

    # Only include trainable parameters in the optimizer — frozen vision
    # encoder params would waste optimizer state memory (momentum/variance)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_total = sum(1 for _ in model.parameters())
    print(f"Optimizer: {len(trainable_params)}/{n_total} trainable parameters")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )

    # Cosine schedule with linear warmup and configurable min_lr floor.
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
        base_lr=args.lr,
        min_lr=args.min_lr,
    )

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
                            f"CRITICAL ERROR: Parameter count mismatch! "
                            f"Rank 0 has {len(ref_params)} params, but "
                            f"Rank {rank_idx} has {len(rank_params)} params."
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

                print(
                    "SUCCESS: All model parameters and input features are perfectly consistent across all ranks!"
                )
        except Exception as check_err:
            print(f"WARNING: Rank consistency check encountered an issue: {check_err}")

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    if demo_dataloader is not None:
        demo_dataloader = accelerator.prepare(demo_dataloader)

    def _cycle_loader(dl):
        while True:
            for _b in dl:
                yield _b

    demo_iter = _cycle_loader(demo_dataloader) if demo_dataloader is not None else None

    step = 0
    if checkpoint_to_try:
        print(f"Checkpoint to try: {checkpoint_to_try}")
        if fallback_checkpoint:
            print(f"Fallback checkpoint: {fallback_checkpoint}")
        args.resume_from, step = load_checkpoint(
            accelerator, checkpoint_to_try, fallback_checkpoint
        )
    else:
        print(
            f"WARNING: No checkpoint found to resume from. "
            f"resolve_checkpoints('{args.resume_from}', '{output_dir}') "
            f"returned None. Training will start from step 0."
        )

    max_duration_seconds = parse_duration_to_seconds(args.duration)

    # 4. Training Loop
    progress_bar = tqdm(total=args.steps, initial=step, desc="RECAP Phase 1")

    while step < args.steps:
        for batch in dataloader:
            # Co-training: with prob demo_mix_ratio, swap in an expert-demo batch
            # (labeled positive). The main-loader batch is skipped this step.
            is_demo = False
            if demo_iter is not None and torch.rand(1).item() < args.demo_mix_ratio:
                batch = next(demo_iter)
                is_demo = True
            if max_duration_seconds is not None:
                elapsed = time.time() - start_time
                if elapsed >= max_duration_seconds - args.duration_buffer:
                    print(
                        f"Approaching duration limit ({elapsed:.1f}s / {max_duration_seconds}s). "
                        f"Saving state at step {step} and exiting cleanly..."
                    )
                    save_path = os.path.join(output_dir, f"state_{step}.pt")
                    accelerator.wait_for_everyone()
                    accelerator.save_state(save_path)
                    print(f"State saved to {save_path}. Exiting.")
                    progress_bar.close()
                    accelerator.end_training()
                    return

            if camera_map:
                batch = {camera_map.get(k, k): v for k, v in batch.items()}
            if step >= args.steps:
                break

            with accelerator.accumulate(model):
                # Label advantage
                with torch.no_grad():
                    if is_demo or args.expert_mode:
                        # Expert-demo (or expert_mode) frames are guaranteed positive.
                        batch_size = batch["task_index"].shape[0]
                        advantage_bool = [True] * batch_size
                        advantage = torch.ones(
                            batch_size, device=device
                        )  # dummy for logging
                    else:

                        def to_numpy(val):
                            if hasattr(val, "cpu"):
                                val = val.cpu()
                            if hasattr(val, "numpy"):
                                return val.numpy()
                            return np.array(val)

                        # Direct lookup into the pre-computed advantage array by
                        # absolute frame index.  NaN entries are frames corrupt at
                        # precompute time; treat as 0 (neutral) — RobustDataset has
                        # already replaced them with a valid stand-in sample.
                        batch_indices = to_numpy(batch["index"])
                        raw_adv = np.nan_to_num(
                            precomputed_advantages[batch_indices], nan=0.0
                        )
                        advantage = torch.tensor(raw_adv, device=device)

                        # Compare advantage against task-specific epsilon_l
                        task_indices = to_numpy(batch["task_index"])
                        batch_thresholds = torch.tensor(
                            [task_thresholds[int(t)] for t in task_indices],
                            dtype=torch.float32,
                            device=device,
                        )

                        advantage_bool = (advantage > batch_thresholds).tolist()

                        # Force positive advantage for human intervention frames (Phase 3).
                        # Only applied when the episode was successful — an intervention
                        # in a failed episode should not be rewarded as advantageous.
                        if "intervention" in batch:
                            interventions = (
                                batch["intervention"].flatten().cpu().numpy()
                            )
                            ep_indices = to_numpy(batch["episode_index"])
                            for idx, (intervened, ep_idx) in enumerate(
                                zip(interventions, ep_indices)
                            ):
                                if intervened and (
                                    success_flags is None or success_flags[int(ep_idx)]
                                ):
                                    advantage_bool[idx] = True

                # Prepare batch for RECAP (tokenization, normalization)
                recap_batch = pre_recap(batch)

                total_loss, ar_loss, flow_loss = model(
                    recap_batch, advantage=advantage_bool
                )

                accelerator.backward(total_loss)
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if step % args.log_freq == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    adv_frac = sum(advantage_bool) / max(len(advantage_bool), 1)
                    accelerator.log(
                        {
                            "total_loss": total_loss.item(),
                            "ar_loss": ar_loss.item(),
                            "flow_loss": flow_loss.item(),
                            "lr": lr,
                            "step": step,
                            "advantage_positive_frac": adv_frac,
                            "advantage_mean": advantage.mean().item(),
                        }
                    )
                    progress_bar.set_postfix(
                        {
                            "loss": f"{total_loss.item():.4f}",
                            "lr": f"{lr:.2e}",
                            "adv+": f"{adv_frac:.0%}",
                        }
                    )

                if (step + 1) % args.save_freq == 0:
                    save_path = os.path.join(output_dir, f"state_{step + 1}.pt")
                    accelerator.wait_for_everyone()
                    accelerator.save_state(save_path)

                    # Clean up old checkpoints to avoid filling disk
                    if args.keep_last_n_checkpoints > 0 and accelerator.is_main_process:
                        existing = [
                            d
                            for d in os.listdir(output_dir)
                            if d.startswith("state_")
                            and os.path.isdir(os.path.join(output_dir, d))
                        ]
                        existing.sort(
                            key=lambda x: int(x.split("_")[1].split(".")[0]),
                            reverse=True,
                        )
                        for old_ckpt in existing[args.keep_last_n_checkpoints :]:
                            old_path = os.path.join(output_dir, old_ckpt)
                            import shutil

                            shutil.rmtree(old_path, ignore_errors=True)
                            print(f"Removed old checkpoint: {old_path}")

                step += 1
                progress_bar.update(1)

    progress_bar.close()

    # Final save
    save_path = os.path.join(output_dir, "checkpoint_final.pt")
    accelerator.wait_for_everyone()
    model = accelerator.unwrap_model(model)
    torch.save(model.state_dict(), save_path)
    print(f"Training completed. Final model saved to {save_path}")

    # Convert and migrate checkpoint to LeRobot format
    if accelerator.is_main_process:
        print("Converting and migrating policy to LeRobot format...")
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
                num_vlm_layers=args.num_vlm_layers,
                max_action_dim=action_dim,
                input_features=input_features,
                action_stats=action_stats,
                chunk_size=args.action_chunk_size,
                n_action_steps=1,  # Default to 1, can be overridden during evaluation
                use_advantage_conditioning=args.use_advantage_conditioning,
                knowledge_insulation=args.knowledge_insulation,
                device="cpu",
            )
            config.output_features = output_features

            policy = SmolVLARECAPPolicy(config)

            # Map weights to policy format
            clean_state_dict = {}
            for k, v in model.state_dict().items():
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

            # Run migration script preloading our package to register 'smolvla_recap'
            import subprocess

            migrated_dir = os.path.join(output_dir, "migrated")
            cmd = [
                ".venv/bin/python",
                "-c",
                "import lerobot_policy_smolvla_rl; from lerobot.processor.migrate_policy_normalization import main; main()",
                "--pretrained-path",
                exported_dir,
                "--output-dir",
                migrated_dir,
            ]
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print(f"Policy successfully migrated to: {migrated_dir}")

            # Load the migrated processors and inject dataset normalization stats
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
                pretrained_path=migrated_dir,
            )
            # Inject stats into normalizer/unnormalizer steps via load_state_dict
            # which properly updates both stats and _tensor_stats for serialization.
            flat_stats = {}
            for feat_name, stat_dict in stats_tensors.items():
                for stat_name, val in stat_dict.items():
                    flat_stats[f"{feat_name}.{stat_name}"] = val
            for step in preprocessor.steps:
                if hasattr(step, "load_state_dict") and hasattr(step, "norm_map"):
                    step.load_state_dict(flat_stats)
            for step in postprocessor.steps:
                if hasattr(step, "load_state_dict") and hasattr(step, "norm_map"):
                    step.load_state_dict(flat_stats)
            preprocessor.save_pretrained(migrated_dir)
            postprocessor.save_pretrained(migrated_dir)
            print(
                f"Successfully populated migrated processors with statistics in: {migrated_dir}"
            )

        except Exception as export_err:
            print(
                f"Warning: Failed to automatically export and migrate policy: {export_err}"
            )
            import traceback

            traceback.print_exc()

    accelerator.end_training()


if __name__ == "__main__":
    main()
