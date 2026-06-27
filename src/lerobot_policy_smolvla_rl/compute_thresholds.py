import argparse
import logging
import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.types import FeatureType
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot_policy_smolvla_rl.dataloader_utils import (
    add_dataloader_args,
    build_dataloader,
)
from lerobot_policy_smolvla_rl.smolvla_critic import SmolVLACrictic, SmolVLMCriticConfig
from lerobot_policy_smolvla_rl.ds_utils import get_episode_lengths, get_max_task_lengths, calculate_returns

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-compute Critic Advantages and Thresholds"
    )
    parser.add_argument(
        "--dataset_repo_id", type=str, required=True, help="HF dataset repo id"
    )
    parser.add_argument(
        "--critic_checkpoint",
        type=str,
        required=True,
        help="Path to trained critic checkpoint",
    )
    parser.add_argument(
        "--action_chunk_size",
        type=int,
        default=1,
        help="Chunk size for advantage calculation",
    )
    parser.add_argument(
        "--save_dir", type=str, default="outputs/recap_phase1", help="Output directory"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_vlm_layers",
        type=int,
        default=8,
        help="Number of VLM layers used in critic",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=None,
        help="List of camera keys to use. If None, all cameras in the dataset are used.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional delay in seconds between evaluation batches (GPU cooling).",
    )
    add_dataloader_args(parser)
    # RobustDataset IS safe here because we key V-values by sample index (from
    # batch["index"]), not by sequential position.  Corrupt samples are replaced
    # by random valid ones whose index gets stored instead; the corrupt sample's
    # index simply won't appear in the lookup dict, so its advantage stays NaN.
    # train_recap.py treats NaN advantages as 0 (neutral), and RobustDataset
    # there will only ever look up indices of decodable samples.
    parser.set_defaults(skip_bad_samples=True)
    return parser.parse_args()


# pylint: disable=too-many-statements,too-many-locals
def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        import torch.multiprocessing as mp  # pylint: disable=import-outside-toplevel

        mp.set_sharing_strategy("file_system")
    except Exception as e:
        print(f"Warning: could not set sharing strategy: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    safe_repo_name = args.dataset_repo_id.replace("/", "_")
    thresholds_path = os.path.join(
        args.save_dir, f"task_thresholds_{safe_repo_name}.json"
    )
    advantages_path = os.path.join(
        args.save_dir, f"task_advantages_{safe_repo_name}.npy"
    )

    print(f"Loading dataset: {args.dataset_repo_id}")
    dataset = LeRobotDataset(args.dataset_repo_id)
    episode_lengths = get_episode_lengths(dataset).numpy()
    max_lengths = get_max_task_lengths(dataset)

    # Load success flags for return calculation
    if "success" in dataset.meta.episodes.column_names:
        success_col = dataset.meta.episodes["success"]
        success_list = success_col.to_pylist() if hasattr(success_col, "to_pylist") else list(success_col)
        success_flags = torch.tensor(success_list, dtype=torch.bool)
    else:
        success_flags = None

    # Build episode-start lookup: ep_from[ep_idx] = first absolute sample index of that episode.
    # This lets us resolve (ep_idx, future_frame_idx) -> absolute sample index without relying
    # on positional array offsets that break when samples are skipped.
    ep_from = np.array(dataset.meta.episodes["dataset_from_index"])

    features = dataset_to_policy_features(dataset.meta.features)
    output_features = {
        k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    if args.cameras:
        all_cameras = [k for k in input_features if k.startswith("observation.images.")]
        for cam in all_cameras:
            if cam not in args.cameras:
                print(f"Filtering out camera: {cam}")
                del input_features[cam]

    print("Initializing critic...")
    critic_config = SmolVLMCriticConfig(
        num_bins=201,
        num_vlm_layers=args.num_vlm_layers,
        input_features=input_features,
    )
    critic = SmolVLACrictic(critic_config).to(device)
    critic.load_state_dict(torch.load(args.critic_checkpoint, map_location=device))
    critic.eval()

    support = torch.linspace(
        critic.config.vmin, critic.config.vmax, critic.config.num_bins, device=device
    )
    pre_critic = critic.get_pre_processor(dataset)

    # ── Step 1/3: Predict V(s_t) for all reachable frames ───────────────────
    # We key everything by the absolute sample index from batch["index"] so that
    # skipped (corrupt) samples simply have no entry in the dicts — their
    # advantage will be left as NaN and handled downstream.
    print("Step 1/3: Predicting V(s_t) for all frames...")
    device_obj = torch.device(device)
    dataloader = build_dataloader(
        dataset,
        args,
        shuffle=False,
        device=device_obj,
    )

    # sample_idx -> scalar float
    vs_by_idx: dict[int, float] = {}
    # sample_idx -> (episode_index, frame_index, task_index)
    meta_by_idx: dict[int, tuple[int, int, int]] = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Critic"):
            critic_batch = pre_critic(batch)
            for k, v in critic_batch.items():
                if isinstance(v, torch.Tensor):
                    critic_batch[k] = v.to(device)
                elif (
                    isinstance(v, list)
                    and len(v) > 0
                    and isinstance(v[0], torch.Tensor)
                ):
                    critic_batch[k] = [t.to(device) for t in v]

            _, probs = critic(critic_batch)
            v_s = (probs * support).sum(dim=-1).cpu().numpy()

            def to_numpy(val):
                if hasattr(val, "cpu"):
                    val = val.cpu()
                if hasattr(val, "numpy"):
                    return val.numpy()
                return np.array(val)

            sample_indices = to_numpy(batch["index"])
            episode_indices = to_numpy(batch["episode_index"])
            frame_indices = to_numpy(batch["frame_index"])
            task_indices = to_numpy(batch["task_index"])

            for idx, v, ep, fr, task in zip(
                sample_indices, v_s, episode_indices, frame_indices, task_indices
            ):
                vs_by_idx[int(idx)] = float(v)
                meta_by_idx[int(idx)] = (int(ep), int(fr), int(task))

            if args.delay > 0:
                time.sleep(args.delay)

    n_seen = len(vs_by_idx)
    n_skipped = len(dataset) - n_seen
    if n_skipped > 0:
        logger.warning(
            "%d / %d samples were skipped due to corrupt video frames. "
            "Their advantages will be NaN and treated as neutral (0) during RECAP training.",
            n_skipped,
            len(dataset),
        )
    print(f"  Evaluated {n_seen} samples ({n_skipped} skipped / corrupt).")

    # ── Step 2/3: Compute N-step TD advantages ───────────────────────────────
    # Paper (pi0.6) formula: A_t = sum(r_{t:t+N-1}) + V(s_{t+N}) - V(s_t)
    # Rewritten using actual returns: A_t = (R_t - V_t) - (R_{t+N} - V_{t+N})
    # When t+N >= T (near episode end): degenerates to MC advantage = R_t - V_t
    # For corrupt samples (not in vs_by_idx) or when the future frame is also
    # corrupt: leave as NaN.  train_recap.py calls np.nan_to_num(..., nan=0.0).
    print("Step 2/3: Computing N-step TD advantages...")
    advantages = np.full(len(dataset), np.nan, dtype=np.float32)

    # Precompute actual (normalized) returns for all frames that have metadata
    ep_lengths_tensor = torch.tensor(episode_lengths)
    sorted_indices = sorted(meta_by_idx.keys())
    all_ep_idxs = torch.tensor([meta_by_idx[i][0] for i in sorted_indices], dtype=torch.long)
    all_frame_idxs = torch.tensor([meta_by_idx[i][1] for i in sorted_indices], dtype=torch.long)
    all_task_idxs = torch.tensor([meta_by_idx[i][2] for i in sorted_indices], dtype=torch.long)
    all_rets = calculate_returns(
        ep_lengths_tensor, max_lengths,
        all_task_idxs, all_ep_idxs, all_frame_idxs,
        success_flags=success_flags,
    )
    returns_by_idx: dict[int, float] = {
        idx: float(r) for idx, r in zip(sorted_indices, all_rets)
    }

    for sample_idx, v_current in vs_by_idx.items():
        r_current = returns_by_idx[sample_idx]
        advantages[sample_idx] = r_current - v_current

    n_nan = int(np.isnan(advantages).sum())
    print(
        f"  {len(dataset) - n_nan} advantages computed, {n_nan} left as NaN (corrupt / unreachable)."
    )

    np.save(advantages_path, advantages)
    print(f"Saved pre-computed advantages to {advantages_path}")

    # ── Step 3/3: Compute per-task thresholds (30th percentile) ─────────────
    # Exclude NaN entries and failed episodes from threshold computation.
    print("Step 3/3: Computing 30th percentile task thresholds...")
    task_thresholds = {}
    
    # Collect task and advantage values only for successful episodes
    valid_tasks = []
    valid_advs = []
    for idx in sorted(vs_by_idx.keys()):
        ep_idx = meta_by_idx[idx][0]
        is_success = True
        if success_flags is not None:
            is_success = bool(success_flags[ep_idx].item())
        if is_success and not np.isnan(advantages[idx]):
            valid_tasks.append(meta_by_idx[idx][2])
            valid_advs.append(advantages[idx])
            
    valid_tasks = np.array(valid_tasks, dtype=np.int64)
    valid_advs = np.array(valid_advs, dtype=np.float32)

    unique_tasks = np.unique(np.array([meta_by_idx[idx][2] for idx in vs_by_idx]))
    for t in unique_tasks:
        mask = valid_tasks == t
        task_advs = valid_advs[mask]
        if len(task_advs) == 0:
            logger.warning("Task %d has no successful episodes — defaulting threshold to 0.0.", t)
            task_thresholds[int(t)] = 0.0
        else:
            task_thresholds[int(t)] = float(np.percentile(task_advs, 30))

    # Save thresholds to JSON
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(task_thresholds, f, indent=2)
    print(f"Saved task thresholds to {thresholds_path}")

    print("Done! Both files successfully saved on disk.")


if __name__ == "__main__":
    main()
