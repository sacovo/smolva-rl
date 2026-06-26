import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from lerobot_policy_smolvla_rl.ds_utils import (
    get_episode_lengths,
    get_max_task_lengths,
    calculate_returns as calculate_returns_fn,
)


class FutureFrameWrapper(Dataset):
    def __init__(self, dataset, chunk_size):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.episode_lengths = get_episode_lengths(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        frame_idx = item["frame_index"].item()
        ep_idx = item["episode_index"].item()
        ep_len = self.episode_lengths[ep_idx].item()

        future_frame = frame_idx + self.chunk_size
        has_future = future_frame < ep_len

        # We store has_future
        item["has_future"] = torch.tensor(has_future, dtype=torch.bool)

        if has_future:
            future_item = self.dataset[idx + self.chunk_size]
        else:
            future_item = self.dataset[idx]

        # Prefix future item keys
        for k, v in list(future_item.items()):
            if k != "has_future":
                item[f"future_{k}"] = v

        return item


def extract_future_batch(batch):
    """
    Extracts the future items prefixed with 'future_' into a separate batch dictionary.
    """
    future_batch = {}
    for k in list(batch.keys()):
        if k.startswith("future_"):
            future_batch[k[len("future_") :]] = batch[k]
    return future_batch


# pylint: disable=too-many-arguments,too-many-positional-arguments
def compute_temporal_advantage(
    critic, pre_critic, batch, future_batch, support, has_future,
    actual_returns=None, future_returns=None,
):
    """
    Computes the N-step TD advantage per the pi0.6 paper:
    A(s_t) = sum(r_{t:t+N-1}) + V(s_{t+N}) - V(s_t)

    Rewritten using actual returns:
    A(s_t) = (R_t - V(s_t)) - (R_{t+N} - V(s_{t+N}))

    When actual_returns / future_returns are None, falls back to the
    pure temporal difference V(s_{t+N}) - V(s_t) (no reward sum).

    If not has_future (end of episode), both R_{t+N} and V(s_{t+N})
    are 0, so A(s_t) = R_t - V(s_t)  (Monte Carlo advantage).
    """
    # 1. Current V(s_t)
    critic_batch = pre_critic(batch)
    _, probs = critic(critic_batch)
    v_s = (probs * support).sum(dim=-1)

    # 2. Future V(s_{t+chunk_size})
    future_critic_batch = pre_critic(future_batch)
    _, future_probs = critic(future_critic_batch)
    v_s_future = (future_probs * support).sum(dim=-1)

    # Where not has_future, v_s_future should be 0.0
    v_s_future = torch.where(has_future, v_s_future, torch.zeros_like(v_s_future))

    if actual_returns is not None and future_returns is not None:
        # N-step TD advantage: (R_t - V_t) - (R_{t+N} - V_{t+N})
        future_returns = torch.where(has_future, future_returns, torch.zeros_like(future_returns))
        mc_error_current = actual_returns - v_s
        mc_error_future = future_returns - v_s_future
        advantage = mc_error_current - mc_error_future
    else:
        # Fallback: pure temporal difference (no reward sum)
        advantage = v_s_future - v_s

    return advantage, v_s, v_s_future


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def get_task_thresholds(
    critic_model,
    dataset,
    support,
    chunk_size,
    save_path,
    device="cuda",
    batch_size=8,
    num_workers=4,
):
    """
    Computes or loads task-specific advantage thresholds (epsilon_l).
    Uses the pi0.6 N-step TD advantage:
    A_t = (R_t - V_t) - (R_{t+N} - V_{t+N})
    """

    if os.path.exists(save_path):
        print(f"Loading advantage thresholds from {save_path}")
        with open(save_path, "r", encoding="utf-8") as f:
            str_keys = json.load(f)
            return {int(k): v for k, v in str_keys.items()}

    print("Computing V(s_t) for all frames to determine thresholds...")
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    pre_critic = critic_model.get_pre_processor(dataset)
    all_vs_list = []
    all_tasks_list = []
    all_episodes_list = []
    all_frames_list = []

    critic_model.eval()
    # Move critic to correct device just in case
    critic_model.to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting V(s)"):
            # Move relevant parts to device if not already
            critic_batch = pre_critic(batch)
            for k, v in critic_batch.items():
                if isinstance(v, torch.Tensor):
                    critic_batch[k] = v.to(device)
                elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                    critic_batch[k] = [t.to(device) for t in v]

            _, probs = critic_model(critic_batch)
            v_s = (probs * support).sum(dim=-1).cpu().numpy()

            def to_numpy(val):
                if hasattr(val, "cpu"):
                    val = val.cpu()
                if hasattr(val, "numpy"):
                    return val.numpy()
                return np.array(val)

            all_vs_list.append(v_s)
            all_tasks_list.append(to_numpy(batch["task_index"]))
            all_episodes_list.append(to_numpy(batch["episode_index"]))
            all_frames_list.append(to_numpy(batch["frame_index"]))

    all_vs = np.concatenate(all_vs_list)
    all_tasks = np.concatenate(all_tasks_list)
    all_episodes = np.concatenate(all_episodes_list)
    all_frames = np.concatenate(all_frames_list)

    ep_lengths = get_episode_lengths(dataset)
    max_lengths = get_max_task_lengths(dataset)

    # Load success flags for return calculation
    if "success" in dataset.meta.episodes.column_names:
        success_col = dataset.meta.episodes["success"]
        success_list = success_col.to_pylist() if hasattr(success_col, "to_pylist") else list(success_col)
        success_flags = torch.tensor(success_list, dtype=torch.bool)
    else:
        success_flags = None

    # Precompute actual normalized returns for all frames (vectorized)
    all_returns = calculate_returns_fn(
        ep_lengths,
        max_lengths,
        torch.tensor(all_tasks.astype(np.int64)),
        torch.tensor(all_episodes.astype(np.int64)),
        torch.tensor(all_frames.astype(np.int64)),
        success_flags=success_flags,
    ).numpy()

    # Compute N-step TD advantages: A_t = (R_t - V_t) - (R_{t+N} - V_{t+N})
    advantages = np.zeros(len(dataset), dtype=np.float32)
    ep_lengths_np = ep_lengths.numpy()
    for i in range(len(dataset)):
        ep_idx = int(all_episodes[i])
        frame_idx = int(all_frames[i])
        ep_len = ep_lengths_np[ep_idx]

        v_current = all_vs[i]
        r_current = all_returns[i]
        mc_error_current = r_current - v_current

        future_frame = frame_idx + chunk_size
        if future_frame >= ep_len:
            # Past episode end: MC advantage
            advantages[i] = mc_error_current
        else:
            # N-step TD advantage
            future_idx = i + chunk_size
            mc_error_future = all_returns[future_idx] - all_vs[future_idx]
            advantages[i] = mc_error_current - mc_error_future

    # 3. Calculate 30th percentile per task
    task_thresholds = {}
    unique_tasks = np.unique(all_tasks)
    for t in unique_tasks:
        task_advs = advantages[all_tasks == t]
        threshold = np.percentile(task_advs, 30)
        task_thresholds[int(t)] = float(threshold)

    # Save to JSON
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(task_thresholds, f, indent=2)

    print(f"Saved thresholds to {save_path}")
    return task_thresholds
