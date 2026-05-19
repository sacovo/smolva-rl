import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from lerobot_policy_smolvla_rl.ds_utils import get_episode_lengths


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
            future_item = item  # Dummy to keep shapes consistent

        # Prefix future item keys
        for k, v in future_item.items():
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


def compute_temporal_advantage(
    critic, pre_critic, batch, future_batch, support, has_future
):
    """
    Computes A(s_t) = V(s_{t+chunk_size}) - V(s_t) on the fly.
    If not has_future, V(s_{t+chunk_size}) = 0.0.
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

    advantage = v_s_future - v_s
    return advantage, v_s, v_s_future


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
    """

    if os.path.exists(save_path):
        print(f"Loading advantage thresholds from {save_path}")
        with open(save_path, "r") as f:
            str_keys = json.load(f)
            return {int(k): v for k, v in str_keys.items()}

    print("Computing V(s_t) for all frames to determine thresholds...")
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    pre_critic = critic_model.get_pre_processor(dataset)
    all_vs = np.zeros(len(dataset), dtype=np.float32)
    all_tasks = np.zeros(len(dataset), dtype=np.int32)
    all_episodes = np.zeros(len(dataset), dtype=np.int32)
    all_frames = np.zeros(len(dataset), dtype=np.int32)

    critic_model.eval()
    # Move critic to correct device just in case
    critic_model.to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting V(s)"):
            # Move relevant parts to device if not already
            # pre_critic usually handles device transfer or we assume batch is on CPU and handled by model inside
            # Wait, pre_critic returns a dictionary that might need device transfer.
            critic_batch = pre_critic(batch)
            # Make sure critic_batch is on device
            for k, v in critic_batch.items():
                if isinstance(v, torch.Tensor):
                    critic_batch[k] = v.to(device)
                elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                    critic_batch[k] = [t.to(device) for t in v]

            _, probs = critic_model(critic_batch)
            v_s = (probs * support).sum(dim=-1).cpu().numpy()

            indices = batch["index"].numpy()
            all_vs[indices] = v_s
            all_tasks[indices] = batch["task_index"].numpy()
            all_episodes[indices] = batch["episode_index"].numpy()
            all_frames[indices] = batch["frame_index"].numpy()

    ep_lengths = get_episode_lengths(dataset).numpy()

    advantages = np.zeros(len(dataset), dtype=np.float32)
    for i in range(len(dataset)):
        ep_idx = all_episodes[i]
        frame_idx = all_frames[i]
        ep_len = ep_lengths[ep_idx]

        v_current = all_vs[i]

        future_frame = frame_idx + chunk_size
        if future_frame >= ep_len:
            v_future = 0.0
        else:
            # We assume indices within the same episode are contiguous.
            v_future = all_vs[i + chunk_size]

        advantages[i] = v_future - v_current

    # 3. Calculate 30th percentile per task
    task_thresholds = {}
    unique_tasks = np.unique(all_tasks)
    for t in unique_tasks:
        task_advs = advantages[all_tasks == t]
        threshold = np.percentile(task_advs, 30)
        task_thresholds[int(t)] = float(threshold)

    # Save to JSON
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(task_thresholds, f, indent=2)

    print(f"Saved thresholds to {save_path}")
    return task_thresholds
