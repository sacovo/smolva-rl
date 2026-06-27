"""Advantage computation for RECAP (single source of truth).

Implements the N-step TD advantage from the RECAP paper (arXiv:2511.14759,
Appendix A-F):

    A(o_t) = sum_{t'=t}^{t+N-1} r_{t'} + V(o_{t+N}) - V(o_t)

Rewritten with the normalized per-task returns R_t = sum_{t'=t}^{T} r_{t'}
(see ``ds_utils.calculate_returns``):

    A(o_t) = (R_t - V(o_t)) - (R_{t+N} - V(o_{t+N}))

When t+N reaches the end of the episode the future terms vanish and the
expression degenerates to the Monte-Carlo advantage R_t - V(o_t).  Setting the
horizon N >= the longest episode therefore reproduces the pre-training
(full Monte-Carlo) regime of the paper, while N = 50 reproduces the
task-specific post-training regime.

This module exposes two pure, easily testable helpers
(``nstep_td_advantages`` and ``task_thresholds_from_advantages``) plus a single
orchestrator (``precompute_advantages_and_thresholds``) that runs the critic and
ties them together.  All advantage computation in the project flows through
here so the formula can never drift between the offline script and the trainer.
"""

import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot_policy_smolvla_rl.dataloader_utils import RobustDataset
from lerobot_policy_smolvla_rl.ds_utils import (
    calculate_returns,
    get_episode_lengths,
    get_max_task_lengths,
    load_success_flags,
)


def _to_numpy(val):
    if hasattr(val, "cpu"):
        val = val.cpu()
    if hasattr(val, "numpy"):
        return val.numpy()
    return np.array(val)


# pylint: disable=too-many-arguments,too-many-positional-arguments
def nstep_td_advantages(
    vs_by_idx,
    returns_by_idx,
    meta_by_idx,
    ep_from,
    episode_lengths,
    advantage_horizon,
    size=None,
):
    """Vectorized-per-frame N-step TD advantage A_t = (R_t - V_t) - (R_{t+N} - V_{t+N}).

    Everything is keyed by the dataset's absolute frame index so that skipped
    (corrupt) samples simply have no entry and end up as NaN.

    Parameters
    ----------
    vs_by_idx, returns_by_idx:
        dict[abs_index -> float] of V(o_t) and normalized return R_t.
    meta_by_idx:
        dict[abs_index -> (episode_index, frame_index, task_index)].
    ep_from:
        array where ``ep_from[ep]`` is the absolute index of the first frame of
        episode ``ep`` (``dataset.meta.episodes["dataset_from_index"]``).
    episode_lengths:
        array indexed by episode index.
    advantage_horizon:
        N, the lookahead.  Use a value >= the longest episode for the
        Monte-Carlo (pre-training) regime.
    size:
        Length of the returned dense array.  Defaults to ``max(index) + 1``.

    Returns
    -------
    np.ndarray[size] float32, NaN where the frame or its future frame is
    corrupt/unreachable.
    """
    if size is None:
        size = (max(meta_by_idx) + 1) if meta_by_idx else 0
    advantages = np.full(size, np.nan, dtype=np.float32)

    for idx, v_t in vs_by_idx.items():
        ep_idx, frame_idx, _ = meta_by_idx[idx]
        mc_error_current = returns_by_idx[idx] - v_t

        future_frame = frame_idx + advantage_horizon
        if future_frame >= episode_lengths[ep_idx]:
            # Past episode end: future terms are zero -> Monte-Carlo advantage.
            advantages[idx] = mc_error_current
        else:
            future_idx = int(ep_from[ep_idx]) + future_frame
            if future_idx in vs_by_idx:
                mc_error_future = returns_by_idx[future_idx] - vs_by_idx[future_idx]
                advantages[idx] = mc_error_current - mc_error_future
            # else: future frame corrupt/unreachable -> leave NaN.

    return advantages


def task_thresholds_from_advantages(
    advantages, meta_by_idx, success_flags=None, percentile=30
):
    """Per-task advantage threshold ε_ℓ as a percentile of positive-advantage data.

    Failed episodes and NaN (corrupt) frames are excluded.  A task with no
    valid frames defaults to a threshold of 0.0.
    """
    valid_tasks = []
    valid_advs = []
    all_tasks = []
    for idx in meta_by_idx:
        ep_idx, _, task_idx = meta_by_idx[idx]
        all_tasks.append(task_idx)
        is_success = success_flags is None or bool(success_flags[ep_idx].item())
        if is_success and not np.isnan(advantages[idx]):
            valid_tasks.append(task_idx)
            valid_advs.append(advantages[idx])

    valid_tasks = np.array(valid_tasks, dtype=np.int64)
    valid_advs = np.array(valid_advs, dtype=np.float32)

    task_thresholds = {}
    for t in np.unique(np.array(all_tasks, dtype=np.int64)):
        task_advs = valid_advs[valid_tasks == t]
        if len(task_advs) == 0:
            task_thresholds[int(t)] = 0.0
        else:
            task_thresholds[int(t)] = float(np.percentile(task_advs, percentile))
    return task_thresholds


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def precompute_advantages_and_thresholds(
    critic,
    dataset,
    *,
    advantage_horizon,
    device="cuda",
    batch_size=8,
    num_workers=0,
    skip_bad_samples=True,
    percentile=30,
    delay=0.0,
):
    """Run the critic over the whole dataset and compute N-step TD advantages.

    Returns ``(advantages, task_thresholds)`` where ``advantages`` is a dense
    float32 array indexed by absolute frame index (NaN for corrupt/unreachable
    frames) and ``task_thresholds`` maps task_index -> ε_ℓ.
    """
    critic.eval()
    critic.to(device)

    support = torch.linspace(
        critic.config.vmin,
        critic.config.vmax,
        critic.config.num_bins,
        device=device,
    )
    pre_critic = critic.get_pre_processor(dataset)

    episode_lengths = get_episode_lengths(dataset)
    max_lengths = get_max_task_lengths(dataset)
    ep_from = np.array(dataset.meta.episodes["dataset_from_index"])
    success_flags = load_success_flags(dataset)

    loader_dataset = RobustDataset(dataset) if skip_bad_samples else dataset
    dataloader = DataLoader(
        loader_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # abs_index -> V(o_t) and abs_index -> (episode, frame, task)
    vs_by_idx: dict[int, float] = {}
    meta_by_idx: dict[int, tuple[int, int, int]] = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting V(s)"):
            critic_batch = pre_critic(batch)
            for k, v in critic_batch.items():
                if isinstance(v, torch.Tensor):
                    critic_batch[k] = v.to(device)
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    critic_batch[k] = [t.to(device) for t in v]

            _, probs = critic(critic_batch)
            v_s = (probs * support).sum(dim=-1).cpu().numpy()

            sample_indices = _to_numpy(batch["index"])
            episode_indices = _to_numpy(batch["episode_index"])
            frame_indices = _to_numpy(batch["frame_index"])
            task_indices = _to_numpy(batch["task_index"])

            for idx, v, ep, fr, task in zip(
                sample_indices, v_s, episode_indices, frame_indices, task_indices
            ):
                vs_by_idx[int(idx)] = float(v)
                meta_by_idx[int(idx)] = (int(ep), int(fr), int(task))

            if delay > 0:
                time.sleep(delay)

    n_seen = len(vs_by_idx)
    n_skipped = len(dataset) - n_seen
    if n_skipped > 0:
        print(
            f"  {n_skipped} / {len(dataset)} samples skipped (corrupt). "
            "Their advantages stay NaN (treated as neutral 0 during training)."
        )

    # Normalized returns R_t for every frame that has metadata.
    sorted_indices = sorted(meta_by_idx.keys())
    all_ep_idxs = torch.tensor([meta_by_idx[i][0] for i in sorted_indices], dtype=torch.long)
    all_frame_idxs = torch.tensor([meta_by_idx[i][1] for i in sorted_indices], dtype=torch.long)
    all_task_idxs = torch.tensor([meta_by_idx[i][2] for i in sorted_indices], dtype=torch.long)
    all_returns = calculate_returns(
        episode_lengths,
        max_lengths,
        all_task_idxs,
        all_ep_idxs,
        all_frame_idxs,
        success_flags=success_flags,
    )
    returns_by_idx = {idx: float(r) for idx, r in zip(sorted_indices, all_returns)}

    size = getattr(dataset.meta, "total_frames", None)
    if size is None:
        size = (max(meta_by_idx) + 1) if meta_by_idx else len(dataset)

    advantages = nstep_td_advantages(
        vs_by_idx,
        returns_by_idx,
        meta_by_idx,
        ep_from,
        episode_lengths.numpy(),
        advantage_horizon,
        size=int(size),
    )

    task_thresholds = task_thresholds_from_advantages(
        advantages, meta_by_idx, success_flags=success_flags, percentile=percentile
    )

    return advantages, task_thresholds


def save_advantages_and_thresholds(
    advantages, task_thresholds, advantages_path, thresholds_path
):
    """Persist the advantage array (.npy) and per-task thresholds (.json)."""
    os.makedirs(os.path.dirname(advantages_path), exist_ok=True)
    np.save(advantages_path, advantages)
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(task_thresholds, f, indent=2)


def load_thresholds(thresholds_path):
    """Load per-task thresholds from JSON with integer task keys."""
    with open(thresholds_path, "r", encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}
