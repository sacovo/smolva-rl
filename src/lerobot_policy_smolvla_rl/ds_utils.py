from collections import defaultdict
from itertools import batched

import torch

import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset


C_FAIL = 10000


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


def get_episode_lengths(dataset: LeRobotDataset):
    return torch.tensor(dataset.meta.episodes["length"])


# pylint: disable=too-many-arguments,too-many-positional-arguments
def calculate_returns(
    episode_lengths,
    max_lengths,
    task_idxs,
    episode_idxs,
    frame_idxs,
    post_goal_buffer=0,
    success_flags=None,
):
    if not isinstance(episode_idxs, torch.Tensor):
        episode_idxs = torch.tensor(episode_idxs, dtype=torch.long)
    if not isinstance(frame_idxs, torch.Tensor):
        frame_idxs = torch.tensor(frame_idxs, dtype=torch.long)
    if not isinstance(task_idxs, torch.Tensor):
        task_idxs = torch.tensor(task_idxs, dtype=torch.long)

    T = episode_lengths[episode_idxs]  # Shape: [batch_size]
    # Remaining steps until goal. Goal is shifted by post_goal_buffer.
    rem_steps = torch.clamp(T - post_goal_buffer - frame_idxs - 1, min=0).float()

    if success_flags is not None:
        batch_success = success_flags[episode_idxs]
        rem_steps = torch.where(batch_success, rem_steps, rem_steps + C_FAIL)

    max_lens = max_lengths[task_idxs]  # Max lengths for the tasks in the batch
    returns = -(rem_steps / max_lens)  # Normalize values between (-1, 0)
    return returns


# pylint: disable=too-many-locals
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


def calculate_ds_returns(dataset: LeRobotDataset, batch_size=1024):
    episode_lengths = get_episode_lengths(dataset)
    max_lengths = get_max_task_lengths(dataset)

    if "success" in dataset.meta.episodes.column_names:
        success_list = dataset.meta.episodes["success"].to_pylist()
        success_flags = torch.tensor(success_list, dtype=torch.bool)
    else:
        success_flags = None

    returns = []

    for batch in batched(dataset, n=batch_size):  # type: ignore
        task_idxs = [int(frame["task_index"]) for frame in batch]
        episode_idxs = [int(frame["episode_index"]) for frame in batch]
        frame_idxs = torch.tensor([int(frame["frame_index"]) for frame in batch])

        returns.append(
            calculate_returns(
                episode_lengths,
                max_lengths,
                task_idxs,
                episode_idxs,
                frame_idxs,
                success_flags=success_flags,
            )
        )

    return torch.cat(returns)


def calculate_advantage(expected_return, actual_return, advantage_threshold=0.0):
    """
    Calculates the advantage and the boolean mask of positive advantages.
    Returns:
        advantage: A tensor representing the difference (expected - actual)
        advantage_bool: A list of booleans indicating if the advantage is positive
    """
    advantage = expected_return - actual_return
    advantage_bool = (advantage > advantage_threshold).tolist()
    return advantage, advantage_bool
