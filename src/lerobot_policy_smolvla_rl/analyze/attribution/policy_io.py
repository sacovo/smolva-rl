import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator


@dataclass
class PrefixLayout:
    image_spans: dict[str, range]  # camera key -> token range
    lang_span: range
    state_span: range


def camera_keys(policy) -> list[str]:
    """Extracts camera image observation keys from policy configuration."""
    image_keys = (
        list(policy.config.image_features.keys())
        if hasattr(policy.config, "image_features")
        else []
    )
    if not image_keys and hasattr(policy.config, "input_features"):
        image_keys = [
            k
            for k, v in policy.config.input_features.items()
            if getattr(v, "type", None) == "image"
        ]
    return image_keys


def load_recap_policy(checkpoint: Path, device: str):
    """Loads a ReCap policy from an exported folder or a full checkpoint."""
    from lerobot_policy_smolvla_rl.modeling_smolvla_recap import SmolVLARECAPPolicy
    from lerobot.configs.policies import PreTrainedConfig

    checkpoint = Path(checkpoint)

    # Check if checkpoint is an exported folder (has config.json)
    if checkpoint.is_dir() and (checkpoint / "config.json").exists():
        policy = SmolVLARECAPPolicy.from_pretrained(str(checkpoint))
    else:
        # Check for config.json in folder/parent folder
        config_path = None
        if checkpoint.is_file():
            config_path = checkpoint.parent / "config.json"
        elif checkpoint.is_dir():
            config_path = checkpoint / "config.json"

        if config_path and config_path.exists():
            config = PreTrainedConfig.from_pretrained(str(config_path.parent))
            policy = SmolVLARECAPPolicy(config)
            # Load state dict
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if "model" in state_dict:
                policy.model.load_state_dict(state_dict["model"])
            else:
                policy.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(
                f"Could not load checkpoint from '{checkpoint}'. No config.json found."
            )

    policy.eval()
    policy.to(device)
    return policy


def prefix_layout(policy, lang_len: int | None = None) -> PrefixLayout:
    """Computes the prefix layout (image spans, lang span, state span) of the policy.

    Note: Assumes exactly one state token (as state is projected into a single vector token).
    """
    device = next(policy.parameters()).device

    # 1. Determine number of image tokens dynamically
    resize_imgs = getattr(policy.config, "resize_imgs_with_padding", (512, 512))
    dummy_img = torch.zeros(1, 3, *resize_imgs, device=device)

    try:
        with torch.no_grad():
            num_img_tokens = policy.model.vlm_with_expert.embed_image(dummy_img).shape[
                1
            ]
    except Exception as e:
        import logging

        logging.warning(
            f"Could not dynamically query num_img_tokens from policy: {e}. Falling back to 64."
        )
        num_img_tokens = 64

    # 2. Iterate cameras
    image_keys = camera_keys(policy)

    curr_idx = 0
    image_spans = {}
    add_special = getattr(policy.model, "add_image_special_tokens", False)

    for key in image_keys:
        start_idx = curr_idx
        if add_special:
            curr_idx += 1  # start token
        curr_idx += num_img_tokens
        if add_special:
            curr_idx += 1  # end token
        image_spans[key] = range(start_idx, curr_idx)

    # 3. Language span
    if lang_len is None:
        lang_len = getattr(policy.config, "tokenizer_max_length", 48)

    lang_start = curr_idx
    lang_end = lang_start + lang_len
    lang_span = range(lang_start, lang_end)

    # 4. State span
    state_start = lang_end
    state_end = state_start + 1
    state_span = range(state_start, state_end)

    return PrefixLayout(
        image_spans=image_spans, lang_span=lang_span, state_span=state_span
    )


def iter_eval_batches(
    dataset_repo_id: str,
    policy,
    *,
    episodes: list[int] | None = None,
    stride: int = 1,
    batch_size: int = 8,
) -> Iterator[dict]:
    """Iterates through LeRobot dataset and yields preprocessed and original batches."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from lerobot_policy_smolvla_rl.analyze.datasets import dataset_summary

    # Dynamically extract FPS from dataset metadata to avoid hardcoding 10 FPS
    try:
        summary = dataset_summary(dataset_repo_id, metadata_only=True)
        fps = summary.fps if (summary and summary.fps) else 10.0
    except Exception:
        fps = 10.0

    delta_timestamps = {
        "action": [i / float(fps) for i in range(policy.config.chunk_size)],
    }
    dataset = LeRobotDataset(
        repo_id=dataset_repo_id, episodes=episodes, delta_timestamps=delta_timestamps
    )

    pre_recap = policy.model.get_pre_processor(dataset)
    indices = list(range(0, len(dataset), stride))
    device = next(policy.parameters()).device

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        items = [dataset[idx] for idx in batch_indices]

        batch = {}
        for key in items[0].keys():
            val = items[0][key]
            if isinstance(val, torch.Tensor):
                batch[key] = torch.stack([item[key] for item in items], dim=0)
            else:
                batch[key] = [item[key] for item in items]

        # Preprocess
        recap_batch = pre_recap(batch)
        recap_batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in recap_batch.items()
        }
        batch_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Add language tokens automatically if not present in recap_batch
        if "observation.language.tokens" not in recap_batch:
            tokenizer = policy.model.vlm_with_expert.processor.tokenizer
            prompts = []
            for item in items:
                task_instruction = item.get("task", "do the task")
                prompts.append(f"User: {task_instruction}\nAssistant: ")
            tokens = tokenizer(
                prompts, return_tensors="pt", padding=True, add_special_tokens=True
            )
            recap_batch["observation.language.tokens"] = tokens["input_ids"].to(device)
            recap_batch["observation.language.attention_mask"] = (
                tokens["attention_mask"].bool().to(device)
            )

        yield {"recap_batch": recap_batch, "batch": batch_dev}
