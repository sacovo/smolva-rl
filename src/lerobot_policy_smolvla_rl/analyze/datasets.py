import json
import numpy as np
from dataclasses import dataclass, field
from huggingface_hub import hf_hub_download


@dataclass
class DatasetSummary:
    repo_id: str
    action_shape: list[int] | None = None
    state_shape: list[int] | None = None
    feature_dtypes: dict[str, str] = field(default_factory=dict)
    action_stats: dict[str, list[float]] | None = None
    episode_count: int | None = None
    fps: int | None = None
    camera_keys: list[str] = field(default_factory=list)
    episode_lengths: dict[str, float] | None = None  # min, max, mean, total


def dataset_summary(repo_id: str, *, metadata_only: bool = True) -> DatasetSummary:
    """Summarizes HF dataset metadata (shapes, stats, counts, camera keys, episode lengths)."""
    # 1. Download info.json
    try:
        info_path = hf_hub_download(
            repo_id=repo_id, repo_type="dataset", filename="meta/info.json"
        )
        with open(info_path, "r") as f:
            info = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Could not load metadata info.json for '{repo_id}': {e}"
        ) from e

    features = info.get("features", {})
    action_shape = features.get("action", {}).get("shape")
    state_shape = features.get("observation.state", {}).get("shape")

    feature_dtypes = {}
    camera_keys = []
    for k, v in features.items():
        feature_dtypes[k] = v.get("dtype")
        if v.get("type") == "image" or k.startswith("observation.images"):
            camera_keys.append(k)

    episode_count = info.get("total_episodes")
    fps = info.get("fps")

    # 2. Download stats.json if available
    action_stats = None
    try:
        stats_path = hf_hub_download(
            repo_id=repo_id, repo_type="dataset", filename="meta/stats.json"
        )
        with open(stats_path, "r") as f:
            stats = json.load(f)
        action_stats = stats.get("action")
    except Exception as e:
        import logging

        logging.info(
            f"Could not load action stats from meta/stats.json for {repo_id}: {e}"
        )

    episode_lengths = None

    # 3. Load full dataset using LeRobotDataset if metadata_only=False
    if not metadata_only:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot_policy_smolvla_rl.ds_utils import get_episode_lengths

        try:
            dataset = LeRobotDataset(repo_id)
            lengths = get_episode_lengths(dataset)
            if len(lengths) > 0:
                episode_lengths = {
                    "min": float(np.min(lengths)),
                    "max": float(np.max(lengths)),
                    "mean": float(np.mean(lengths)),
                    "total": int(np.sum(lengths)),
                }
            episode_count = dataset.num_episodes
            fps = dataset.fps
        except Exception as e:
            raise RuntimeError(
                f"Could not load LeRobotDataset for '{repo_id}': {e}"
            ) from e

    return DatasetSummary(
        repo_id=repo_id,
        action_shape=action_shape,
        state_shape=state_shape,
        feature_dtypes=feature_dtypes,
        action_stats=action_stats,
        episode_count=episode_count,
        fps=fps,
        camera_keys=camera_keys,
        episode_lengths=episode_lengths,
    )
