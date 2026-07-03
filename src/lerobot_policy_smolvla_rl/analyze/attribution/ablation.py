import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from lerobot_policy_smolvla_rl.analyze.attribution.policy_io import (
    iter_eval_batches,
    camera_keys,
)


@dataclass
class AblationResult:
    per_task: pd.DataFrame  # task, ablated_camera ("none"|key), action_mse, policy_divergence_mse, per_dim_mse (list), n_frames


def run_camera_ablation(
    policy,
    dataset_repo_id: str,
    *,
    cameras: list[str] | None = None,
    episodes: list[int] | None = None,
    noise_seed: int = 0,
) -> AblationResult:
    """Runs camera ablation, comparing predictions with and without specific camera inputs."""
    # Resolve cameras
    image_keys = camera_keys(policy)
    if cameras is None:
        cameras = image_keys

    device = next(policy.parameters()).device

    # Set seed
    torch.manual_seed(noise_seed)
    np.random.seed(noise_seed)

    task_stats = {}

    batches = list(
        iter_eval_batches(
            dataset_repo_id, policy, episodes=episodes, stride=1, batch_size=1
        )
    )

    for batch_dict in tqdm(batches, desc="Running camera ablation"):
        recap_batch = batch_dict["recap_batch"]
        batch = batch_dict["batch"]

        task_instruction = batch["task"][0] if "task" in batch else "default_task"
        if isinstance(task_instruction, bytes):
            task_instruction = task_instruction.decode("utf-8")

        if task_instruction not in task_stats:
            task_stats[task_instruction] = {}
            for cam in ["none"] + cameras:
                task_stats[task_instruction][cam] = {
                    "errors_gt": [],
                    "errors_full": [],
                    "per_dim_errors_gt": [],
                }

        # Prepare inputs
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)

        dtype = policy.model.vlm_with_expert.vlm.dtype
        images = [img.to(dtype) for img in images]
        state = state.to(dtype)

        lang_tokens = recap_batch["observation.language.tokens"]
        lang_masks = recap_batch["observation.language.attention_mask"]

        gt_action = batch["action"].to(device)
        action_dim = policy.config.output_features["action"].shape[0]
        gt_action = gt_action[:, :, :action_dim]

        # Sample noise (seeded)
        actions_shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
        noise = policy.model.sample_noise(actions_shape, device)

        # 1. Full prediction
        with torch.no_grad():
            pred_full = policy.model.sample_actions(
                images=images,
                img_masks=img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                state=state,
                noise=noise,
            )
            pred_full = pred_full[:, :, :action_dim]

            err_gt_none = F.mse_loss(pred_full, gt_action, reduction="none")
            task_stats[task_instruction]["none"]["errors_gt"].append(
                err_gt_none.mean().item()
            )
            task_stats[task_instruction]["none"]["errors_full"].append(0.0)
            task_stats[task_instruction]["none"]["per_dim_errors_gt"].append(
                err_gt_none.mean(dim=(0, 1)).cpu().numpy()
            )

            # 2. Ablate per camera
            for cam_key in cameras:
                if cam_key not in image_keys:
                    continue
                actual_idx = image_keys.index(cam_key)

                ablated_masks = [mask.clone() for mask in img_masks]
                ablated_masks[actual_idx] = torch.zeros_like(ablated_masks[actual_idx])

                pred_ablated = policy.model.sample_actions(
                    images=images,
                    img_masks=ablated_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    state=state,
                    noise=noise,
                )
                pred_ablated = pred_ablated[:, :, :action_dim]

                err_gt = F.mse_loss(pred_ablated, gt_action, reduction="none")
                err_full = F.mse_loss(pred_ablated, pred_full, reduction="none")

                task_stats[task_instruction][cam_key]["errors_gt"].append(
                    err_gt.mean().item()
                )
                task_stats[task_instruction][cam_key]["errors_full"].append(
                    err_full.mean().item()
                )
                task_stats[task_instruction][cam_key]["per_dim_errors_gt"].append(
                    err_gt.mean(dim=(0, 1)).cpu().numpy()
                )

    # Format as DataFrame
    rows = []
    for task_name, cams in task_stats.items():
        for cam_key, stats in cams.items():
            n_frames = len(stats["errors_gt"])
            if n_frames == 0:
                continue
            mean_mse_gt = float(np.mean(stats["errors_gt"]))
            mean_mse_full = float(np.mean(stats["errors_full"]))
            mean_per_dim = np.mean(stats["per_dim_errors_gt"], axis=0).tolist()

            rows.append(
                {
                    "task": task_name,
                    "ablated_camera": cam_key,
                    "action_mse": mean_mse_gt,
                    "policy_divergence_mse": mean_mse_full,
                    "per_dim_mse": mean_per_dim,
                    "n_frames": n_frames,
                }
            )

    return AblationResult(per_task=pd.DataFrame(rows))
