import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from lerobot_policy_smolvla_rl.analyze.attribution.ablation import AblationResult


@dataclass
class AttributionScores:
    scores: dict[str, torch.Tensor]  # camera key -> (batch, 64) scores
    tasks: list[str]  # list of tasks matching the batch sequence


def importance_table(results, reduction: str = "mean") -> pd.DataFrame:
    """Aggregates importance scores or ablation results per task and camera."""
    if isinstance(results, AblationResult):
        return results.per_task

    # Else it is AttributionScores or a generic dict/dataframe
    if hasattr(results, "scores") and hasattr(results, "tasks"):
        rows = []
        tasks = results.tasks
        unique_tasks = list(set(tasks))

        for task in unique_tasks:
            task_indices = [i for i, t in enumerate(tasks) if t == task]
            if not task_indices:
                continue

            for cam, score_tensor in results.scores.items():
                task_scores = score_tensor[task_indices]
                if reduction == "sum":
                    mean_score = float(task_scores.sum(dim=-1).mean().item())
                else:
                    mean_score = float(task_scores.mean().item())
                rows.append(
                    {"task": task, "camera": cam, "importance_score": mean_score}
                )
        return pd.DataFrame(rows)

    raise TypeError("Unsupported results type for importance_table.")


def saliency_overlay(image: torch.Tensor, scores_64: torch.Tensor) -> plt.Figure:
    """Upsamples 8x8 score grid to match image dimensions and alpha-blends it over the image."""
    import torch.nn.functional as F

    # 1. Normalize image to [0, 1] range and shape (H, W, 3)
    if image.min() < 0:
        image = (image + 1.0) / 2.0
    image = image.clamp(0.0, 1.0)

    if image.ndim == 4:
        image = image[0]  # Take first batch element

    if image.shape[0] == 3:  # (3, H, W)
        image_np = image.permute(1, 2, 0).cpu().numpy()
    else:
        image_np = image.cpu().numpy()

    H, W = image_np.shape[0], image_np.shape[1]

    # 2. Reshape and upsample scores
    L = scores_64.numel()
    grid_size = int(np.sqrt(L))
    scores_tensor = scores_64.detach().float().cpu().reshape(1, 1, grid_size, grid_size)
    s_min, s_max = scores_tensor.min(), scores_tensor.max()
    if s_max > s_min:
        scores_tensor = (scores_tensor - s_min) / (s_max - s_min)

    upsampled = F.interpolate(
        scores_tensor, size=(H, W), mode="bilinear", align_corners=False
    )
    upsampled_np = upsampled.squeeze().numpy()

    # 3. Plot
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(image_np)
    ax.imshow(upsampled_np, cmap="jet", alpha=0.5)
    ax.axis("off")
    plt.tight_layout()
    return fig


def method_agreement(
    ablation: pd.DataFrame, attention: pd.DataFrame, gradients: pd.DataFrame
) -> pd.DataFrame:
    """Computes Spearman rank correlation between the three attribution methods, per task."""
    import scipy.stats as stats

    # Clean and standardize ablation
    ab = ablation.copy()
    if "ablated_camera" in ab.columns:
        ab = ab[ab["ablated_camera"] != "none"]
        ab = ab.rename(columns={"ablated_camera": "camera"})
    # Use policy_divergence_mse or action_mse
    ab["ablation_score"] = (
        ab["policy_divergence_mse"]
        if "policy_divergence_mse" in ab.columns
        else ab["action_mse"]
    )
    ab = ab[["task", "camera", "ablation_score"]]

    # Clean and standardize attention
    at = attention.copy()
    if "importance_score" in at.columns:
        at = at.rename(columns={"importance_score": "attention_score"})
    else:
        at = at.rename(columns={at.columns[2]: "attention_score"})
    at = at[["task", "camera", "attention_score"]]

    # Clean and standardize gradients
    gr = gradients.copy()
    if "importance_score" in gr.columns:
        gr = gr.rename(columns={"importance_score": "gradient_score"})
    else:
        gr = gr.rename(columns={gr.columns[2]: "gradient_score"})
    gr = gr[["task", "camera", "gradient_score"]]

    # Merge
    df = ab.merge(at, on=["task", "camera"], how="inner")
    df = df.merge(gr, on=["task", "camera"], how="inner")

    if df.empty:
        return pd.DataFrame(
            columns=[
                "task",
                "spearman_ablation_vs_attention",
                "spearman_ablation_vs_gradient",
                "spearman_attention_vs_gradient",
            ]
        )

    results = []
    for task_name, group in df.groupby("task"):
        if len(group) < 2:
            continue

        corr_ab_at, _ = stats.spearmanr(
            group["ablation_score"], group["attention_score"]
        )
        corr_ab_gr, _ = stats.spearmanr(
            group["ablation_score"], group["gradient_score"]
        )
        corr_at_gr, _ = stats.spearmanr(
            group["attention_score"], group["gradient_score"]
        )

        results.append(
            {
                "task": task_name,
                "spearman_ablation_vs_attention": corr_ab_at,
                "spearman_ablation_vs_gradient": corr_ab_gr,
                "spearman_attention_vs_gradient": corr_at_gr,
            }
        )

    return pd.DataFrame(results)
