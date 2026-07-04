import click
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from lerobot_policy_smolvla_rl.analyze.runs import (
    load_wandb_csv,
    fetch_wandb_run,
    find_wandb_runs,
)
from lerobot_policy_smolvla_rl.analyze.loss_curves import block_stats, summarize
from lerobot_policy_smolvla_rl.analyze.plotting import (
    plot_metric_comparison,
    save_figure,
    StepMarker,
)
from lerobot_policy_smolvla_rl.analyze.datasets import dataset_summary
from lerobot_policy_smolvla_rl.analyze.eval_results import collect_eval_results


def parse_episodes(episodes_str: str | None) -> list[int] | None:
    if not episodes_str:
        return None
    if ":" in episodes_str:
        start, end = map(int, episodes_str.split(":"))
        return list(range(start, end))
    return [int(x) for x in episodes_str.split(",")]


def parse_slice(slice_str: str | None) -> slice | None:
    if not slice_str:
        return None
    parts = slice_str.split(":")
    if len(parts) == 1:
        return slice(int(parts[0]))
    start = int(parts[0]) if parts[0] else None
    end = int(parts[1]) if parts[1] else None
    step = int(parts[2]) if len(parts) > 2 and parts[2] else None
    return slice(start, end, step)


@click.group()
def main():
    """CLI for the analyze module."""
    pass


@main.command("loss-summary")
@click.option(
    "--csv",
    required=True,
    type=click.Path(exists=True),
    help="Path to local WandB CSV export.",
)
@click.option("--run", required=True, help="Substring to match the run column name.")
@click.option(
    "--max-step", type=int, default=None, help="Maximum training step to analyze."
)
@click.option(
    "--block", type=int, default=500, help="Block size in steps for bin statistics."
)
@click.option(
    "--log-interval", type=int, default=10, help="Step log interval multiplier."
)
def cmd_loss_summary(csv, run, max_step, block, log_interval):
    """Computes and prints block stats and summary for a single run in a CSV."""
    run_data = load_wandb_csv(Path(csv), run, log_interval=log_interval)
    metric = [c for c in run_data.df.columns if c != "step"][0]

    stats_df = block_stats(run_data, metric, block=block, max_step=max_step)
    summary_dict = summarize(run_data, metric, max_step=max_step)

    click.echo(f"\n=================== Summary for {run_data.name} ===================")
    click.echo(f"Metric resolved: {metric}")

    total_pts = summary_dict.get("total_points")
    click.echo(f"Total logged points: {total_pts if total_pts is not None else 'N/A'}")

    def fmt_val(key):
        val = summary_dict.get(key)
        return f"{val:.4f}" if val is not None else "N/A"

    click.echo(f"Initial loss: {fmt_val('initial')}")
    if "initial_50_mean" in summary_dict:
        click.echo(
            f"Initial 50 steps mean: {fmt_val('initial_50_mean')} (std: {fmt_val('initial_50_std')})"
        )
    click.echo(f"Overall Mean: {fmt_val('mean')} (std: {fmt_val('std')})")
    click.echo(f"Min: {fmt_val('min')}")
    click.echo(f"Final tail mean: {fmt_val('tail_mean')}")
    click.echo(f"Final SMA ({metric}): {fmt_val('sma_at_end')}")

    click.echo("\n=================== Block Stats ===================")
    click.echo(stats_df.to_string(index=False))


@main.command("loss-compare")
@click.option(
    "--csv",
    multiple=True,
    required=True,
    type=click.Path(exists=True),
    help="Path to local WandB CSV export.",
)
@click.option(
    "--run",
    multiple=True,
    required=True,
    help="Substring to match the run column name.",
)
@click.option(
    "--max-step", type=int, default=None, help="Maximum training step to plot."
)
@click.option("--out", default="loss_comparison.png", help="Output filename.")
@click.option("--title", default=None, help="Plot title.")
@click.option(
    "--log-interval", type=int, default=10, help="Step log interval multiplier."
)
@click.option(
    "--step-offset", multiple=True, type=int, help="Step offset to apply to each run."
)
def cmd_loss_compare(csv, run, max_step, out, title, log_interval, step_offset):
    """Compares multiple runs from CSV files and plots them."""
    if len(csv) != len(run):
        raise click.BadParameter(
            "Number of --csv arguments must match the number of --run arguments."
        )

    runs = []
    offsets = list(step_offset) if step_offset else []
    # Fill remaining offsets with 0
    while len(offsets) < len(csv):
        offsets.append(0)

    for i, (csv_path, run_sub) in enumerate(zip(csv, run)):
        offset = offsets[i]
        run_data = load_wandb_csv(
            Path(csv_path), run_sub, log_interval=log_interval, step_offset=offset
        )
        runs.append(run_data)

    # Resolve metric: assume first non-step column of first run
    metric = [c for c in runs[0].df.columns if c != "step"][0]

    # Check if we should add resumed checkpoint line (if any offset is > 0)
    annotations = []
    for offset in offsets:
        if offset > 0:
            annotations.append(StepMarker(step=offset, label="Resumed Checkpoint"))

    fig = plot_metric_comparison(
        runs, metric, max_step=max_step, title=title, annotations=annotations
    )
    out_path = save_figure(fig, "loss-compare", out)
    click.echo(f"Saved comparison plot to: {out_path.absolute()}")


@main.command("wandb-runs")
@click.option(
    "--project", required=True, help="WandB project identifier (entity/project)."
)
@click.option(
    "--dataset-contains",
    default=None,
    help="Substring to match in dataset_repo_id config.",
)
@click.option("--name-contains", default=None, help="Substring to match in run name.")
def cmd_wandb_runs(project, dataset_contains, name_contains):
    """Searches and lists WandB runs matching query filters."""
    summaries = find_wandb_runs(
        project, dataset_contains=dataset_contains, name_contains=name_contains
    )
    if not summaries:
        click.echo(f"No runs found in '{project}' matching filters.")
        return

    rows = []
    for s in summaries:
        rows.append(
            {
                "id": s.id,
                "name": s.name,
                "dataset": s.config.get("dataset_repo_id", "Unknown"),
            }
        )
    df = pd.DataFrame(rows)
    click.echo(df.to_string(index=False))


@main.command("wandb-history")
@click.option(
    "--run",
    required=True,
    help="WandB run identifier (entity/project/run_id or project/run_id or run_id).",
)
@click.option("--keys", required=True, help="Comma-separated metric keys to fetch.")
@click.option("--out", default="wandb_history.csv", help="Output filename.")
def cmd_wandb_history(run, keys, out):
    """Fetches metric history for a run from WandB and saves it as a CSV."""
    key_list = [k.strip() for k in keys.split(",")]
    run_data = fetch_wandb_run(run, key_list)

    out_dir = Path("outputs/analysis/wandb-history")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out

    run_data.df.to_csv(out_path, index=False)
    click.echo(f"Saved history to: {out_path.absolute()}")
    click.echo("\nTail of history:")
    click.echo(run_data.df.tail(10).to_string(index=False))


@main.command("dataset")
@click.option("--repo-id", required=True, help="Hugging Face dataset repository ID.")
@click.option(
    "--full", is_flag=True, help="Load full dataset to calculate episode lengths."
)
def cmd_dataset(repo_id, full):
    """Inspects action/state shape, camera keys, and metadata of a LeRobot dataset."""
    summary = dataset_summary(repo_id, metadata_only=not full)

    click.echo(f"\n=================== Dataset: {summary.repo_id} ===================")
    click.echo(f"Action shape: {summary.action_shape}")
    click.echo(f"State shape: {summary.state_shape}")
    click.echo(f"Total episodes: {summary.episode_count}")
    click.echo(f"FPS: {summary.fps}")
    click.echo(f"Camera keys: {summary.camera_keys}")

    if summary.action_stats:
        click.echo("\nAction stats:")
        click.echo(f"  Min: {summary.action_stats.get('min')}")
        click.echo(f"  Max: {summary.action_stats.get('max')}")
        click.echo(f"  Mean: {summary.action_stats.get('mean')}")
        click.echo(f"  Std: {summary.action_stats.get('std')}")

    if summary.episode_lengths:
        click.echo("\nEpisode lengths (frames):")
        click.echo(f"  Min: {summary.episode_lengths['min']:.1f}")
        click.echo(f"  Max: {summary.episode_lengths['max']:.1f}")
        click.echo(f"  Mean: {summary.episode_lengths['mean']:.1f}")
        click.echo(f"  Total frames: {summary.episode_lengths['total']}")


@main.command("eval-results")
@click.option(
    "--eval-dir",
    default="outputs/eval",
    type=click.Path(exists=True),
    help="Directory to scan.",
)
@click.option("--pattern", default="*", help="Job pattern filter.")
@click.option(
    "--pivot", default=None, help="Comma-separated column names to pivot table on."
)
def cmd_eval_results(eval_dir, pattern, pivot):
    """Gathers and compiles evaluation results from eval_info.json files."""
    df = collect_eval_results(Path(eval_dir), pattern)
    if df.empty:
        click.echo("No evaluation results found.")
        return

    click.echo("\n=================== Flat Results ===================")
    columns_to_show = ["job_name", "timestamp", "pc_success", "n_episodes"]
    for extra in ["checkpoint", "suite", "cfg"]:
        if extra in df.columns:
            columns_to_show.append(extra)
    click.echo(df[columns_to_show].to_string(index=False))

    # Save flat results
    out_dir = Path(eval_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "sweep_summary.csv", index=False)
    click.echo(f"\nFlat results saved to: {(out_dir / 'sweep_summary.csv').absolute()}")

    if pivot:
        pivot_cols = [c.strip() for c in pivot.split(",")]
        # Map nice casing
        mapped_pivot_cols = []
        for c in pivot_cols:
            c_low = c.lower()
            if c_low == "checkpoint" and "checkpoint" in df.columns:
                mapped_pivot_cols.append("checkpoint")
            elif c_low == "suite" and "suite" in df.columns:
                mapped_pivot_cols.append("suite")
            elif (c_low == "cfg" or c_low == "cfg weight") and "cfg" in df.columns:
                mapped_pivot_cols.append("cfg")
            else:
                if c in df.columns:
                    mapped_pivot_cols.append(c)

        if len(mapped_pivot_cols) >= 2:
            try:
                # We need columns, index, and values
                # Assume last mapped_pivot_cols is column, others are index
                col = mapped_pivot_cols[-1]
                idx = mapped_pivot_cols[:-1]
                pivoted = df.pivot_table(
                    index=idx, columns=col, values="pc_success", aggfunc="mean"
                )
                pivoted["Average"] = pivoted.mean(axis=1)

                click.echo("\n=================== Pivoted Summary ===================")
                click.echo(pivoted.round(2).to_string())

                pivoted.to_csv(out_dir / "sweep_summary_pivoted.csv")
                click.echo(
                    f"\nPivoted results saved to: {(out_dir / 'sweep_summary_pivoted.csv').absolute()}"
                )
            except Exception as e:
                click.echo(f"Could not construct pivot table: {e}")


# Attribution subcommands defined next
@main.command("ablate")
@click.option(
    "--checkpoint",
    required=True,
    type=click.Path(exists=True),
    help="Path to policy checkpoint directory.",
)
@click.option(
    "--dataset-repo-id", required=True, help="Hugging Face dataset repository ID."
)
@click.option(
    "--episodes", default=None, help="Episodes to run on (e.g. 0:50 or 0,1,2)."
)
@click.option("--cameras", default=None, help="Comma-separated camera keys to ablate.")
@click.option("--seed", type=int, default=0, help="Noise seed.")
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to run on.",
)
def cmd_ablate(checkpoint, dataset_repo_id, episodes, cameras, seed, device):
    """Runs camera-level input ablation to measure importance."""
    from lerobot_policy_smolvla_rl.analyze.attribution.policy_io import (
        load_recap_policy,
    )
    from lerobot_policy_smolvla_rl.analyze.attribution.ablation import (
        run_camera_ablation,
    )

    policy = load_recap_policy(Path(checkpoint), device)
    ep_list = parse_episodes(episodes)
    cam_list = [c.strip() for c in cameras.split(",")] if cameras else None

    res = run_camera_ablation(
        policy, dataset_repo_id, cameras=cam_list, episodes=ep_list, noise_seed=seed
    )

    checkpoint_name = Path(checkpoint).name
    out_dir = Path("outputs/analysis/attribution") / checkpoint_name / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    res.per_task.to_csv(out_dir / "table.csv", index=False)
    click.echo("\n=================== Ablation Table ===================")
    click.echo(res.per_task.to_string(index=False))
    click.echo(f"\nResults saved to: {(out_dir / 'table.csv').absolute()}")


@main.command("attention")
@click.option(
    "--checkpoint",
    required=True,
    type=click.Path(exists=True),
    help="Path to policy checkpoint directory.",
)
@click.option(
    "--dataset-repo-id", required=True, help="Hugging Face dataset repository ID."
)
@click.option("--episodes", default=None, help="Episodes to run on.")
@click.option("--rollout", is_flag=True, help="Use attention rollout across layers.")
@click.option("--seed", type=int, default=0, help="Random seed for action sampling.")
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to run on.",
)
def cmd_attention(checkpoint, dataset_repo_id, episodes, rollout, seed, device):
    """Runs attention capture to record attention map and attribution."""
    from lerobot_policy_smolvla_rl.analyze.attribution.policy_io import (
        load_recap_policy,
        prefix_layout,
        iter_eval_batches,
    )
    from lerobot_policy_smolvla_rl.analyze.attribution.attention import (
        AttentionRecorder,
        action_to_image_attention,
    )
    from lerobot_policy_smolvla_rl.analyze.attribution.report import (
        AttributionScores,
        importance_table,
        saliency_overlay,
    )

    # Seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = load_recap_policy(Path(checkpoint), device)
    ep_list = parse_episodes(episodes)

    layout = prefix_layout(policy)

    # We will accumulate scores across batches
    batches = list(
        iter_eval_batches(
            dataset_repo_id, policy, episodes=ep_list, stride=1, batch_size=1
        )
    )

    all_scores = {}
    tasks = []

    # We will also save a few saliency overlays (e.g. first 5 frames)
    checkpoint_name = Path(checkpoint).name
    out_dir = Path("outputs/analysis/attribution") / checkpoint_name / "attention"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for idx, batch_dict in enumerate(batches):
        recap_batch = batch_dict["recap_batch"]
        batch = batch_dict["batch"]

        task_instruction = batch["task"][0] if "task" in batch else "default_task"
        if isinstance(task_instruction, bytes):
            task_instruction = task_instruction.decode("utf-8")
        tasks.append(task_instruction)

        # Prepare inputs for the denoising inference pass
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)

        dtype = policy.model.vlm_with_expert.vlm.dtype
        images = [img.to(device).to(dtype) for img in images]
        state = state.to(device).to(dtype)

        lang_tokens = recap_batch["observation.language.tokens"]
        lang_masks = recap_batch["observation.language.attention_mask"]

        actions_shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
        noise = policy.model.sample_noise(actions_shape, device)

        # Capture attention during inference denoising pass
        with AttentionRecorder(policy) as recorder:
            with torch.no_grad():
                policy.model.sample_actions(
                    images=images,
                    img_masks=img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    state=state,
                    noise=noise,
                )

        # Resolve attention to camera attns
        # cam_attns is dict of camera -> (batch, num_img_tokens)
        cam_attns = action_to_image_attention(recorder, layout, rollout=rollout)

        # Save overlays for first few frames
        if idx < 5:
            for cam_key, scores_64 in cam_attns.items():
                if cam_key in batch:
                    img = batch[cam_key][0]
                    fig = saliency_overlay(img, scores_64[0])
                    fig.savefig(
                        overlay_dir / f"frame_{idx}_{cam_key.replace('.', '_')}.png",
                        bbox_inches="tight",
                    )
                    plt.close(fig)

        # Store detailed scores for importance table
        for cam_key, scores_64 in cam_attns.items():
            if cam_key not in all_scores:
                all_scores[cam_key] = []
            all_scores[cam_key].append(scores_64)

    # Combine scores
    combined_scores = {}
    for cam_key, lst in all_scores.items():
        combined_scores[cam_key] = torch.cat(lst, dim=0)

    scores_obj = AttributionScores(scores=combined_scores, tasks=tasks)
    table_df = importance_table(scores_obj, reduction="sum")

    table_df.to_csv(out_dir / "table.csv", index=False)
    click.echo("\n=================== Attention Importance Table ===================")
    click.echo(table_df.to_string(index=False))
    click.echo(f"\nResults saved to: {(out_dir / 'table.csv').absolute()}")


@main.command("gradients")
@click.option(
    "--checkpoint",
    required=True,
    type=click.Path(exists=True),
    help="Path to policy checkpoint directory.",
)
@click.option(
    "--dataset-repo-id", required=True, help="Hugging Face dataset repository ID."
)
@click.option(
    "--method",
    type=click.Choice(["gxi", "ig"]),
    default="gxi",
    help="Attribution method (gxi=Grad*Input, ig=Integrated Gradients).",
)
@click.option("--action-dims", default=None, help="Action dimensions slice (e.g. 0:3).")
@click.option("--episodes", default=None, help="Episodes to run on.")
@click.option("--seed", type=int, default=0, help="Random seed for action sampling.")
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to run on.",
)
def cmd_gradients(
    checkpoint, dataset_repo_id, method, action_dims, episodes, seed, device
):
    """Runs gradient-based token attribution (Grad*Input or Integrated Gradients)."""
    from lerobot_policy_smolvla_rl.analyze.attribution.policy_io import (
        load_recap_policy,
        prefix_layout,
        iter_eval_batches,
    )
    from lerobot_policy_smolvla_rl.analyze.attribution.gradients import (
        grad_x_input_attribution,
        integrated_gradients_attribution,
    )
    from lerobot_policy_smolvla_rl.analyze.attribution.report import (
        AttributionScores,
        importance_table,
        saliency_overlay,
    )

    # Seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = load_recap_policy(Path(checkpoint), device)
    ep_list = parse_episodes(episodes)
    act_slice = parse_slice(action_dims)

    layout = prefix_layout(policy)

    batches = list(
        iter_eval_batches(
            dataset_repo_id, policy, episodes=ep_list, stride=1, batch_size=1
        )
    )

    all_scores = {}
    tasks = []

    checkpoint_name = Path(checkpoint).name
    out_dir = Path("outputs/analysis/attribution") / checkpoint_name / "gradients"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for idx, batch_dict in enumerate(batches):
        batch_dict["recap_batch"]
        batch = batch_dict["batch"]

        task_instruction = batch["task"][0] if "task" in batch else "default_task"
        if isinstance(task_instruction, bytes):
            task_instruction = task_instruction.decode("utf-8")
        tasks.append(task_instruction)

        # Compute gradient attribution
        if method == "gxi":
            grad_attns = grad_x_input_attribution(
                policy, batch_dict, layout, action_dims=act_slice
            )
        else:
            grad_attns = integrated_gradients_attribution(
                policy, batch_dict, layout, steps=16, action_dims=act_slice
            )

        # grad_attns is dict of camera -> (batch, num_img_tokens)
        for cam_key, scores_64 in grad_attns.items():
            if cam_key not in all_scores:
                all_scores[cam_key] = []
            all_scores[cam_key].append(
                scores_64
            )  # Keep the (batch, num_img_tokens) tensors

            # Save overlays for first few frames
            if idx < 5 and cam_key in batch:
                img = batch[cam_key][0]
                fig = saliency_overlay(img, scores_64[0])
                fig.savefig(
                    overlay_dir / f"frame_{idx}_{cam_key.replace('.', '_')}.png",
                    bbox_inches="tight",
                )
                plt.close(fig)

    combined_scores = {}
    for cam_key, lst in all_scores.items():
        combined_scores[cam_key] = torch.cat(lst, dim=0)

    scores_obj = AttributionScores(scores=combined_scores, tasks=tasks)
    table_df = importance_table(scores_obj, reduction="mean")

    table_df.to_csv(out_dir / "table.csv", index=False)
    click.echo(
        f"\n=================== Gradient ({method.upper()}) Importance Table ==================="
    )
    click.echo(table_df.to_string(index=False))
    click.echo(f"\nResults saved to: {(out_dir / 'table.csv').absolute()}")


@main.command("layer-redundancy")
@click.option(
    "--checkpoint",
    required=True,
    type=click.Path(exists=True),
    help="Path to policy checkpoint directory.",
)
@click.option(
    "--dataset-repo-id", required=True, help="Hugging Face dataset repository ID."
)
@click.option("--episodes", default=None, help="Episodes to run on.")
@click.option(
    "--calibration-frames", type=int, default=256, help="Minimum number of frames for redundancy analysis."
)
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device to run on.",
)
def cmd_layer_redundancy(checkpoint, dataset_repo_id, episodes, calibration_frames, device):
    """Runs layer activation similarity analysis to measure redundancy."""
    from lerobot_policy_smolvla_rl.analyze.layer_redundancy import run_layer_redundancy_analysis
    from lerobot_policy_smolvla_rl.analyze.attribution.policy_io import load_recap_policy

    policy = load_recap_policy(Path(checkpoint), device)
    ep_list = parse_episodes(episodes)

    res_df = run_layer_redundancy_analysis(
        policy, dataset_repo_id, episodes=ep_list, calibration_frames=calibration_frames, device=device
    )

    checkpoint_name = Path(checkpoint).name
    out_dir = Path("outputs/analysis/redundancy") / checkpoint_name
    out_dir.mkdir(parents=True, exist_ok=True)

    res_df.to_csv(out_dir / "layer_redundancy.csv", index=False)
    click.echo("\n=================== Layer Redundancy Rankings ===================")
    click.echo(res_df.to_string(index=False))
    click.echo(f"\nResults saved to: {(out_dir / 'layer_redundancy.csv').absolute()}")


if __name__ == "__main__":
    main()
