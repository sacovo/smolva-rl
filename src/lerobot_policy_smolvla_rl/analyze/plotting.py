import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from lerobot_policy_smolvla_rl.analyze.runs import RunData


@dataclass
class StepMarker:
    step: int
    label: str
    color: str = "#888888"
    linestyle: str = "--"


def apply_style():
    """Applies clean, modern plotting style."""
    if "seaborn-v0_8-whitegrid" in plt.style.available:
        plt.style.use("seaborn-v0_8-whitegrid")
    else:
        plt.style.use("default")
        plt.rcParams["grid.color"] = "#e0e0e0"
        plt.rcParams["axes.edgecolor"] = "#cccccc"


def plot_metric_comparison(
    runs: list[RunData],
    metric: str,
    *,
    sma_window: int = 10,
    max_step: int | None = None,
    title: str | None = None,
    annotations: list[StepMarker] = (),
) -> plt.Figure:
    """Plots raw trace and SMA per run, with optional annotations."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    has_data = False
    for i, run in enumerate(runs):
        if metric not in run.df.columns:
            continue
        df = run.df.copy()
        if max_step is not None:
            df = df[df["step"] <= max_step]
        if df.empty:
            continue

        has_data = True
        color = colors[i % len(colors)]
        ax.plot(
            df["step"], df[metric], color=color, alpha=0.15, label=f"{run.name} (raw)"
        )
        sma = df[metric].rolling(window=sma_window, min_periods=1).mean()
        ax.plot(df["step"], sma, color=color, linewidth=2, label=f"{run.name} (SMA)")

    if not has_data:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    else:
        ax.set_title(
            f"{metric.capitalize()} Comparison", fontsize=14, fontweight="bold", pad=15
        )

    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel(metric.capitalize(), fontsize=11)
    ax.legend(
        frameon=True, facecolor="white", framealpha=0.9, loc="upper right", fontsize=10
    )

    # Annotate markers
    for marker in annotations:
        ax.axvline(
            x=marker.step, color=marker.color, linestyle=marker.linestyle, linewidth=1
        )
        ylim = ax.get_ylim()
        y_pos = ylim[0] + 0.1 * (ylim[1] - ylim[0])
        ax.text(
            marker.step,
            y_pos,
            f" {marker.label}",
            rotation=90,
            color=marker.color,
            fontsize=10,
            va="bottom",
            ha="right",
        )

    plt.tight_layout()
    return fig


def save_figure(fig: plt.Figure, analysis_name: str, filename: str) -> Path:
    """Saves a figure to the standard output structure outputs/analysis/<analysis-name>/."""
    out_dir = Path("outputs/analysis") / analysis_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
