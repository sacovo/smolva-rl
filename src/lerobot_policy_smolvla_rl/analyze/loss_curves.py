import pandas as pd
import numpy as np
from lerobot_policy_smolvla_rl.analyze.runs import RunData


def block_stats(
    run: RunData, metric: str, *, block: int = 500, max_step: int | None = None
) -> pd.DataFrame:
    """Computes basic stats (mean, std, min) in bins of training steps."""
    if metric not in run.df.columns:
        raise KeyError(
            f"Metric '{metric}' not found in run data columns: {list(run.df.columns)}"
        )

    df = run.df.copy()
    if max_step is not None:
        df = df[df["step"] <= max_step]

    if df.empty:
        return pd.DataFrame(columns=["step_start", "step_end", "mean", "std", "min"])

    results = []
    min_step = df["step"].min()
    max_step_val = df["step"].max()

    start = (min_step // block) * block
    while start <= max_step_val:
        end = start + block
        chunk = df[(df["step"] >= start) & (df["step"] < end)]
        if not chunk.empty:
            vals = chunk[metric].values
            results.append(
                {
                    "step_start": int(chunk["step"].min()),
                    "step_end": int(chunk["step"].max()),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                }
            )
        start = end

    return pd.DataFrame(results)


def summarize(
    run: RunData,
    metric: str,
    *,
    max_step: int | None = None,
    tail: int = 100,
    sma_window: int = 10,
) -> dict:
    """Summarizes a run's metric with standard statistics (initial, mean, tail, SMA)."""
    if metric not in run.df.columns:
        raise KeyError(
            f"Metric '{metric}' not found in run data columns: {list(run.df.columns)}"
        )

    df = run.df.copy()
    if max_step is not None:
        df = df[df["step"] <= max_step]

    if df.empty:
        return {}

    vals = df[metric].values
    sma = df[metric].rolling(window=sma_window, min_periods=1).mean().values

    res = {
        "initial": float(vals[0]),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "tail_mean": float(np.mean(vals[-tail:]))
        if len(vals) >= tail
        else float(np.mean(vals)),
        "sma_at_end": float(sma[-1]) if len(sma) > 0 else float("nan"),
        "total_points": len(df),
    }

    # Matches initial_50_mean from analyze_loss.py
    init_50 = vals[:50]
    if len(init_50) > 0:
        res["initial_50_mean"] = float(np.mean(init_50))
        res["initial_50_std"] = float(np.std(init_50))

    return res


def compare_runs(
    runs: list[RunData],
    metric: str,
    *,
    max_step: int | None = None,
    sma_window: int = 10,
) -> pd.DataFrame:
    """Combines multiple runs into a long-form DataFrame with raw and SMA values."""
    dfs = []
    for run in runs:
        if metric not in run.df.columns:
            continue
        df = run.df[["step", metric]].copy()
        if max_step is not None:
            df = df[df["step"] <= max_step]
        if df.empty:
            continue

        df = df.rename(columns={metric: "value"})
        df["run"] = run.name
        df["sma"] = df["value"].rolling(window=sma_window, min_periods=1).mean()
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["run", "step", "value", "sma"])
    return pd.concat(dfs, ignore_index=True)
