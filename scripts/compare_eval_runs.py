#!/usr/bin/env python3
"""Print a per-run comparison of LIBERO eval results.

Parses every ``eval_info.json`` under ``outputs/eval`` and prints a sorted
pandas table with the overall success rate and per-task-group success rates
for each run. Used by ``scripts/eval_cfg.sh`` and ``scripts/eval_snapflow.sh``
to summarize a sweep once all evaluations finish.

For the checkpoint × CFG × suite pivot used by the sweep report, see
``scripts/compile_results.py`` (built on ``analyze.eval_results``).
"""
import json
from pathlib import Path

import pandas as pd


def main():
    eval_dir = Path("outputs/eval")
    results = []

    # Walk through outputs/eval to find eval_info.json
    for path in eval_dir.glob("**/eval_info.json"):
        try:
            with open(path, "r") as f:
                data = json.load(f)

            # Path format: outputs/eval/YYYY-MM-DD/HH-MM-SS_job_name/eval_info.json
            parts = path.parts
            date_str = parts[-3]
            time_job_str = parts[-2]

            # Read overall stats
            overall = data.get("overall", {})
            per_group = data.get("per_group", {})

            # Build entry
            entry = {
                "Date": date_str,
                "Run Folder": time_job_str,
                "Overall Success (%)": overall.get("pc_success", 0.0),
                "Total Ep": overall.get("n_episodes", 0),
            }

            # Add each task group success rate
            for group, group_data in per_group.items():
                entry[group] = group_data.get("pc_success", 0.0)

            results.append(entry)
        except Exception as e:
            print(f"Error parsing {path}: {e}")

    if not results:
        print("No evaluation results found.")
        return

    df = pd.DataFrame(results)
    # Sort by date and run folder desc
    df = df.sort_values(by=["Date", "Run Folder"], ascending=[False, False])

    # Reorder columns: fixed columns first, then task groups sorted
    cols = ["Date", "Run Folder", "Overall Success (%)", "Total Ep"]
    group_cols = [c for c in df.columns if c not in cols]
    ordered_cols = cols + sorted(group_cols)
    df = df[ordered_cols]

    print("\n=======================================================")
    print("EVALUATION RUN COMPARISON")
    print("=======================================================")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df.to_string(index=False))
    print("=======================================================\n")


if __name__ == "__main__":
    main()
