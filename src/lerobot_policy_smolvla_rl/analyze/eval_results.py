import json
import fnmatch
from pathlib import Path
import pandas as pd


def collect_eval_results(
    eval_dir: Path = Path("outputs/eval"), job_pattern: str = "*"
) -> pd.DataFrame:
    """Collects and aggregates evaluation results from outputs/eval into a DataFrame."""
    eval_dir = Path(eval_dir)
    if not eval_dir.exists():
        return pd.DataFrame()

    rows = []
    # Search for all eval_info.json files recursively
    for info_path in eval_dir.glob("**/eval_info.json"):
        folder = info_path.parent
        folder_name = folder.name

        # Match timestamp prefix, e.g. 2026-06-04T12-42-51_recap_...
        # Split by first '_'
        parts = folder_name.split("_", 1)
        if len(parts) == 2 and any(char.isdigit() for char in parts[0]):
            timestamp = parts[0]
            job_name = parts[1]
        else:
            timestamp = "Unknown"
            job_name = folder_name

        # Check pattern match
        if not fnmatch.fnmatch(job_name, job_pattern):
            continue

        try:
            with open(info_path, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        success_rate = data.get("overall", {}).get("pc_success", 0.0)
        n_episodes = data.get("overall", {}).get("n_episodes", 0)

        row = {
            "job_name": job_name,
            "timestamp": timestamp,
            "pc_success": success_rate,
            "n_episodes": n_episodes,
            "path": str(info_path),
        }

        # Parse checkpoint, suite, and cfg weight from job_name if it matches the pattern
        if "_cfg_" in job_name:
            prefix, cfg_val = job_name.split("_cfg_", 1)
            try:
                row["cfg"] = float(cfg_val)
            except ValueError:
                row["cfg"] = cfg_val

            if prefix.startswith("recap_"):
                prefix = prefix[len("recap_") :]

            known_suites = [
                "libero_spatial",
                "libero_object",
                "libero_goal",
                "libero_10",
            ]
            matched_suite = None
            for suite in known_suites:
                if prefix.endswith(suite):
                    matched_suite = suite
                    checkpoint = prefix[: -len(suite)].rstrip("_")
                    break
            if matched_suite:
                row["checkpoint"] = checkpoint
                row["suite"] = matched_suite
            else:
                parts_prefix = prefix.rsplit("_", 1)
                if len(parts_prefix) == 2:
                    row["checkpoint"] = parts_prefix[0]
                    row["suite"] = parts_prefix[1]

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)
