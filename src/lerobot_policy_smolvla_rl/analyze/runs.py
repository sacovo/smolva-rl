import pandas as pd
from pathlib import Path
from dataclasses import dataclass


class RunNotFoundError(Exception):
    """Raised when a WandB run cannot be located."""

    pass


@dataclass
class RunSummary:
    id: str
    name: str
    project: str
    entity: str
    config: dict


@dataclass
class RunData:
    name: str  # run display name
    df: pd.DataFrame  # columns: step (true training step), <metric>...
    meta: dict  # config subset (dataset_repo_id, ...)


def load_wandb_csv(
    path: Path, run_column: str, *, log_interval: int = 10, step_offset: int = 0
) -> RunData:
    """Loads run data from a local WandB CSV export, aligning steps and resolving columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found at: {path}")

    df_raw = pd.read_csv(path)

    # 1. Resolve column
    matched_col = None
    if run_column in df_raw.columns:
        matched_col = run_column
    else:
        # Find columns containing the substring, excluding min/max unless specified
        candidates = [
            col
            for col in df_raw.columns
            if run_column in col
            and not col.endswith("__MIN")
            and not col.endswith("__MAX")
        ]
        if not candidates:
            candidates = [col for col in df_raw.columns if run_column in col]

        if not candidates:
            raise KeyError(
                f"Could not find column matching '{run_column}' in {path}. Available: {list(df_raw.columns)}"
            )

        # Prefer candidate containing "loss" or " - "
        loss_candidates = [c for c in candidates if "loss" in c.lower()]
        matched_col = loss_candidates[0] if loss_candidates else candidates[0]

    # 2. Extract run name and metric
    if " - " in matched_col:
        run_name, metric = matched_col.split(" - ", 1)
    else:
        run_name = path.stem
        metric = matched_col

    # 3. Clean and align
    df_clean = df_raw[["Step", matched_col]].dropna().copy()
    df_clean["step"] = df_clean["Step"] * log_interval + step_offset
    df_clean = df_clean.drop(columns=["Step"])
    df_clean = df_clean.rename(columns={matched_col: metric})
    df_clean = df_clean[["step", metric]].sort_values("step").reset_index(drop=True)

    return RunData(name=run_name, df=df_clean, meta={})


def fetch_wandb_run(run_path: str, keys: list[str]) -> RunData:
    """Fetches a run's configuration and metric history directly from WandB.

    Note: If run_path is a bare run ID (e.g. "xtjfg3f2"), this function will scan all
    projects in the user's entity scope sequentially to locate the run, which can generate
    significant API overhead for entities with many projects. Pass the full project/run_id
    or entity/project/run_id to avoid this.
    """
    import wandb

    api = wandb.Api()

    run = None
    if "/" in run_path:
        parts = run_path.split("/")
        if len(parts) == 3:
            entity, project, run_id = parts
        elif len(parts) == 2:
            entity = api.default_entity
            project, run_id = parts
        else:
            raise ValueError(
                f"Invalid run path format: '{run_path}'. Expected 'entity/project/run_id' or 'run_id'."
            )

        try:
            run = api.run(f"{entity}/{project}/{run_id}")
        except Exception as e:
            raise RunNotFoundError(
                f"Could not find WandB run at '{entity}/{project}/{run_id}': {e}"
            ) from e
    else:
        run_id = run_path
        try:
            projects = api.projects()
        except Exception as e:
            raise RunNotFoundError(f"Could not list WandB projects: {e}") from e

        for proj in projects:
            try:
                run = api.run(f"{proj.entity}/{proj.name}/{run_id}")
                if run:
                    break
            except Exception as e:
                msg = str(e).lower()
                if "not found" in msg or "404" in msg:
                    pass
                else:
                    raise e

        if not run:
            raise RunNotFoundError(
                f"Could not find WandB run '{run_id}' in any project."
            )

    meta = dict(run.config)

    # scan history
    history = run.scan_history(keys=["step"] + keys)
    rows = [row for row in history]
    df = pd.DataFrame(rows)

    if df.empty:
        df = pd.DataFrame(columns=["step"] + keys)
    else:
        if "step" not in df.columns:
            df["step"] = 0
        cols = ["step"] + [k for k in keys if k in df.columns]
        df = df[cols].sort_values("step").reset_index(drop=True)

    return RunData(name=run.name, df=df, meta=meta)


def find_wandb_runs(
    project: str,
    *,
    dataset_contains: str | None = None,
    name_contains: str | None = None,
) -> list[RunSummary]:
    """Finds runs matching filters in the specified project."""
    import wandb

    api = wandb.Api()
    try:
        runs = api.runs(project)
    except Exception as e:
        raise RunNotFoundError(
            f"Could not list runs in project '{project}': {e}"
        ) from e

    summaries = []
    for run in runs:
        if name_contains and name_contains not in run.name:
            continue
        ds = run.config.get("dataset_repo_id", "")
        if dataset_contains and dataset_contains not in ds:
            continue

        summaries.append(
            RunSummary(
                id=run.id,
                name=run.name,
                project=run.project,
                entity=run.entity,
                config=dict(run.config),
            )
        )
    return summaries
