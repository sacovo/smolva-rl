import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from lerobot_policy_smolvla_rl.ds_utils import (
        calculate_returns,
        get_episode_lengths,
        get_max_task_lengths,
    )

    from itertools import batched

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import torch

    import seaborn as sns
    from matplotlib import pyplot as plt

    return (
        LeRobotDataset,
        batched,
        calculate_returns,
        get_episode_lengths,
        get_max_task_lengths,
        plt,
        sns,
        torch,
    )


@app.cell
def _(LeRobotDataset):
    ds = LeRobotDataset("lerobot/berkeley_fanuc_manipulation")
    return (ds,)


@app.cell
def _(ds, get_episode_lengths, get_max_task_lengths):
    max_lengths = get_max_task_lengths(ds)
    episode_lengths = get_episode_lengths(ds)
    return episode_lengths, max_lengths


@app.cell
def _(batched, calculate_returns, ds, episode_lengths, max_lengths, torch):
    returns = []

    for batch in batched(ds, 1024):
        task_idxs = [int(frame["task_index"]) for frame in batch]
        episode_idxs = [int(frame["episode_index"]) for frame in batch]
        frame_idxs = torch.tensor([int(frame["frame_index"]) for frame in batch])

        returns.append(
            calculate_returns(
                episode_lengths, max_lengths, task_idxs, episode_idxs, frame_idxs
            )
        )

    returns = torch.cat(returns)
    return (returns,)


@app.cell
def _(plt, returns, sns):
    plt.figure(figsize=(12, 6))
    sns.histplot(returns, bins=51).set(title="Frame returns", xlabel="return")
    return


@app.cell
def _(torch):
    torch.rand((4, 201)).unsqueeze(1).shape
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
