import os
import sys
import json
import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock, patch


from lerobot_policy_smolvla_rl.analyze.runs import (
    load_wandb_csv,
    fetch_wandb_run,
    find_wandb_runs,
    RunData,
)
from lerobot_policy_smolvla_rl.analyze.loss_curves import (
    block_stats,
    summarize,
    compare_runs,
)
from lerobot_policy_smolvla_rl.analyze.datasets import dataset_summary
from lerobot_policy_smolvla_rl.analyze.eval_results import collect_eval_results
from lerobot_policy_smolvla_rl.analyze.attribution.policy_io import prefix_layout
from lerobot_policy_smolvla_rl.analyze.attribution.attention import (
    AttentionRecorder,
    action_to_image_attention,
)
from lerobot_policy_smolvla_rl.analyze.attribution.gradients import (
    grad_x_input_attribution,
    integrated_gradients_attribution,
)
from lerobot_policy_smolvla_rl.analyze.attribution.ablation import run_camera_ablation

# ---------------------------------------------------------------------------
# 1. runs.py tests
# ---------------------------------------------------------------------------


def test_load_wandb_csv(tmp_path):
    # Create synthetic CSV
    csv_file = tmp_path / "test_runs.csv"
    data = {
        "Step": [0, 1, 2, 3],
        "run1_scratch_0603 - loss": [5.0, 4.0, 3.0, 2.0],
        "run1_scratch_0603 - loss__MIN": [4.9, 3.9, 2.9, 1.9],
        "run2_finetune_0604 - loss": [4.5, 3.5, 2.5, 1.5],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    # Test substring matching: "scratch" should resolve to "run1_scratch_0603 - loss"
    run_data = load_wandb_csv(csv_file, "scratch", log_interval=10, step_offset=100)
    assert run_data.name == "run1_scratch_0603"
    assert list(run_data.df.columns) == ["step", "loss"]
    assert list(run_data.df["step"]) == [100, 110, 120, 130]
    assert list(run_data.df["loss"]) == [5.0, 4.0, 3.0, 2.0]

    # Test matching when exact key provided
    run_data2 = load_wandb_csv(csv_file, "run2_finetune_0604 - loss", log_interval=5)
    assert run_data2.name == "run2_finetune_0604"
    assert list(run_data2.df["step"]) == [0, 5, 10, 15]


@patch("wandb.Api")
def test_fetch_wandb_run_mocked(mock_api_class):
    mock_api = MagicMock()
    mock_api_class.return_value = mock_api

    # Mock project runs lookup
    mock_run = MagicMock()
    mock_run.name = "mocked_run_1"
    mock_run.config = {"dataset_repo_id": "test_dataset_repo"}
    mock_run.scan_history.return_value = [
        {"step": 0, "total_loss": 1.2, "ar_loss": 0.5},
        {"step": 10, "total_loss": 1.0, "ar_loss": 0.4},
    ]

    mock_api.run.return_value = mock_run

    run_data = fetch_wandb_run("my_entity/my_proj/my_id", ["total_loss", "ar_loss"])
    assert run_data.name == "mocked_run_1"
    assert run_data.meta["dataset_repo_id"] == "test_dataset_repo"
    assert len(run_data.df) == 2
    assert "total_loss" in run_data.df.columns
    assert "ar_loss" in run_data.df.columns


@patch("wandb.Api")
def test_find_wandb_runs_mocked(mock_api_class):
    mock_api = MagicMock()
    mock_api_class.return_value = mock_api

    mock_run1 = MagicMock()
    mock_run1.id = "run1"
    mock_run1.name = "recap_run_1"
    mock_run1.project = "smolvla-recap"
    mock_run1.entity = "test_entity"
    mock_run1.config = {"dataset_repo_id": "HuggingFaceVLA/libero"}

    mock_run2 = MagicMock()
    mock_run2.id = "run2"
    mock_run2.name = "other_run"
    mock_run2.project = "smolvla-recap"
    mock_run2.entity = "test_entity"
    mock_run2.config = {"dataset_repo_id": "other_dataset"}

    mock_api.runs.return_value = [mock_run1, mock_run2]

    summaries = find_wandb_runs("test_entity/smolvla-recap", dataset_contains="libero")
    assert len(summaries) == 1
    assert summaries[0].id == "run1"
    assert summaries[0].name == "recap_run_1"


# ---------------------------------------------------------------------------
# 2. loss_curves.py tests
# ---------------------------------------------------------------------------


def test_loss_curves_helpers():
    df = pd.DataFrame(
        {
            "step": [0, 100, 200, 300, 400, 500, 600],
            "loss": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
        }
    )
    run = RunData(name="test_run", df=df, meta={})

    # Test block_stats
    stats = block_stats(run, "loss", block=300)
    assert len(stats) == 3
    # Bin 1: steps 0, 100, 200 -> mean 9.0, min 8.0
    assert stats.iloc[0]["mean"] == 9.0
    assert stats.iloc[0]["min"] == 8.0

    # Test summarize
    summary = summarize(run, "loss", tail=3, sma_window=2)
    assert summary["initial"] == 10.0
    assert summary["min"] == 4.0
    assert summary["tail_mean"] == 5.0  # mean of 6.0, 5.0, 4.0
    assert summary["sma_at_end"] == 4.5  # mean of 5.0, 4.0

    # Test compare_runs
    df2 = pd.DataFrame({"step": [0, 100, 200], "loss": [5.0, 4.0, 3.0]})
    run2 = RunData(name="run_2", df=df2, meta={})

    compared = compare_runs([run, run2], "loss", sma_window=2)
    assert len(compared) == len(df) + len(df2)
    assert "sma" in compared.columns
    assert "run" in compared.columns


# ---------------------------------------------------------------------------
# 3. datasets.py tests
# ---------------------------------------------------------------------------


@patch("lerobot_policy_smolvla_rl.analyze.datasets.hf_hub_download")
def test_dataset_summary(mock_hf_download):
    # Mock hf_hub_download to return local fixture files
    info_data = {
        "features": {
            "action": {"shape": [7], "dtype": "float32"},
            "observation.state": {"shape": [8], "dtype": "float32"},
            "observation.images.agent": {
                "type": "image",
                "shape": [3, 224, 224],
                "dtype": "float32",
            },
        },
        "total_episodes": 100,
        "fps": 10,
    }
    stats_data = {"action": {"min": [-1.0], "max": [1.0], "mean": [0.0], "std": [0.5]}}

    # Create temp files
    info_file = Path("info_temp.json")
    stats_file = Path("stats_temp.json")
    with open(info_file, "w") as f:
        json.dump(info_data, f)
    with open(stats_file, "w") as f:
        json.dump(stats_data, f)

    def side_effect(repo_id, repo_type, filename):
        if "info.json" in filename:
            return str(info_file)
        return str(stats_file)

    mock_hf_download.side_effect = side_effect

    try:
        summary = dataset_summary("test/repo", metadata_only=True)
        assert summary.repo_id == "test/repo"
        assert summary.action_shape == [7]
        assert summary.state_shape == [8]
        assert summary.episode_count == 100
        assert summary.fps == 10
        assert summary.camera_keys == ["observation.images.agent"]
        assert summary.action_stats["min"] == [-1.0]
    finally:
        if info_file.exists():
            info_file.unlink()
        if stats_file.exists():
            stats_file.unlink()


# ---------------------------------------------------------------------------
# 4. eval_results.py tests
# ---------------------------------------------------------------------------


def test_collect_eval_results(tmp_path):
    # Create temp eval results directory structure
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()

    # Create job directories matching the naming pattern
    job1_dir = eval_dir / "2026-06-04-12-00-00_recap_150k_libero_spatial_cfg_1.0"
    job1_dir.mkdir()
    with open(job1_dir / "eval_info.json", "w") as f:
        json.dump({"overall": {"pc_success": 85.5, "n_episodes": 20}}, f)

    job2_dir = eval_dir / "2026-06-04-13-00-00_recap_350k_libero_goal_cfg_2.0"
    job2_dir.mkdir()
    with open(job2_dir / "eval_info.json", "w") as f:
        json.dump({"overall": {"pc_success": 92.0, "n_episodes": 20}}, f)

    df = collect_eval_results(eval_dir)
    assert len(df) == 2
    assert "pc_success" in df.columns
    assert "cfg" in df.columns
    assert "checkpoint" in df.columns
    assert "suite" in df.columns

    # Verify values
    row1 = df[df["checkpoint"] == "150k"].iloc[0]
    assert row1["pc_success"] == 85.5
    assert row1["cfg"] == 1.0
    assert row1["suite"] == "libero_spatial"


# ---------------------------------------------------------------------------
# 5. attribution tests
# ---------------------------------------------------------------------------


class MockPolicyConfig:
    def __init__(self):
        self.chunk_size = 4
        self.max_action_dim = 2
        self.tokenizer_max_length = 8
        self.resize_imgs_with_padding = (32, 32)
        self.input_features = {
            "observation.images.agent": MagicMock(type="image"),
            "observation.state": MagicMock(type="vector"),
        }
        self.image_features = {"observation.images.agent": MagicMock()}
        self.output_features = {"action": MagicMock(shape=[2])}
        self.use_cache = True


class MockVLMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MagicMock(num_hidden_layers=2)
        self.dtype = torch.float32


class MockVLMWithExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = MockVLMModel()
        self.num_attention_heads = 4
        self.num_key_value_heads = 2
        self.processor = MagicMock()

    def embed_image(self, img):
        # input is (batch, 3, 32, 32) -> output is (batch, 8, 16)
        bsize = img.shape[0]
        return torch.zeros(bsize, 8, 16, device=img.device, dtype=img.dtype)

    def get_attention_interface(self):
        return self.eager_attention_forward

    def eager_attention_forward(
        self,
        attention_mask,
        batch_size,
        head_dim,
        query_states,
        key_states,
        value_states,
    ):
        # Mock eager forward
        att_weights = torch.zeros(
            batch_size,
            self.num_attention_heads,
            query_states.shape[1],
            key_states.shape[1],
            device=query_states.device,
        )
        torch.nn.functional.softmax(att_weights, dim=-1)
        return torch.zeros(
            batch_size,
            query_states.shape[1],
            self.num_attention_heads * head_dim,
            device=query_states.device,
        )

    def forward(
        self,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=True,
        fill_kv_cache=False,
    ):
        dummy_out = [None, None]
        # Return inputs_embeds[0] (prefix_embs) as past_key_values
        prefix_embs = inputs_embeds[0] if inputs_embeds is not None else None
        if inputs_embeds is not None:
            dummy_out = []
            for item in inputs_embeds:
                if item is not None:
                    dummy_out.append(item * 1.0)
                else:
                    dummy_out.append(None)
        return dummy_out, prefix_embs


class MockRECAPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vlm_with_expert = MockVLMWithExpert()
        self.add_image_special_tokens = False
        self.state_proj = nn.Linear(8, 16)

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
        img_emb = self.vlm_with_expert.embed_image(images[0])
        # prefix: [img_emb (8 tokens) | lang (8 tokens) | state (1 token)] -> 17 tokens
        bsize = state.shape[0]
        state_emb = self.state_proj(state).unsqueeze(1)
        embs = torch.cat(
            [img_emb, torch.zeros(bsize, 8, 16, device=state.device), state_emb], dim=1
        )
        pad_masks = torch.ones(bsize, 17, dtype=torch.bool, device=state.device)
        att_masks = torch.zeros(1, 17, dtype=torch.bool, device=state.device)
        return embs, pad_masks, att_masks

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        # past_key_values is prefix_embs from the forward pass
        chunk_size = self.config.chunk_size
        action_dim = self.config.max_action_dim
        # prefix_embs has shape (batch, 17, 16)
        # We take the mean over the image tokens (first 8 tokens) and slice to action_dim
        # Add to x_t to preserve computation graph
        if past_key_values is not None:
            img_part = past_key_values[:, :8, :action_dim]
            v_t = x_t * 0.5 + img_part.mean(dim=1, keepdim=True).expand(
                -1, chunk_size, -1
            )
        else:
            v_t = x_t * 0.5
        return v_t

    def sample_noise(self, shape, device):
        return torch.randn(shape, device=device)

    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs
    ):
        # Simulates sample_actions
        return noise * 0.5


class MockPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockPolicyConfig()
        self.model = MockRECAPModel(self.config)

    def prepare_images(self, batch):
        return [batch["observation.images.agent"]], [
            torch.ones(batch["observation.images.agent"].shape[0], dtype=torch.bool)
        ]

    def prepare_state(self, batch):
        return batch["observation.state"]


def test_prefix_layout_resolution():
    policy = MockPolicy()
    layout = prefix_layout(policy, lang_len=8)
    assert layout.image_spans["observation.images.agent"] == range(0, 8)
    assert layout.lang_span == range(8, 16)
    assert layout.state_span == range(16, 17)


def test_attention_recorder():
    policy = MockPolicy()
    layout = prefix_layout(policy, lang_len=8)

    # Mock data
    {
        "observation.language.tokens": torch.zeros(1, 8, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(1, 8, dtype=torch.bool),
    }
    {
        "observation.images.agent": torch.zeros(1, 3, 32, 32),
        "observation.state": torch.zeros(1, 8),
        "task": ["do the task"],
        "action": torch.zeros(1, 4, 2),
    }

    # We call standard model forward to trigger get_attention_interface
    with AttentionRecorder(policy) as recorder:
        # Run suffix pass forward pass manually
        # E.g. trigger custom_attention_forward call
        vlm_model = policy.model.vlm_with_expert
        att_interface = vlm_model.get_attention_interface()
        # Query length = suffix_len = 4, KV length = prefix_len (17) + suffix_len (4) = 21
        att_interface(
            attention_mask=torch.ones(1, 4, 21, dtype=torch.bool),
            batch_size=1,
            head_dim=4,
            query_states=torch.zeros(1, 4, 4, 4),
            key_states=torch.zeros(1, 21, 2, 4),
            value_states=torch.zeros(1, 21, 2, 4),
        )

    assert len(recorder.layers) == 1
    # Check shape: (batch, heads, q_len, kv_len)
    assert recorder.layers[0].shape == (1, 4, 4, 21)

    # Compute attention dict
    attns = action_to_image_attention(recorder, layout, rollout=False)
    assert "observation.images.agent" in attns
    assert attns["observation.images.agent"].shape == (1, 8)


def test_gradient_attribution():
    policy = MockPolicy()
    layout = prefix_layout(policy, lang_len=8)

    recap_batch = {
        "observation.language.tokens": torch.zeros(1, 8, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(1, 8, dtype=torch.bool),
    }
    batch = {
        "recap_batch": recap_batch,
        "batch": {
            "observation.images.agent": torch.zeros(1, 3, 32, 32),
            "observation.state": torch.zeros(1, 8),
            "task": ["do the task"],
            "action": torch.zeros(1, 4, 2),
        },
    }

    # Test Grad*Input
    gxi_attns = grad_x_input_attribution(policy, batch, layout)
    assert "observation.images.agent" in gxi_attns
    assert gxi_attns["observation.images.agent"].shape == (1, 8)

    # Test Integrated Gradients
    ig_attns = integrated_gradients_attribution(policy, batch, layout, steps=4)
    assert "observation.images.agent" in ig_attns
    assert ig_attns["observation.images.agent"].shape == (1, 8)


@patch("lerobot_policy_smolvla_rl.analyze.attribution.ablation.iter_eval_batches")
def test_camera_ablation(mock_iter):
    policy = MockPolicy()

    recap_batch = {
        "observation.language.tokens": torch.zeros(1, 8, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(1, 8, dtype=torch.bool),
    }
    batch = {
        "observation.images.agent": torch.zeros(1, 3, 32, 32),
        "observation.state": torch.zeros(1, 8),
        "task": ["do the task"],
        "action": torch.zeros(1, 4, 2),
    }

    mock_iter.return_value = [{"recap_batch": recap_batch, "batch": batch}]

    # Capture calls to verify noise reuse and mask flipping
    sampled_calls = []
    original_sample_actions = policy.model.sample_actions

    def mock_sample_actions(*args, **kwargs):
        sampled_calls.append(
            {
                "img_masks": [m.clone() for m in kwargs.get("img_masks", [])],
                "noise": kwargs.get("noise").clone()
                if kwargs.get("noise") is not None
                else None,
            }
        )
        return original_sample_actions(*args, **kwargs)

    policy.model.sample_actions = mock_sample_actions

    res = run_camera_ablation(
        policy, "test_repo", cameras=["observation.images.agent"], episodes=[0]
    )

    df = res.per_task
    assert len(df) == 2  # none and agent ablated
    assert "task" in df.columns
    assert "ablated_camera" in df.columns
    assert "action_mse" in df.columns

    # Assert noise reuse and mask flipping
    assert len(sampled_calls) == 2
    # 1. Noise is identical (reused)
    assert torch.equal(sampled_calls[0]["noise"], sampled_calls[1]["noise"])
    # 2. Mask for observation.images.agent (first camera) is flipped from all ones to all zeros
    assert torch.all(sampled_calls[0]["img_masks"][0] == 1)
    assert torch.all(sampled_calls[1]["img_masks"][0] == 0)


def test_cli_help():
    from click.testing import CliRunner
    from lerobot_policy_smolvla_rl.analyze.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "loss-compare" in result.output
    assert "gradients" in result.output
    assert "ablate" in result.output
