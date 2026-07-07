import os
import sys
import tempfile
import torch
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import subprocess

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), "src"))

from lerobot_policy_smolvla_rl.ds_utils import calculate_returns, C_FAIL

def test_calculate_returns_success_vs_fail():
    # episode_lengths, max_lengths, task_idxs, episode_idxs, frame_idxs
    episode_lengths = torch.tensor([100, 100], dtype=torch.float32)
    max_lengths = torch.tensor([100], dtype=torch.float32)
    task_idxs = torch.tensor([0, 0], dtype=torch.long)
    episode_idxs = torch.tensor([0, 1], dtype=torch.long)
    frame_idxs = torch.tensor([50, 50], dtype=torch.long) # half-way through the episode
    
    # Success flags: episode 0 succeeded, episode 1 failed
    success_flags = torch.tensor([True, False], dtype=torch.bool)
    
    # Calculate returns
    returns = calculate_returns(
        episode_lengths,
        max_lengths,
        task_idxs,
        episode_idxs,
        frame_idxs,
        post_goal_buffer=0,
        success_flags=success_flags,
    )
    
    # Episode 0: rem_steps = 100 - 0 - 50 - 1 = 49
    # returns[0] = -(49 / 100) = -0.49
    assert torch.allclose(returns[0], torch.tensor(-0.49))
    
    # Episode 1: rem_steps = 49 + C_FAIL
    # returns[1] = -(rem_steps / 100)
    assert torch.allclose(returns[1], torch.tensor(-(49 + C_FAIL) / 100))
    
    # Ensure failed returns clamp to the lowest bin index (0) when normalized to vmin=-1.0, vmax=0.0
    vmin = -1.0
    vmax = 0.0
    num_bins = 201
    
    # Normalize
    norm_indices_fail = ((returns[1] - vmin) / (vmax - vmin)) * (num_bins - 1)
    clamped_fail = torch.clamp(norm_indices_fail, 0, num_bins - 1).long()
    
    norm_indices_succ = ((returns[0] - vmin) / (vmax - vmin)) * (num_bins - 1)
    clamped_succ = torch.clamp(norm_indices_succ, 0, num_bins - 1).long()
    
    assert clamped_fail.item() == 0
    assert clamped_succ.item() > 0


def test_parquet_metadata_injection():
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_dir = Path(tmpdir) / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        episodes_path = meta_dir / "episodes.parquet"
        
        # Write dummy parquet episodes file
        dummy_data = {
            "episode_index": [0, 1, 2],
            "length": [50, 60, 70],
        }
        table = pa.Table.from_pydict(dummy_data)
        pq.write_table(table, episodes_path)
        
        # Inject success column
        episode_successes = [True, False, True]
        
        # Load and write back
        assert episodes_path.exists()
        read_table = pq.read_table(episodes_path)
        if "success" in read_table.column_names:
            success_idx = read_table.column_names.index("success")
            read_table = read_table.set_column(success_idx, "success", pa.array(episode_successes, type=pa.bool_()))
        else:
            read_table = read_table.append_column("success", pa.array(episode_successes, type=pa.bool_()))
        pq.write_table(read_table, episodes_path)
        
        # Read again to verify
        final_table = pq.read_table(episodes_path)
        assert "success" in final_table.column_names
        successes = final_table["success"].to_pylist()
        assert successes == episode_successes


def test_recap_expert_mode_smoke():
    save_dir = "outputs/test_recap_expert_pytest"
    cmd = [
        sys.executable,
        "src/lerobot_policy_smolvla_rl/train_recap.py",
        "--dataset_repo_id",
        "lerobot/pusht",
        "--episodes",
        "0",
        "--action_chunk_size",
        "20",
        "--steps",
        "1",
        "--batch_size",
        "1",
        "--num_vlm_layers",
        "1",
        "--save_dir",
        save_dir,
        "--wandb_project",
        "pytest-expert-smoke-test",
        "--num_workers",
        "0",
        "--expert_mode"
    ]

    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"

    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert res.returncode == 0
    # The script saves state using accelerator.save_state which creates a directory
    assert os.path.exists(save_dir)


def test_normalizer_stats_injection():
    """Verify that normalizer stats can be injected into migrated processor pipelines.
    
    This tests the fix for the bug where exported checkpoints had empty normalizer
    stats, causing state normalization and action unnormalization to be no-ops.
    """
    import json
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create a minimal policy_preprocessor.json with a normalizer step
        preprocessor_config = {
            "name": "policy_preprocessor",
            "steps": [
                {
                    "registry_name": "normalizer_processor",
                    "config": {
                        "eps": 1e-08,
                        "features": {
                            "observation.state": {
                                "type": "STATE",
                                "shape": [8]
                            },
                            "action": {
                                "type": "ACTION",
                                "shape": [7]
                            }
                        },
                        "norm_map": {
                            "STATE": "MEAN_STD",
                            "ACTION": "MEAN_STD"
                        }
                    }
                }
            ]
        }
        
        postprocessor_config = {
            "name": "policy_postprocessor",
            "steps": [
                {
                    "registry_name": "unnormalizer_processor",
                    "config": {
                        "eps": 1e-08,
                        "features": {
                            "action": {
                                "type": "ACTION",
                                "shape": [7]
                            }
                        },
                        "norm_map": {
                            "ACTION": "MEAN_STD"
                        }
                    }
                }
            ]
        }
        
        with open(tmpdir / "policy_preprocessor.json", "w") as f:
            json.dump(preprocessor_config, f)
        with open(tmpdir / "policy_postprocessor.json", "w") as f:
            json.dump(postprocessor_config, f)
        
        # Load processors without stats (simulates the broken case)
        from lerobot.processor.pipeline import PolicyProcessorPipeline
        from lerobot.processor.normalize_processor import NormalizerProcessorStep
        
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=str(tmpdir),
            config_filename="policy_preprocessor.json",
        )
        
        # Verify stats are initially empty
        normalizer_step = None
        for step in preprocessor.steps:
            if isinstance(step, NormalizerProcessorStep):
                normalizer_step = step
                break
        
        assert normalizer_step is not None, "Normalizer step not found"
        assert not normalizer_step.stats, "Stats should be empty before injection"
        
        # Inject stats via load_state_dict (simulates the fix) using flat format
        flat_stats = {
            "observation.state.mean": torch.zeros(8),
            "observation.state.std": torch.ones(8),
            "action.mean": torch.zeros(7),
            "action.std": torch.ones(7),
        }
        
        for step in preprocessor.steps:
            if hasattr(step, 'load_state_dict') and hasattr(step, 'norm_map'):
                step.load_state_dict(flat_stats)
        
        # Verify stats are populated
        assert normalizer_step.stats is not None, "Stats should be set after injection"
        assert "observation.state" in normalizer_step.stats
        assert "action" in normalizer_step.stats
        
        # Save and reload to verify round-trip
        preprocessor.save_pretrained(str(tmpdir))
        
        preprocessor_reloaded = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=str(tmpdir),
            config_filename="policy_preprocessor.json",
        )
        
        for step in preprocessor_reloaded.steps:
            if isinstance(step, NormalizerProcessorStep):
                assert step.stats is not None, "Stats should persist after save/reload"
                assert "observation.state" in step.stats, "State stats lost after reload"
                assert "action" in step.stats, "Action stats lost after reload"
                break
        else:
            raise AssertionError("Normalizer step not found after reload")
