import os
import sys
import random
import logging
from unittest.mock import MagicMock, patch
from contextlib import contextmanager

import pytest
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), "src"))

from lerobot_policy_smolvla_rl.dataloader_utils import (
    RobustDataset,
    CudaPrefetcher,
    add_dataloader_args,
    build_dataloader,
)


# ---------------------------------------------------------------------------
# Mocks & Helpers
# ---------------------------------------------------------------------------

class MockBaseDataset(torch.utils.data.Dataset):
    def __init__(self, data, episode_from=None):
        self.data = data
        if episode_from is None:
            episode_from = [0]
        self.meta = MagicMock()
        self.meta.episodes = {"dataset_from_index": episode_from}
        self.meta.features = "dummy_features"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        val = self.data[idx]
        if isinstance(val, Exception):
            raise val
        return val


class MockStream:
    def __init__(self, device=None):
        self.device = device
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MockCurrentStream:
    def __init__(self, device=None):
        self.device = device
    def wait_stream(self, stream):
        pass


@contextmanager
def mock_stream_context(stream):
    yield


# ---------------------------------------------------------------------------
# RobustDataset Tests
# ---------------------------------------------------------------------------

def test_robust_dataset_passes_through_clean_sample():
    # Arrange
    base_data = [{"obs": 1}, {"obs": 2}, {"obs": 3}]
    base_dataset = MockBaseDataset(base_data)
    robust_dataset = RobustDataset(base_dataset)

    # Act & Assert
    assert len(robust_dataset) == 3
    assert robust_dataset[0] == {"obs": 1}
    assert robust_dataset[1] == {"obs": 2}
    assert robust_dataset[2] == {"obs": 3}


def test_robust_dataset_skips_corrupt_and_retries(caplog):
    # Arrange
    # Index 1 is corrupt (raises ValueError). Indexes 0 and 2 are clean.
    base_data = [{"obs": 10}, ValueError("Corrupted frame"), {"obs": 30}]
    base_dataset = MockBaseDataset(base_data, episode_from=[0, 1, 2])
    robust_dataset = RobustDataset(base_dataset, max_retries=5)

    # Mock random.randint to return 2 when index 1 fails
    with patch("random.randint", return_value=2), caplog.at_level(logging.WARNING):
        # Act
        item = robust_dataset[1]

        # Assert
        assert item == {"obs": 30}
        # Verify a warning was logged indicating skipping of index 1
        assert any(
            "Skipping corrupt sample idx=1" in record.message
            for record in caplog.records
        )


def test_robust_dataset_warns_once_per_episode(caplog):
    # Arrange
    # Index 1 and 2 are corrupt and belong to episode 0 (indices 0 to 4)
    base_data = [
        {"obs": 0},
        ValueError("Corrupt index 1"),
        ValueError("Corrupt index 2"),
        {"obs": 3},
    ]
    base_dataset = MockBaseDataset(base_data, episode_from=[0])
    robust_dataset = RobustDataset(base_dataset, max_retries=5)

    # Mock random.randint to return 3 on retries
    with patch("random.randint", return_value=3), caplog.at_level(logging.WARNING):
        # Act
        # Get index 1 (fails, retries and succeeds, warns)
        item_1 = robust_dataset[1]
        # Get index 2 (fails, retries and succeeds, should NOT warn again for ep 0)
        item_2 = robust_dataset[2]

        # Assert
        assert item_1 == {"obs": 3}
        assert item_2 == {"obs": 3}
        
        # Verify exactly one warning was logged for episode 0
        warnings = [
            record.message
            for record in caplog.records
            if "Skipping corrupt sample" in record.message and "episode 0" in record.message
        ]
        assert len(warnings) == 1


def test_robust_dataset_raises_after_max_retries():
    # Arrange
    # All items are corrupt
    base_data = [
        ValueError("Always corrupt"),
        ValueError("Always corrupt"),
    ]
    base_dataset = MockBaseDataset(base_data)
    robust_dataset = RobustDataset(base_dataset, max_retries=3)

    # Act & Assert
    with pytest.raises(RuntimeError, match="Failed to load a valid sample after 3 retries"):
        _ = robust_dataset[0]


def test_robust_dataset_proxy_getattr():
    # Arrange
    base_dataset = MockBaseDataset([])
    robust_dataset = RobustDataset(base_dataset)

    # Act & Assert
    # Access attribute existing on base_dataset through the proxy
    assert robust_dataset.meta.features == "dummy_features"
    assert robust_dataset.meta.episodes["dataset_from_index"] == [0]


# ---------------------------------------------------------------------------
# CudaPrefetcher Tests
# ---------------------------------------------------------------------------

def test_cuda_prefetcher_basic():
    # Arrange
    device = torch.device("cpu")
    raw_loader = [
        {"data": torch.tensor([1.0]), "label": "a"},
        {"data": torch.tensor([2.0]), "label": "b"},
    ]

    with patch("torch.cuda.Stream", return_value=MockStream(device)), \
         patch("torch.cuda.stream", side_effect=mock_stream_context), \
         patch("torch.cuda.current_stream", return_value=MockCurrentStream(device)):
        
        prefetcher = CudaPrefetcher(raw_loader, device=device)
        
        # Act & Assert
        assert len(prefetcher) == 2
        
        iterator = iter(prefetcher)
        batch_1 = next(iterator)
        assert torch.equal(batch_1["data"], torch.tensor([1.0]))
        assert batch_1["label"] == "a"
        
        batch_2 = next(iterator)
        assert torch.equal(batch_2["data"], torch.tensor([2.0]))
        assert batch_2["label"] == "b"

        with pytest.raises(StopIteration):
            next(iterator)



# ---------------------------------------------------------------------------
# build_dataloader Tests
# ---------------------------------------------------------------------------

class DummyArgs:
    def __init__(self, **kwargs):
        self.num_workers = kwargs.get("num_workers", 0)
        self.prefetch_factor = kwargs.get("prefetch_factor", 2)
        self.prefetch_to_gpu = kwargs.get("prefetch_to_gpu", True)
        self.skip_bad_samples = kwargs.get("skip_bad_samples", True)
        self.batch_size = kwargs.get("batch_size", 1)


def test_build_dataloader_wraps_with_robust_dataset():
    # Arrange
    base_dataset = MockBaseDataset([{"obs": i} for i in range(10)])
    args = DummyArgs(skip_bad_samples=True)

    # Act
    loader = build_dataloader(base_dataset, args, shuffle=False)

    # Assert
    assert isinstance(loader.dataset, RobustDataset)
    assert loader.dataset.dataset is base_dataset


def test_build_dataloader_no_wrap_when_disabled():
    # Arrange
    base_dataset = MockBaseDataset([{"obs": i} for i in range(10)])
    args = DummyArgs(skip_bad_samples=False)

    # Act
    loader = build_dataloader(base_dataset, args, shuffle=False)

    # Assert
    assert not isinstance(loader.dataset, RobustDataset)
    assert loader.dataset is base_dataset



def test_build_dataloader_wraps_with_prefetcher_when_cuda():
    # Arrange
    base_dataset = MockBaseDataset([{"obs": i} for i in range(10)])
    args = DummyArgs(prefetch_to_gpu=True)
    device = torch.device("cpu")

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.Stream", return_value=MockStream(device)), \
         patch("torch.cuda.stream", side_effect=mock_stream_context), \
         patch("torch.cuda.current_stream", return_value=MockCurrentStream(device)):
        
        # Act
        loader = build_dataloader(base_dataset, args, device=device)

        # Assert
        assert isinstance(loader, CudaPrefetcher)


# ---------------------------------------------------------------------------
# Module Interactions Tests
# ---------------------------------------------------------------------------

def test_train_recap_nan_handling():
    # Simulate precomputed advantages array loaded in train_recap.py
    # NaN indices represent corrupt/skipped samples
    precomputed_advantages = np.array([1.5, np.nan, -0.5, np.nan, 2.0], dtype=np.float32)
    
    # Simulate a batch of sample indices
    batch_indices = np.array([0, 1, 2, 3, 4])

    # Apply the nan-to-num operation used in train_recap.py
    raw_adv = np.nan_to_num(precomputed_advantages[batch_indices], nan=0.0)

    # Assert NaNs are mapped to 0.0 and others are kept
    assert raw_adv[0] == 1.5
    assert raw_adv[1] == 0.0
    assert raw_adv[2] == -0.5
    assert raw_adv[3] == 0.0
    assert raw_adv[4] == 2.0


def test_nstep_td_advantages_pure_td():
    """With zero returns, the N-step TD advantage reduces to V(t+N) - V(t),
    and the episode-end case to -V(t).  Exercises the real canonical function."""
    from lerobot_policy_smolvla_rl.advantage_utils import (
        nstep_td_advantages,
        task_thresholds_from_advantages,
    )

    episode_lengths = np.array([5, 5])
    ep_from = np.array([0, 5])
    advantage_horizon = 2
    dataset_len = 10

    # vs_by_idx has some missing (corrupt) indices, e.g. index 8 is missing
    vs_by_idx = {
        0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5,
        5: 0.2, 6: 0.4, 7: 0.6,          9: 0.8,
    }
    meta_by_idx = {
        0: (0, 0, 1), 1: (0, 1, 1), 2: (0, 2, 1), 3: (0, 3, 1), 4: (0, 4, 1),
        5: (1, 0, 2), 6: (1, 1, 2), 7: (1, 2, 2),               9: (1, 4, 2),
    }
    # Zero returns -> A_t = (0 - V_t) - (0 - V_{t+N}) = V_{t+N} - V_t,
    # terminal -> 0 - V_t.
    returns_by_idx = {idx: 0.0 for idx in vs_by_idx}

    advantages = nstep_td_advantages(
        vs_by_idx, returns_by_idx, meta_by_idx, ep_from,
        episode_lengths, advantage_horizon, size=dataset_len,
    )

    # Ep 0:
    assert np.isclose(advantages[0], 0.2)  # V(2) - V(0) = 0.3 - 0.1
    assert np.isclose(advantages[1], 0.2)  # V(3) - V(1) = 0.4 - 0.2
    assert np.isclose(advantages[2], 0.2)  # V(4) - V(2) = 0.5 - 0.3
    assert np.isclose(advantages[3], -0.4) # Terminal: 0.0 - V(3)
    assert np.isclose(advantages[4], -0.5) # Terminal: 0.0 - V(4)
    # Ep 1:
    assert np.isclose(advantages[5], 0.4)  # V(7) - V(5) = 0.6 - 0.2
    assert np.isnan(advantages[6])         # Future frame (8) missing/corrupt -> NaN
    assert np.isclose(advantages[7], 0.2)  # V(9) - V(7) = 0.8 - 0.6
    assert np.isnan(advantages[8])         # Current frame (8) missing/corrupt -> NaN
    assert np.isclose(advantages[9], -0.8) # Terminal: 0.0 - V(9)

    # Default positive_fraction=0.3 -> threshold is the 70th percentile of
    # valid (non-NaN, successful) advantages per task, so ~30% land positive.
    task_thresholds = task_thresholds_from_advantages(advantages, meta_by_idx)
    expected_t1 = float(np.percentile([-0.5, -0.4, 0.2, 0.2, 0.2], 70))
    assert np.isclose(task_thresholds[1], expected_t1)
    expected_t2 = float(np.percentile([-0.8, 0.2, 0.4], 70))
    assert np.isclose(task_thresholds[2], expected_t2)

    # And explicit positive_fraction is honored (0.6 positive -> 40th percentile).
    t60 = task_thresholds_from_advantages(
        advantages, meta_by_idx, positive_fraction=0.6
    )
    assert np.isclose(t60[1], float(np.percentile([-0.5, -0.4, 0.2, 0.2, 0.2], 40)))


def test_nstep_td_advantages_includes_reward_term():
    """The full N-step TD advantage must include the return (reward) terms:
    A_t = (R_t - V_t) - (R_{t+N} - V_{t+N})."""
    from lerobot_policy_smolvla_rl.advantage_utils import nstep_td_advantages

    episode_lengths = np.array([4])
    ep_from = np.array([0])
    vs_by_idx = {0: 0.1, 1: 0.3, 2: 0.5, 3: 0.2}
    meta_by_idx = {0: (0, 0, 0), 1: (0, 1, 0), 2: (0, 2, 0), 3: (0, 3, 0)}
    returns_by_idx = {0: -0.9, 1: -0.6, 2: -0.3, 3: 0.0}

    advantages = nstep_td_advantages(
        vs_by_idx, returns_by_idx, meta_by_idx, ep_from,
        episode_lengths, advantage_horizon=2, size=4,
    )

    # Frame 0: (R_0 - V_0) - (R_2 - V_2) = (-0.9 - 0.1) - (-0.3 - 0.5) = -1.0 + 0.8
    assert np.isclose(advantages[0], -0.2)
    # Frame 1: (R_1 - V_1) - (R_3 - V_3) = (-0.6 - 0.3) - (0.0 - 0.2) = -0.9 + 0.2
    assert np.isclose(advantages[1], -0.7)
    # Frame 2: future frame 4 >= episode end -> MC: R_2 - V_2 = -0.3 - 0.5
    assert np.isclose(advantages[2], -0.8)
    # Frame 3: terminal -> MC: R_3 - V_3 = 0.0 - 0.2
    assert np.isclose(advantages[3], -0.2)
