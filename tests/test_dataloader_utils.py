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
    streaming_collate_fn,
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
# streaming_collate_fn Tests
# ---------------------------------------------------------------------------

def test_streaming_collate_fn():
    # Arrange
    img1 = Image.new("RGB", (10, 10), color="red")
    img2 = Image.new("RGB", (10, 10), color="blue")
    
    batch = [
        {"image": img1, "label": torch.tensor(1), "info": "first"},
        {"image": img2, "label": torch.tensor(2), "info": "second"},
    ]

    # Act
    collated = streaming_collate_fn(batch)

    # Assert
    # Images should be converted to torch tensors of shape [3, 10, 10]
    assert isinstance(collated["image"], torch.Tensor)
    assert collated["image"].shape == (2, 3, 10, 10)
    assert collated["image"].dtype == torch.float32
    
    # Values should be scaled to [0, 1]
    assert collated["image"].min() >= 0.0
    assert collated["image"].max() <= 1.0
    
    # Labels should be standard collated tensors
    assert torch.equal(collated["label"], torch.tensor([1, 2]))
    
    # Non-tensor/non-image values collate to a list of strings
    assert collated["info"] == ["first", "second"]


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


def test_build_dataloader_no_wrap_for_streaming():
    # Arrange
    base_dataset = MockBaseDataset([{"obs": i} for i in range(10)])
    args = DummyArgs(skip_bad_samples=True)

    # Act
    loader = build_dataloader(base_dataset, args, shuffle=False, is_streaming=True)

    # Assert
    assert not isinstance(loader.dataset, RobustDataset)


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


def test_compute_thresholds_advantage_logic():
    # Arrange
    episode_lengths = np.array([5, 5])
    ep_from = np.array([0, 5])
    action_chunk_size = 2
    dataset_len = 10

    # vs_by_idx has some missing (corrupt) indices, e.g. index 8 is missing
    vs_by_idx = {
        0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5,
        5: 0.2, 6: 0.4, 7: 0.6,          9: 0.8
    }
    meta_by_idx = {
        0: (0, 0, 1), 1: (0, 1, 1), 2: (0, 2, 1), 3: (0, 3, 1), 4: (0, 4, 1),
        5: (1, 0, 2), 6: (1, 1, 2), 7: (1, 2, 2),               9: (1, 4, 2)
    }

    # Act
    # Replicate Step 2/3 and 3/3 of compute_thresholds.py
    advantages = np.full(dataset_len, np.nan, dtype=np.float32)

    for sample_idx, v_current in vs_by_idx.items():
        ep_idx, frame_idx, _ = meta_by_idx[sample_idx]
        ep_len = episode_lengths[ep_idx]
        future_frame = frame_idx + action_chunk_size

        if future_frame >= ep_len:
            advantages[sample_idx] = 0.0 - v_current
        else:
            future_sample_idx = int(ep_from[ep_idx]) + future_frame
            if future_sample_idx in vs_by_idx:
                advantages[sample_idx] = vs_by_idx[future_sample_idx] - v_current

    # Assert advantages are calculated properly, skipping NaNs and handling terminals
    # Ep 0:
    assert np.isclose(advantages[0], 0.2)  # V(2) - V(0) = 0.3 - 0.1 = 0.2
    assert np.isclose(advantages[1], 0.2)  # V(3) - V(1) = 0.4 - 0.2 = 0.2
    assert np.isclose(advantages[2], 0.2)  # V(4) - V(2) = 0.5 - 0.3 = 0.2
    assert np.isclose(advantages[3], -0.4) # Terminal: 0.0 - V(3) = 0.0 - 0.4 = -0.4
    assert np.isclose(advantages[4], -0.5) # Terminal: 0.0 - V(4) = 0.0 - 0.5 = -0.5
    # Ep 1:
    assert np.isclose(advantages[5], 0.4)  # V(7) - V(5) = 0.6 - 0.2 = 0.4
    assert np.isnan(advantages[6])         # Future frame (8) is missing/corrupt -> NaN
    assert np.isclose(advantages[7], 0.2)  # V(9) - V(7) = 0.8 - 0.6 = 0.2
    assert np.isnan(advantages[8])         # Current frame (8) is missing/corrupt -> NaN
    assert np.isclose(advantages[9], -0.8) # Terminal: 0.0 - V(9) = 0.0 - 0.8 = -0.8

    # ── Replicate Threshold 30th percentile calculation (Step 3/3 of compute_thresholds) ───
    task_thresholds = {}
    all_tasks = np.array(
        [meta_by_idx[idx][2] for idx in sorted(vs_by_idx.keys())], dtype=np.int64
    )
    all_advs = advantages[sorted(vs_by_idx.keys())]

    unique_tasks = np.unique(all_tasks)
    for t in unique_tasks:
        mask = all_tasks == t
        task_advs = all_advs[mask]
        valid = task_advs[~np.isnan(task_advs)]
        if len(valid) == 0:
            continue
        task_thresholds[int(t)] = float(np.percentile(valid, 30))

    # Task 1 valid advantages: [0.2, 0.2, 0.2, -0.4, -0.5]
    expected_t1 = float(np.percentile([-0.5, -0.4, 0.2, 0.2, 0.2], 30))
    assert np.isclose(task_thresholds[1], expected_t1)

    # Task 2 valid advantages: [0.4, 0.2, -0.8] (excluding NaNs at indices 6 and 8)
    expected_t2 = float(np.percentile([-0.8, 0.2, 0.4], 30))
    assert np.isclose(task_thresholds[2], expected_t2)
