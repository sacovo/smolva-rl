"""Pin the paper-critical advantage-threshold percentile behavior.

``task_thresholds_from_advantages`` sets the per-task threshold ε_ℓ so that
approximately ``positive_fraction`` of valid frames are labeled positive
(advantage > ε_ℓ). This corresponds to the ``100 * (1 - positive_fraction)``
percentile of the advantage distribution — e.g. ``positive_fraction=0.3`` maps
to the **70th** percentile (top 30% positive). These tests lock that mapping in
place, along with the exclusion of failed and NaN frames.
"""
import numpy as np
import torch

from lerobot_policy_smolvla_rl.advantage_utils import (
    task_thresholds_from_advantages,
)


def _single_task_meta(n, task_idx=0):
    """One episode per frame, all belonging to ``task_idx``."""
    return {i: (i, i, task_idx) for i in range(n)}


def test_default_positive_fraction_is_70th_percentile():
    # positive_fraction defaults to 0.3 -> 70th percentile of the advantages.
    advs = np.arange(100, dtype=np.float32)  # 0..99
    meta = _single_task_meta(100)

    thresholds = task_thresholds_from_advantages(advs, meta)

    assert set(thresholds) == {0}
    expected = float(np.percentile(advs, 70.0))
    assert thresholds[0] == expected


def test_positive_fraction_controls_labeled_share():
    # ~positive_fraction of valid frames should end up above the threshold.
    advs = np.arange(1000, dtype=np.float32)
    meta = _single_task_meta(1000)

    for pos_frac in (0.3, 0.4, 0.5):
        thr = task_thresholds_from_advantages(advs, meta, positive_fraction=pos_frac)[0]
        share_positive = float(np.mean(advs > thr))
        assert abs(share_positive - pos_frac) < 0.02


def test_failed_episodes_are_excluded():
    # Failed episodes carry very negative advantages; they must not drag the
    # threshold down. success_flags is indexed by episode index.
    advs = np.concatenate(
        [np.arange(50, dtype=np.float32), np.full(50, -1000.0, dtype=np.float32)]
    )
    meta = _single_task_meta(100)
    success = torch.tensor([True] * 50 + [False] * 50)

    thr = task_thresholds_from_advantages(advs, meta, success_flags=success)[0]
    # Only the successful 0..49 frames count -> 70th percentile of those.
    expected = float(np.percentile(np.arange(50, dtype=np.float32), 70.0))
    assert thr == expected


def test_nan_frames_are_excluded():
    advs = np.arange(100, dtype=np.float32)
    advs[::2] = np.nan  # corrupt half the frames
    meta = _single_task_meta(100)

    thr = task_thresholds_from_advantages(advs, meta)[0]
    valid = advs[~np.isnan(advs)]
    assert thr == float(np.percentile(valid, 70.0))


def test_task_with_no_valid_frames_defaults_to_zero():
    advs = np.array([np.nan, np.nan], dtype=np.float32)
    meta = _single_task_meta(2, task_idx=7)

    thresholds = task_thresholds_from_advantages(advs, meta)
    assert thresholds == {7: 0.0}


def test_per_task_thresholds_are_independent():
    # Task 0 spans 0..99, task 1 spans 1000..1099.
    advs = np.concatenate(
        [np.arange(100, dtype=np.float32), 1000 + np.arange(100, dtype=np.float32)]
    )
    meta = {}
    for i in range(100):
        meta[i] = (i, i, 0)
    for i in range(100, 200):
        meta[i] = (i, i, 1)

    thresholds = task_thresholds_from_advantages(advs, meta)
    assert thresholds[0] == float(np.percentile(np.arange(100, dtype=np.float32), 70.0))
    assert thresholds[1] == float(
        np.percentile(1000 + np.arange(100, dtype=np.float32), 70.0)
    )
