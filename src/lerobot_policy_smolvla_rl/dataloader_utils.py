"""
dataloader_utils.py
-------------------
Shared DataLoader utilities used across all training and evaluation scripts.

Exports
-------
RobustDataset       — wraps a LeRobotDataset and silently retries on corrupt
                      video samples instead of crashing the whole job.
CudaPrefetcher      — overlaps host-to-device transfer of the next batch with
                      the current batch's forward/backward pass via a background
                      CUDA stream.
add_dataloader_args  — registers the shared DataLoader CLI flags on an argparse
                       ArgumentParser.
build_dataloader     — constructs a DataLoader from parsed args + dataset,
                       applying RobustDataset and CudaPrefetcher as configured.
patch_lerobot_dataset_reader — monkeypatches LeRobot's DatasetReader to cache
                       non-image/non-video columns in RAM for fast indexing.
"""

import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RobustDataset
# ---------------------------------------------------------------------------

class RobustDataset(torch.utils.data.Dataset):
    """Wraps a LeRobotDataset and silently skips samples whose video frames
    cannot be decoded (e.g. corrupted files in large community datasets like
    droid_1.0.1).  On a decode error the item is replaced by a random other
    index, up to ``max_retries`` attempts.

    A warning is emitted once per episode that contains a bad sample so the
    Slurm log remains searchable for data-quality issues.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, max_retries: int = 10):
        self.dataset = dataset
        self.max_retries = max_retries
        self._bad_episodes: set[int] = set()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name):
        # Transparent proxy: forward unknown attribute lookups to the wrapped
        # dataset so callers can access dataset.meta, dataset.meta.features, etc.
        return getattr(self.dataset, name)

    def __getitem__(self, idx):
        for _ in range(self.max_retries):
            try:
                return self.dataset[idx]
            except Exception as exc:
                ep_idx = None
                try:
                    import numpy as np
                    ep_from = np.array(self.dataset.meta.episodes["dataset_from_index"])
                    ep_idx = int(np.searchsorted(ep_from, idx, side="right") - 1)
                except Exception:
                    pass

                if ep_idx is not None and ep_idx not in self._bad_episodes:
                    self._bad_episodes.add(ep_idx)
                    logger.warning(
                        "Skipping corrupt sample idx=%d (episode %s): %s: %s",
                        idx,
                        ep_idx,
                        type(exc).__name__,
                        exc,
                    )

                idx = random.randint(0, len(self.dataset) - 1)

        raise RuntimeError(
            f"Failed to load a valid sample after {self.max_retries} retries. "
            "Dataset may have too many corrupt files."
        )


# ---------------------------------------------------------------------------
# CudaPrefetcher
# ---------------------------------------------------------------------------

class CudaPrefetcher:
    """Overlaps the next batch's host-to-device transfer with the current
    batch's forward/backward pass using a background CUDA stream.

    Drop-in wrapper around any DataLoader::

        loader = CudaPrefetcher(dataloader, device)
        for batch in loader:
            ...  # batch tensors are already on *device*
    """

    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self._batch = None

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        self._iter = iter(self.loader)
        self._preload()
        return self

    def _preload(self):
        try:
            raw = next(self._iter)
        except StopIteration:
            self._batch = None
            return
        with torch.cuda.stream(self.stream):
            self._batch = {
                k: v.to(self.device, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in raw.items()
            }

    def __next__(self):
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self._batch
        if batch is None:
            raise StopIteration
        self._preload()  # kick off transfer of the NEXT batch in the background
        return batch




# ---------------------------------------------------------------------------
# Shared argparse helpers
# ---------------------------------------------------------------------------

def add_dataloader_args(parser) -> None:
    """Register shared DataLoader flags onto *parser* (mutates in-place)."""
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="Number of batches to prefetch per DataLoader worker "
             "(ignored when num_workers=0)",
    )
    parser.add_argument(
        "--prefetch_to_gpu",
        action="store_true",
        default=True,
        help="Use a background CUDA stream to prefetch the next batch to GPU "
             "while the current batch is being processed (default: True)",
    )
    parser.add_argument(
        "--no_prefetch_to_gpu",
        dest="prefetch_to_gpu",
        action="store_false",
        help="Disable CUDA prefetching",
    )
    parser.add_argument(
        "--skip_bad_samples",
        action="store_true",
        default=True,
        help="Silently skip corrupt video samples instead of crashing "
             "(default: True)",
    )
    parser.add_argument(
        "--no_skip_bad_samples",
        dest="skip_bad_samples",
        action="store_false",
        help="Disable corrupt-sample skipping (crash on first bad file)",
    )


def build_dataloader(
    dataset: torch.utils.data.Dataset,
    args,
    *,
    shuffle: bool = True,
    device: torch.device | None = None,
) -> DataLoader | CudaPrefetcher:
    """Build a DataLoader (+ optional wrappers) from parsed CLI *args*.

    Parameters
    ----------
    dataset:
        The base dataset.  If ``args.skip_bad_samples`` is True, it will be wrapped in :class:`RobustDataset`.
    args:
        Parsed argparse namespace.  Must contain the flags registered by
        :func:`add_dataloader_args`.
    shuffle:
        Whether to shuffle the DataLoader.
    device:
        Target CUDA device for :class:`CudaPrefetcher`.  If *None*, the
        prefetcher is disabled even when ``args.prefetch_to_gpu`` is True.
    """
    # Optionally wrap for robustness against corrupt video files
    if args.skip_bad_samples:
        if not isinstance(dataset, RobustDataset):
            dataset = RobustDataset(dataset, max_retries=10)
            logger.info(
                "RobustDataset enabled: corrupt samples will be skipped automatically."
            )

    num_workers = args.num_workers
    prefetch_factor = args.prefetch_factor if num_workers > 0 else None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=(num_workers > 0),
    )

    # Optionally wrap with CUDA prefetcher
    if args.prefetch_to_gpu and torch.cuda.is_available() and device is not None:
        logger.info(
            "CudaPrefetcher enabled: next batch will be transferred to GPU in background."
        )
        return CudaPrefetcher(loader, device)

    return loader


# ---------------------------------------------------------------------------
# LeRobot DatasetReader caching patch
# ---------------------------------------------------------------------------

def patch_lerobot_dataset_reader():
    """Patches LeRobot's DatasetReader to cache non-image/non-video columns in RAM.
    This avoids extremely slow disk reads and PNG/JPEG decoding during delta-timestamp queries.
    """
    try:
        import time
        from lerobot.datasets.dataset_reader import DatasetReader

        # Avoid double patching
        if getattr(DatasetReader, "_is_patched_for_caching", False):
            return

        orig_try_load = DatasetReader.try_load
        orig_load_and_activate = DatasetReader.load_and_activate

        def patched_init_cache(self):
            if self.hf_dataset is None:
                return
            if not hasattr(self, "_cached_columns"):
                logger.info("Caching non-image/non-video columns in RAM for high-speed indexing...")
                t_start = time.time()
                self._cached_columns = {}
                for key in self.hf_dataset.column_names:
                    # Check if it's an image or video key
                    is_video_or_image = False
                    if key in self._meta.video_keys:
                        is_video_or_image = True
                    elif key in self._meta.features:
                        ftype = self._meta.features[key].get("dtype")
                        if ftype in ("image", "video"):
                            is_video_or_image = True

                    if not is_video_or_image:
                        # Cache the column as a torch tensor
                        with self.hf_dataset.formatted_as(type="numpy", columns=[key]):
                            col_data_np = self.hf_dataset[:][key]

                        if isinstance(col_data_np, np.ndarray):
                            col_data = torch.from_numpy(col_data_np)
                        else:
                            col_data = torch.tensor(col_data_np)
                        self._cached_columns[key] = col_data
                logger.info(f"Cached {list(self._cached_columns.keys())} in {time.time() - t_start:.4f}s")

        def patched_try_load(self):
            res = orig_try_load(self)
            if res:
                patched_init_cache(self)
            return res

        def patched_load_and_activate(self):
            orig_load_and_activate(self)
            patched_init_cache(self)

        def patched_query_hf_dataset(self, query_indices):
            result = {}
            for key, q_idx in query_indices.items():
                if key in self._meta.video_keys:
                    continue
                relative_indices = (
                    q_idx
                    if self._absolute_to_relative_idx is None
                    else [self._absolute_to_relative_idx[idx] for idx in q_idx]
                )

                if hasattr(self, "_cached_columns") and key in self._cached_columns:
                    cache = self._cached_columns[key]
                    if isinstance(cache, torch.Tensor):
                        result[key] = cache[relative_indices]
                    else:
                        result[key] = torch.tensor([cache[i] for i in relative_indices])
                else:
                    # Fallback to original querying
                    try:
                        result[key] = torch.stack(self.hf_dataset[key][relative_indices])
                    except (KeyError, TypeError, IndexError):
                        result[key] = torch.stack(self.hf_dataset[relative_indices][key])
            return result

        DatasetReader.try_load = patched_try_load
        DatasetReader.load_and_activate = patched_load_and_activate
        DatasetReader._query_hf_dataset = patched_query_hf_dataset
        DatasetReader._is_patched_for_caching = True
        logger.info("Successfully patched LeRobot DatasetReader for fast column caching.")
    except Exception as e:
        logger.error(f"Failed to patch LeRobot DatasetReader: {e}")
