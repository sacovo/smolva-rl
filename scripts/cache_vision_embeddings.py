#!/usr/bin/env python3
"""
Pre-compute and cache frozen SigLIP vision embeddings for critic training.

Since freeze_vision_encoder=True, the vision encoder output is deterministic
for each frame. Running this once lets train_critic.py skip both video decoding
and the vision forward pass, dramatically reducing the data bottleneck.

Usage:
    uv run python scripts/cache_vision_embeddings.py \
        --dataset_repo_id lerobot/droid_1.0.1 \
        --output_dir outputs/embedding_cache/droid_1.0.1 \
        --cameras observation.images.exterior_1_left observation.images.wrist_left \
        --batch_size 64 \
        --num_workers 8
"""

import argparse
import os

import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot_policy_smolvla_rl.smolvla_critic import SmolVLACrictic, SmolVLMCriticConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Cache frozen vision embeddings")
    parser.add_argument("--dataset_repo_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--model_id",
        type=str,
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    )
    parser.add_argument("--num_vlm_layers", type=int, default=8)
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=None,
        help="Camera keys to cache. Must match what train_critic.py will use.",
    )
    parser.add_argument(
        "--tolerance_s",
        type=float,
        default=0.01,
        help="Tolerance in seconds for timestamp matching",
    )
    parser.add_argument(
        "--video_backend",
        type=str,
        default="pyav",
        help="Backend for video decoding during this one-time cache run",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Subset of episodes to cache (default: all)",
    )
    parser.add_argument(
        "--limit_episodes",
        type=int,
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    episodes_to_load = args.episodes
    if episodes_to_load is None and args.limit_episodes is not None:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        meta = LeRobotDatasetMetadata(args.dataset_repo_id)
        episodes_to_load = list(range(min(args.limit_episodes, meta.total_episodes)))
        print(f"Limiting to first {len(episodes_to_load)} episodes")

    print(f"Loading dataset: {args.dataset_repo_id}")
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        episodes=episodes_to_load,
        tolerance_s=args.tolerance_s,
        video_backend=args.video_backend,
    )

    # ── Build camera map (same logic as train_critic.py) ─────────────────────
    camera_map = {}
    features = dataset_to_policy_features(dataset.meta.features)
    if args.cameras:
        dataset_cameras = sorted(
            [k for k in features if k.startswith("observation.images.")]
        )
        if any(cam not in features for cam in args.cameras):
            if len(dataset_cameras) == len(args.cameras):
                for src, dst in zip(dataset_cameras, args.cameras):
                    print(f"Mapping dataset camera '{src}' -> policy expected '{dst}'")
                    camera_map[src] = dst
            else:
                raise ValueError(
                    f"Cannot remap cameras: dataset has {len(dataset_cameras)} "
                    f"cameras ({dataset_cameras}), but requested {len(args.cameras)} "
                    f"cameras ({args.cameras})"
                )

    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }
    if camera_map:
        mapped_features = {}
        for k, v in input_features.items():
            mapped_features[camera_map.get(k, k)] = v
        input_features = mapped_features

    if args.cameras:
        all_cams = [k for k in input_features if k.startswith("observation.images.")]
        for cam in all_cams:
            if cam not in args.cameras:
                del input_features[cam]

    # ── Load model (vision encoder only needed) ───────────────────────────────
    print(f"Loading critic model from {args.model_id}")
    config = SmolVLMCriticConfig(
        num_bins=201,
        freeze_vision_encoder=True,
        num_vlm_layers=args.num_vlm_layers,
        input_features=input_features,
        device=device,
    )
    model = SmolVLACrictic(config).to(device)
    model.eval()

    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    pre, _ = make_smolvla_pre_post_processors(config, dataset.meta.stats)

    # ── DataLoader ────────────────────────────────────────────────────────────
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,          # keep deterministic index ordering
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    # ── Cache embeddings ──────────────────────────────────────────────────────
    # We cache: img_embs per camera, img_masks, lang_tokens, lang_masks
    # indexed by dataset sample index (batch["index"]).
    # Files are saved as individual .pt files per sample for random-access loading.
    # Alternatively: one big .pt per shard — here we use shards of 10k samples.

    SHARD_SIZE = 10_000
    total_cached = 0

    print(f"Caching vision embeddings to: {args.output_dir}")
    print(f"Total samples: {len(dataset)}")

    def flush_shard(shard_id, data):
        path = os.path.join(args.output_dir, f"shard_{shard_id:04d}.pt")
        torch.save(data, path)
        print(f"  Saved shard {shard_id} ({len(data)} samples) -> {path}")

    shard_id = 0
    shard_data = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Caching"):
            if camera_map:
                batch = {camera_map.get(k, k): v for k, v in batch.items()}

            processed = pre(batch)
            sample_indices = batch["index"].tolist()

            # Run vision encoder only
            images, img_masks = model.prepare_images(processed)

            # embed_image per camera
            emb_list = []
            for img in images:
                img = img.to(device)
                img_emb = model.vlm_with_expert.embed_image(img)
                img_emb_dim = img_emb.shape[-1]
                img_emb = img_emb * (img_emb_dim ** 0.5)
                emb_list.append(img_emb.cpu())

            img_masks_cpu = [m.cpu() for m in img_masks]

            from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
            lang_tokens = processed[OBS_LANGUAGE_TOKENS].cpu()
            lang_masks = processed[OBS_LANGUAGE_ATTENTION_MASK].cpu()

            for i, idx in enumerate(sample_indices):
                shard_data[idx] = {
                    "img_embs": [e[i] for e in emb_list],
                    "img_masks": [m[i] for m in img_masks_cpu],
                    "lang_tokens": lang_tokens[i],
                    "lang_masks": lang_masks[i],
                }
                total_cached += 1

            if len(shard_data) >= SHARD_SIZE:
                flush_shard(shard_id, shard_data)
                shard_id += 1
                shard_data = {}

    if shard_data:
        flush_shard(shard_id, shard_data)

    # Save metadata
    meta_path = os.path.join(args.output_dir, "meta.pt")
    torch.save({
        "dataset_repo_id": args.dataset_repo_id,
        "total_samples": total_cached,
        "num_shards": shard_id + 1,
        "cameras": args.cameras,
        "camera_map": camera_map,
        "model_id": args.model_id,
        "num_vlm_layers": args.num_vlm_layers,
    }, meta_path)

    print(f"\nDone. Cached {total_cached} samples across {shard_id + 1} shards.")
    print(f"Pass --embedding_cache_dir {args.output_dir} to train_critic.py to use the cache.")


if __name__ == "__main__":
    main()
