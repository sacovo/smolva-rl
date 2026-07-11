"""Latency benchmark for SmolVLA-RECAP on edge hardware (Jetson Orin Nano).

Times action-chunk generation for the multi-step flow-matching expert vs the
SnapFlow one-step path on the same randomly initialized policy. Weights do not
affect latency, so no checkpoint is needed -- only a config.json describing the
deployed architecture (cameras, chunk size, VLM depth).

Usage:
    python scripts/bench_jetson.py --config_dir outputs/libero_recap_250000 \
        [--device cuda] [--iters 30]
"""

import argparse
import contextlib
import statistics
import time

import torch

from lerobot.configs.policies import PreTrainedConfig

# import registers the "smolvla_recap" config type with the lerobot registry
from lerobot_policy_smolvla_rl.configuration_smolvla_recap import (  # noqa: F401
    SmolVLARECAPConfig,
)
from lerobot_policy_smolvla_rl.modeling_smolvla_recap import SmolVLARECAPPolicy


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config_dir", required=True, help="Dir containing config.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--lang_len", type=int, default=48)
    p.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Policy dtype; bfloat16 halves memory (needed on <8GB unified memory)",
    )
    return p.parse_args()


def make_batch(config, device, lang_len):
    batch = {}
    for key, ft in config.input_features.items():
        if key.startswith("observation.images."):
            batch[key] = torch.rand(1, *ft.shape, device=device)
        elif key == "observation.state":
            batch[key] = torch.randn(1, *ft.shape, device=device)
    batch["observation.language.tokens"] = torch.randint(
        0, 1000, (1, lang_len), device=device
    )
    batch["observation.language.attention_mask"] = torch.ones(
        1, lang_len, dtype=torch.bool, device=device
    )
    return batch


def time_chunks(policy, batch, iters, warmup, device, dtype):
    if dtype == "float32":
        autocast = contextlib.nullcontext()
    else:
        # non-fp32 weights: autocast reconciles the fp32 noise/state tensors
        # lerobot creates internally with the reduced-precision weights
        autocast = torch.autocast(device.split(":")[0], dtype=getattr(torch, dtype))
    for _ in range(warmup):
        with torch.no_grad(), autocast:
            policy.predict_action_chunk(dict(batch))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        with torch.no_grad(), autocast:
            policy.predict_action_chunk(dict(batch))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return times


def report(label, times, chunk_size):
    mean = statistics.mean(times)
    std = statistics.stdev(times)
    print(
        f"{label}: {mean:.1f} +/- {std:.1f} ms/chunk "
        f"(median {statistics.median(times):.1f}, n={len(times)}) "
        f"-> {mean / chunk_size:.1f} ms/action, "
        f"{1000 / mean:.2f} chunks/s"
    )
    return mean


def main():
    args = parse_args()
    config = PreTrainedConfig.from_pretrained(args.config_dir)
    config.device = args.device

    print(f"torch {torch.__version__}, device: ", end="")
    if args.device.startswith("cuda"):
        print(torch.cuda.get_device_name(0))
    else:
        print("cpu")
    print(
        f"config: {config.num_vlm_layers} VLM layers, chunk {config.chunk_size}, "
        f"{config.num_steps} FM steps, "
        f"{len([k for k in config.input_features if 'images' in k])} cameras"
    )

    policy = SmolVLARECAPPolicy(config).to(getattr(torch, args.dtype)).to(args.device)
    policy.eval()
    n_params = sum(p.numel() for p in policy.parameters()) / 1e6
    print(f"policy params: {n_params:.0f}M (random init), dtype {args.dtype}")

    batch = make_batch(config, args.device, args.lang_len)

    policy.config.snapflow_enabled = False
    fm = time_chunks(policy, batch, args.iters, args.warmup, args.device, args.dtype)
    mean_fm = report(f"flow matching ({config.num_steps} steps)", fm, config.chunk_size)

    policy.config.snapflow_enabled = True
    sf = time_chunks(policy, batch, args.iters, args.warmup, args.device, args.dtype)
    mean_sf = report("SnapFlow (1 step)", sf, config.chunk_size)

    print(f"speed-up: {mean_fm / mean_sf:.2f}x")


if __name__ == "__main__":
    main()
