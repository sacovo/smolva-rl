#!/usr/bin/env python3
import argparse
import os
import sys
import torch

# Add src to sys.path to enable importing lerobot_policy_smolvla_rl
sys.path.append(os.path.join(os.getcwd(), "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot_policy_smolvla_rl import SmolVLARECAPConfig, SmolVLARECAPPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Convert raw SmolVLA-RECAP .pt checkpoint to LeRobot/Hugging Face format")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the raw PyTorch checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the exported LeRobot policy should be saved",
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        default="lerobot/droid_100",
        help="Dataset repo ID to extract features and action dimensions from (e.g. lerobot/droid_100 or downstream libero dataset)",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        help="VLM Model ID",
    )
    parser.add_argument(
        "--num_vlm_layers",
        type=int,
        default=-1,
        help="Number of VLM layers used (-1 for full backbone)",
    )
    parser.add_argument(
        "--num_fast_tokens",
        type=int,
        default=1024,
        help="Number of FAST tokens",
    )
    parser.add_argument(
        "--use_advantage_conditioning",
        action="store_true",
        default=True,
        help="Whether advantage conditioning is active",
    )
    parser.add_argument(
        "--no_advantage_conditioning",
        action="store_false",
        dest="use_advantage_conditioning",
        help="Disable advantage conditioning",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load weights on",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load dataset metadata to reconstruct input/output features
    print(f"Loading dataset metadata from: {args.dataset_repo_id}")
    dataset = LeRobotDataset(args.dataset_repo_id, episodes=[0])
    
    features = dataset_to_policy_features(dataset.meta.features)
    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }
    action_dim = features["action"].shape[0]

    print(f"Dataset features loaded. Action dimension: {action_dim}")
    print(f"Input features: {list(input_features.keys())}")

    # 2. Build Policy Configuration
    print("Building SmolVLARECAPConfig...")
    config = SmolVLARECAPConfig(
        num_vlm_layers=args.num_vlm_layers,
        num_fast_tokens=args.num_fast_tokens,
        use_advantage_conditioning=args.use_advantage_conditioning,
        model_id=args.model_id,
        max_action_dim=action_dim,
        input_features=input_features,
        action_stats=dataset.meta.stats.get("action"),
        chunk_size=1,
        n_action_steps=1,
        device=args.device,
    )
    config.output_features = output_features

    # 3. Instantiate Policy
    print("Instantiating SmolVLARECAPPolicy...")
    policy = SmolVLARECAPPolicy(config)

    # 4. Load weights from .pt state dict
    print(f"Loading state dict from: {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path, map_location=args.device)

    # Clean state dict to match local model structure (strip DDP/wrapper prefixes)
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("_orig_mod."):
            k = k[10:]
            
        # The policy expects weights mapped into the 'model.' attribute
        if not k.startswith("model."):
            k = "model." + k
            
        clean_state_dict[k] = v

    # Load weights into the policy module
    missing_keys, unexpected_keys = policy.load_state_dict(clean_state_dict, strict=False)
    print(f"Weights loaded successfully!")
    print(f"Missing keys (should be empty or minimal): {len(missing_keys)}")
    print(f"Unexpected keys (should be empty or minimal): {len(unexpected_keys)}")
    if len(missing_keys) > 0:
        print(f"Missing keys sample: {missing_keys[:10]}")
    if len(unexpected_keys) > 0:
        print(f"Unexpected keys sample: {unexpected_keys[:10]}")

    # 5. Export Policy to LeRobot/Hugging Face format (model.safetensors + config.json)
    print(f"Exporting policy to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    policy.save_pretrained(args.output_dir)
    print("Export completed successfully! Ready for simulated evaluation or Hugging Face upload.")


if __name__ == "__main__":
    main()
