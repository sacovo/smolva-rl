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
        "--action_chunk_size",
        type=int,
        default=50,
        help="Action chunk size used during training/inference",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=1,
        help="Number of action steps to execute per policy call during inference",
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

    action_stats = dataset.meta.stats.get("action")
    if action_stats is not None:
        import numpy as np
        action_stats = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in action_stats.items()
        }

    # 2. Build Policy Configuration
    print("Building SmolVLARECAPConfig...")
    config = SmolVLARECAPConfig(
        num_vlm_layers=args.num_vlm_layers,
        num_fast_tokens=args.num_fast_tokens,
        use_advantage_conditioning=args.use_advantage_conditioning,
        model_id=args.model_id,
        max_action_dim=action_dim,
        input_features=input_features,
        action_stats=action_stats,
        chunk_size=args.action_chunk_size,
        n_action_steps=args.n_action_steps,
        device=args.device,
    )
    config.output_features = output_features

    # 3. Instantiate Policy
    print("Instantiating SmolVLARECAPPolicy...")
    policy = SmolVLARECAPPolicy(config)

    # 4. Load weights from state dict (.safetensors or .pt or a folder containing model.safetensors/pytorch_model.bin)
    print(f"Loading state dict from: {args.checkpoint_path}")
    if os.path.isdir(args.checkpoint_path):
        safetensors_file = os.path.join(args.checkpoint_path, "model.safetensors")
        bin_file = os.path.join(args.checkpoint_path, "pytorch_model.bin")
        if os.path.exists(safetensors_file):
            print("Found model.safetensors inside checkpoint directory.")
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_file, device=args.device)
        elif os.path.exists(bin_file):
            print("Found pytorch_model.bin inside checkpoint directory.")
            state_dict = torch.load(bin_file, map_location=args.device)
        else:
            raise FileNotFoundError(
                f"Could not find model.safetensors or pytorch_model.bin in directory: {args.checkpoint_path}"
            )
    else:
        # It's a file
        if args.checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(args.checkpoint_path, device=args.device)
        else:
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
    print("Weights loaded successfully!")
    print(f"Missing keys (should be empty or minimal): {len(missing_keys)}")
    print(f"Unexpected keys (should be empty or minimal): {len(unexpected_keys)}")
    if len(missing_keys) > 0:
        print(f"Missing keys sample: {missing_keys[:10]}")
    if len(unexpected_keys) > 0:
        print(f"Unexpected keys sample: {unexpected_keys[:10]}")

    # 5. Export base policy to a temporary folder
    exported_dir = os.path.join(args.output_dir, "raw")
    print(f"Exporting base policy to: {exported_dir}")
    os.makedirs(exported_dir, exist_ok=True)
    policy.save_pretrained(exported_dir)

    # 6. Run migration script to produce the pipeline-based format
    import subprocess
    print("Migrating policy to the newer LeRobot format...")
    cmd = [
        sys.executable,
        "-c",
        "import lerobot_policy_smolvla_rl; from lerobot.processor.migrate_policy_normalization import main; main()",
        "--pretrained-path", exported_dir,
        "--output-dir", args.output_dir
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Policy successfully migrated to: {args.output_dir}")

    # 7. Load the migrated processors and inject the actual dataset statistics
    print("Populating migrated processors with dataset statistics...")
    stats_tensors = {}
    if dataset.meta.stats is not None:
        for feature_name, stat_dict in dataset.meta.stats.items():
            stats_tensors[feature_name] = {}
            for stat_name, val in stat_dict.items():
                stats_tensors[feature_name][stat_name] = torch.tensor(val)
    
    from lerobot.policies.factory import make_pre_post_processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=args.output_dir,
    )
    # Inject stats into normalizer/unnormalizer steps via load_state_dict
    # which properly updates both stats and _tensor_stats for serialization.
    flat_stats = {}
    for feat_name, stat_dict in stats_tensors.items():
        for stat_name, val in stat_dict.items():
            flat_stats[f"{feat_name}.{stat_name}"] = val
    for step in preprocessor.steps:
        if hasattr(step, 'load_state_dict') and hasattr(step, 'norm_map'):
            step.load_state_dict(flat_stats)
    for step in postprocessor.steps:
        if hasattr(step, 'load_state_dict') and hasattr(step, 'norm_map'):
            step.load_state_dict(flat_stats)
    preprocessor.save_pretrained(args.output_dir)
    postprocessor.save_pretrained(args.output_dir)
    print(f"Successfully populated migrated processors with statistics in: {args.output_dir}")

    # 8. Clean up raw temporary folder
    import shutil
    print(f"Cleaning up raw temporary folder: {exported_dir}")
    shutil.rmtree(exported_dir, ignore_errors=True)

    print("Export and migration completed successfully! Ready for simulated evaluation or Hugging Face upload.")


if __name__ == "__main__":
    main()
