#!/usr/bin/env python3
import os
import sys
import json
import torch
# Add src to python path to enable loading local modules
sys.path.append(os.path.join(os.getcwd(), "src"))

# Side-effect import: registers the custom `smolvla_recap` policy/config with
# the LeRobot / Draccus choice registries so make_policy() below can resolve a
# smolvla_recap checkpoint. (pyflakes reports this as unused — it is not; the
# import is kept for its registration side effect.)
import lerobot_policy_smolvla_rl  # noqa: F401

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors

def run_verification(dataset_name, policy_path, output_json):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running verification on device: {device}")
    
    results = {}
    
    # 1. Load Dataset
    print(f"Loading dataset {dataset_name}...")
    try:
        dataset = LeRobotDataset(dataset_name)
        results["dataset_load"] = "SUCCESS"
    except Exception as e:
        results["dataset_load"] = f"FAILED: {str(e)}"
        print(f"Failed to load dataset: {e}")
        with open(output_json, "w") as f:
            json.dump(results, f, indent=4)
        return

    # 2. Check Dataset Statistics
    state_stats = dataset.meta.stats.get("observation.state")
    if state_stats is not None:
        # Check standard deviation of indices 6 & 7 (gripper / orientations)
        # Raw LIBERO states have very small gripper std (~0.014).
        # Double-normalized datasets have std ~1.0.
        std_6 = float(state_stats["std"][6])
        std_7 = float(state_stats["std"][7])
        is_raw_state = (std_6 < 0.1 and std_7 < 0.1)
        results["dataset_state_stats"] = {
            "mean": [float(x) for x in state_stats["mean"]],
            "std": [float(x) for x in state_stats["std"]],
            "is_raw_state_coordinates": is_raw_state
        }
        print(f"Dataset State Std (Indices 6, 7): {std_6:.4f}, {std_7:.4f} (Is Raw: {is_raw_state})")
    else:
        results["dataset_state_stats"] = "MISSING"
        is_raw_state = False

    # 3. Load Policy
    print(f"Loading policy from {policy_path}...")
    try:
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
        policy_cfg.device = device
        policy = make_policy(policy_cfg, ds_meta=dataset.meta)
        policy.to(device)
        policy.eval()
        
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": device}}
        )
        results["policy_load"] = "SUCCESS"
    except Exception as e:
        results["policy_load"] = f"FAILED: {str(e)}"
        print(f"Failed to load policy: {e}")
        with open(output_json, "w") as f:
            json.dump(results, f, indent=4)
        return

    # 4. Compare Normalization Alignment on Sample Frames
    # Frame 0 of Episode 0 (success episode)
    sample_indices = [0, 30, 60]
    alignment_tests = []
    
    for idx in sample_indices:
        sample = dataset[idx]
        
        # Simulating Training Dataloader State (Normalized using dataset statistics)
        # LeRobotDataset returns already normalized observations when loading via dataloader.
        # But here we load the raw item and pass it to the preprocessor.
        batch = {
            "observation.images.image": sample["observation.images.image"].unsqueeze(0),
            "observation.state": sample["observation.state"].unsqueeze(0),
            "task_index": sample["task_index"].unsqueeze(0),
            "task": [sample["task"]]
        }
        
        # Preprocess using the evaluation-time preprocessor
        with torch.no_grad():
            preprocessed = preprocessor(batch)
            preprocessed = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in preprocessed.items()
            }
            
            # Run inference
            pred_actions = {}
            for cfg_w in [1.0, 1.5]:
                batch_copy = {k: v for k, v in preprocessed.items()}
                pred = policy.select_action(batch_copy, cfg_weight=cfg_w)
                if isinstance(pred, torch.Tensor):
                    pred = pred.cpu().numpy()
                pred_actions[f"cfg_{cfg_w}"] = [float(x) for x in pred[0]]
        
        gt_action = [float(x) for x in sample["action"].numpy()]
        
        # State values
        raw_state = [float(x) for x in sample["observation.state"].numpy()]
        norm_state = [float(x) for x in preprocessed["observation.state"][0].cpu().numpy()]
        
        alignment_tests.append({
            "frame_index": idx,
            "raw_state": raw_state,
            "normalized_state": norm_state,
            "ground_truth_action": gt_action,
            "predicted_actions": pred_actions
        })
    
    results["alignment_tests"] = alignment_tests
    
    # 5. Overall Pipeline Check Status
    # Pipeline is PASSED if:
    # 1. Dataset statistics represent raw coordinates (std of gripper is small, ~0.014).
    # 2. Predicted actions are within normal bounds.
    all_passed = is_raw_state
    
    # Check if actions look sane (not exploding, e.g., absolute values < 5.0)
    for test in alignment_tests:
        for cfg_name, action in test["predicted_actions"].items():
            if any(abs(x) > 5.0 for x in action):
                all_passed = False
                print(f"WARNING: Action exploding in {cfg_name}: {action}")
                
    results["pipeline_check_status"] = "PASSED" if all_passed else "FAILED"
    
    print(f"Writing results to {output_json}...")
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
    print("Done!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Verify pipeline representation and normalization alignment.")
    parser.add_argument("--dataset", type=str, default="sancov/smolvla_recap_libero_spatial")
    parser.add_argument("--policy", type=str, required=True, help="Path to policy checkpoint directory")
    parser.add_argument("--output", type=str, default="pipeline_verification.json", help="Path to output JSON results file")
    args = parser.parse_args()
    
    run_verification(args.dataset, args.policy, args.output)
