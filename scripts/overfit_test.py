#!/usr/bin/env python3
import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

# Add src to python path to enable loading local modules
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from lerobot_policy_smolvla_rl import SmolVLARECAPPolicy, SmolVLARECAPConfig, SmolVLARECAP
except ImportError:
    pass

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors

def run_overfit_test(dataset_name, pretrained_model_path, output_dir, steps=250, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running overfit test on: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    json_output_path = os.path.join(output_dir, "overfit_results.json")
    
    results = {
        "losses": [],
        "errors": []
    }
    
    # 1. Load Dataset and slice Episode 0
    print("Loading dataset...")
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot_policy_smolvla_rl import SmolVLARECAPConfig
    
    ds_meta = LeRobotDatasetMetadata(dataset_name)
    policy_cfg = PreTrainedConfig.from_pretrained(pretrained_model_path)
    policy_cfg.device = device
    
    dummy_config = SmolVLARECAPConfig(chunk_size=policy_cfg.chunk_size, n_action_steps=1)
    delta_timestamps = resolve_delta_timestamps(dummy_config, ds_meta)
    
    dataset = LeRobotDataset(dataset_name, delta_timestamps=delta_timestamps)
    
    from_idx = int(dataset.meta.episodes[0]["dataset_from_index"])
    to_idx = int(dataset.meta.episodes[0]["dataset_to_index"])
    print(f"Overfitting on Episode 0: frames {from_idx} to {to_idx} (length: {to_idx - from_idx})")
    
    episode_dataset = Subset(dataset, range(from_idx, to_idx))
    dataloader = DataLoader(episode_dataset, batch_size=4, shuffle=True)
    
    # 2. Initialize Model
    print(f"Loading base config/model from: {pretrained_model_path}")
    
    # Instantiate the model directly for training
    from lerobot_policy_smolvla_rl import SmolVLARECAPConfig
    model_config = SmolVLARECAPConfig(
        num_vlm_layers=policy_cfg.num_vlm_layers,
        chunk_size=policy_cfg.chunk_size,
        max_action_dim=policy_cfg.max_action_dim,
        input_features=policy_cfg.input_features,
        device=device,
        use_advantage_conditioning=True,
        adv_dropout_rate=0.0,
        # Only optimise the first action step — inference uses n_action_steps=1
        # so computing FM loss over the full chunk wastes capacity and slows
        # convergence by chunk_size × for no gain in what the test measures.
        loss_n_action_steps=1,
    )
    model_config.output_features = policy_cfg.output_features
    
    # Load SmolVLARECAP weights
    model = SmolVLARECAP(model_config).to(device)
    # Load weights from the checkpoint state dict
    state_dict_path = os.path.join(pretrained_model_path, "model.safetensors")
    if os.path.exists(state_dict_path):
        from safetensors.torch import load_file
        sd = load_file(state_dict_path)
        # Strip model. prefix if present
        clean_sd = {}
        for k, v in sd.items():
            if k.startswith("model."):
                k = k[6:]
            clean_sd[k] = v
        model.load_state_dict(clean_sd, strict=False)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    
    # 3. Training Loop (Overfitting)
    print("Starting training...")
    pre_train = model.get_pre_processor(dataset)
    step = 0
    while step < steps:
        for batch in dataloader:
            if step >= steps:
                break
            
            # Preprocess batch (tokenizes language tasks, applies correct stats, etc.)
            batch = pre_train(batch)
            
            # Prepare batch for model (move tensors to device)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            # Force advantage conditioning to True (positive advantage)
            advantage_bool = [True] * batch["observation.state"].shape[0]
            
            optimizer.zero_grad()
            loss, ar_loss, flow_loss = model.compute_loss(batch, advantage=advantage_bool)
            loss.backward()
            optimizer.step()
            
            loss_val = float(loss.item())
            ar_val = float(ar_loss.item())
            flow_val = float(flow_loss.item())
            results["losses"].append({"step": step, "loss": loss_val})
            if step % 50 == 0:
                print(f"Step {step}/{steps} - Loss: {loss_val:.4f} (AR: {ar_val:.4f}, FM step0: {flow_val:.4f})")
            step += 1
            
    # 4. Save Temporary Exported Model
    temp_export_dir = os.path.join(output_dir, "temp_overfit_policy")
    print(f"Exporting temporary policy to {temp_export_dir}...")
    
    # Set up config for export
    action_stats = dataset.meta.stats.get("action")
    if action_stats is not None:
        action_stats = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in action_stats.items()}
        
    export_config = SmolVLARECAPConfig(
        num_vlm_layers=policy_cfg.num_vlm_layers,
        max_action_dim=policy_cfg.max_action_dim,
        input_features=policy_cfg.input_features,
        action_stats=action_stats,
        chunk_size=policy_cfg.chunk_size,
        n_action_steps=1,
        device="cpu",
        adv_dropout_rate=0.0,
    )
    export_config.output_features = policy_cfg.output_features
    
    policy = SmolVLARECAPPolicy(export_config)
    clean_sd = {"model." + k: v for k, v in model.state_dict().items()}
    policy.load_state_dict(clean_sd, strict=False)
    policy.save_pretrained(temp_export_dir)
    
    # Run migration script
    import subprocess
    migrated_dir = os.path.join(output_dir, "temp_overfit_policy_migrated")
    cmd = [
        ".venv/bin/python",
        "-c",
        "import lerobot_policy_smolvla_rl; from lerobot.processor.migrate_policy_normalization import main; main()",
        "--pretrained-path", temp_export_dir,
        "--output-dir", migrated_dir
    ]
    subprocess.run(cmd, check=True)
    
    # Populate stats in migrated processors
    stats_tensors = {}
    if dataset.meta.stats is not None:
        for feature_name, stat_dict in dataset.meta.stats.items():
            stats_tensors[feature_name] = {}
            for stat_name, val in stat_dict.items():
                stats_tensors[feature_name][stat_name] = torch.tensor(val)
                
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=export_config,
        pretrained_path=migrated_dir,
    )
    flat_stats = {f"{f}.{s}": v for f, sd in stats_tensors.items() for s, v in sd.items()}
    for step in preprocessor.steps + postprocessor.steps:
        if hasattr(step, 'load_state_dict') and hasattr(step, 'norm_map'):
            step.load_state_dict(flat_stats)
            
    preprocessor.save_pretrained(migrated_dir)
    postprocessor.save_pretrained(migrated_dir)
    
    # 5. Evaluate End-to-End Inference Alignment
    print("Running inference alignment checks...")
    eval_policy = make_policy(
        PreTrainedConfig.from_pretrained(migrated_dir),
        ds_meta=dataset.meta
    )
    eval_policy.to(device)
    eval_policy.eval()
    
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=export_config,
        pretrained_path=migrated_dir,
        preprocessor_overrides={"device_processor": {"device": device}}
    )
    
    l1_errors = []
    # Test on a subset of frames in Episode 0 (e.g. step 0, 10, 20, 30, 40, 50, 60, 70)
    test_indices = list(range(from_idx, to_idx, 10))
    for idx in test_indices:
        eval_policy.reset()
        sample = dataset[idx]
        
        # Simulating environment observation (raw coordinates)
        batch = {
            "observation.images.image": sample["observation.images.image"].unsqueeze(0),
            "observation.images.image2": sample["observation.images.image2"].unsqueeze(0),
            "observation.state": sample["observation.state"].unsqueeze(0),
            "task_index": sample["task_index"].unsqueeze(0),
            "task": [sample["task"]]
        }
        
        preprocessed = preprocessor(batch)
        preprocessed = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in preprocessed.items()}
        
        with torch.no_grad():
            batch_copy = {k: v for k, v in preprocessed.items()}
            pred = eval_policy.select_action(batch_copy, cfg_weight=1.0)
            # Unnormalize prediction to match raw coordinate space of gt_action
            pred = postprocessor(pred)
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
                
        gt_action = sample["action"][0].numpy()
        pred_action = pred[0]
        
        l1 = float(np.mean(np.abs(gt_action - pred_action)))
        l1_errors.append(l1)
        results["errors"].append({
            "frame": idx - from_idx,
            "gt_action": [float(x) for x in gt_action],
            "pred_action": [float(x) for x in pred_action],
            "l1_error": l1
        })
        
    mean_l1 = float(np.mean(l1_errors))
    results["mean_l1_error"] = mean_l1
    print(f"Mean L1 Action Reconstruction Error: {mean_l1:.4f}")
    
    # Overfit passes if error is extremely low (meaning training & inference alignment is perfect)
    results["status"] = "PASSED" if mean_l1 < 0.15 else "FAILED"
    print(f"Overfit Status: {results['status']}")
    
    with open(json_output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results written to: {json_output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Overfit verification script")
    parser.add_argument("--dataset", type=str, default="sancov/smolvla_recap_libero_spatial")
    parser.add_argument("--pretrained_model", type=str, default="outputs/policies/recap_libero_pretrained")
    parser.add_argument("--output_dir", type=str, default="outputs/overfit_test")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    args = parser.parse_args()
    
    run_overfit_test(args.dataset, args.pretrained_model, args.output_dir, args.steps, args.lr)
