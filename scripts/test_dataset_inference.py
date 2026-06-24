import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_policy_smolvla_rl import SmolVLARECAPPolicy

def test_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Dataset
    print("Loading dataset...")
    dataset = LeRobotDataset("HuggingFaceVLA/libero", episodes=[0])
    
    # 2. Load Policy
    print("Loading policy...")
    policy = SmolVLARECAPPolicy.from_pretrained("outputs/recap_libero_exported")
    policy.eval()
    policy.to(device)

    # 3. Get a sample
    print("Fetching sample...")
    print("Testing MSE over 50 random frames...")
    
    # We need to manually tokenize the language instruction for inference
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    
    mse_cfg_1 = 0.0
    mse_cfg_0 = 0.0
    num_frames = 50
    
    import random
    random.seed(42)
    indices = random.sample(range(len(dataset)), num_frames)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            item = dataset[idx]
            batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}

            task_instruction = item.get('task', 'do the task')
            prompt = f"User: {task_instruction}\nAssistant: "
            tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            
            eval_batch = {}
            for k, v in batch.items():
                if k.startswith("observation"):
                    eval_batch[k] = v
                    
            eval_batch["observation.language.tokens"] = tokens["input_ids"].to(device)
            eval_batch["observation.language.attention_mask"] = tokens["attention_mask"].bool().to(device)
            
            gt_action = batch["action"].squeeze(0)
            
            # Predict CFG=1.0
            policy.reset()
            pred_1 = policy.select_action(eval_batch, cfg_weight=1.0).squeeze()
            mse_1 = torch.nn.functional.mse_loss(pred_1, gt_action).item()
            mse_cfg_1 += mse_1
            
            # Predict CFG=0.0
            policy.reset()
            pred_0 = policy.select_action(eval_batch, cfg_weight=0.0).squeeze()
            mse_0 = torch.nn.functional.mse_loss(pred_0, gt_action).item()
            mse_cfg_0 += mse_0
            
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{num_frames} frames...")

    print("\n--- Results ---")
    print(f"Average MSE (CFG=1.0): {mse_cfg_1 / num_frames:.4f}")
    print(f"Average MSE (CFG=0.0): {mse_cfg_0 / num_frames:.4f}")

if __name__ == "__main__":
    test_inference()
