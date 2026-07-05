import torch
import json
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_policy_smolvla_rl.modeling_smolvla_recap import SmolVLARECAPPolicy

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load the exported policy
    print("Loading policy...")
    policy = SmolVLARECAPPolicy.from_pretrained("outputs/recap_libero_exported")
    policy.to(device)
    policy.eval()

    # 1. Load dataset using the policy config
    print("Loading dataset...")
    dataset = LeRobotDataset(
        repo_id="HuggingFaceVLA/libero",
        delta_timestamps={
            "action": [i / 10.0 for i in range(policy.config.chunk_size)],
        }
    )

    # 3. Load precomputed advantages and thresholds
    print("Loading precomputed advantages...")
    precomputed_adv = np.load("outputs/recap_phase1/task_advantages_HuggingFaceVLA_libero.npy")
    with open("outputs/recap_phase1/task_thresholds_HuggingFaceVLA_libero.json", "r") as f:
        threshold_data = json.load(f)
    
    # 4. Get the preprocessor
    pre_recap = policy.model.get_pre_processor(dataset)

    print("Testing forward pass (Loss computation) over 10 random frames...")
    
    num_frames = 10
    import random
    random.seed(42)
    indices = random.sample(range(len(dataset)), num_frames)

    total_ar_loss = 0.0
    total_fm_loss = 0.0

    with torch.no_grad():
        for i, idx in enumerate(indices):
            item = dataset[idx]
            
            # Form batch
            batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v for k, v in item.items()}

            # Get advantage for this frame
            task_idx = int(batch["task_index"].item())
            adv_val = precomputed_adv[idx]
            thresh_val = float(threshold_data[str(task_idx)])
            advantage_bool = [adv_val > thresh_val]

            # Preprocess to add language tokens with advantage code
            recap_batch = pre_recap(batch)
            recap_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in recap_batch.items()}

            # Forward pass through the model
            total_loss, ar_loss, flow_loss = policy.model(recap_batch, advantage=advantage_bool)

            print(f"Frame {idx}: AR Loss = {ar_loss.item():.4f}, FM Loss = {flow_loss.item():.4f}")
            
            total_ar_loss += ar_loss.item()
            total_fm_loss += flow_loss.item()

    print("\n--- Average Results ---")
    print(f"Average AR Loss: {total_ar_loss / num_frames:.4f}")
    print(f"Average FM Loss: {total_fm_loss / num_frames:.4f}")

if __name__ == "__main__":
    main()
