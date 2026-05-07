import torch
import numpy as np

# Let's use a local import if possible or just rely on the package structure
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))

from lerobot_policy_smolvla_rl.smolvla_fast import SmolVLAFast

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize model
    # Using small number of tokens for quick test if needed, but 1024 is default.
    model = SmolVLAFast(device=device, torch_dtype=torch.float32) # Use float32 for CPU test if needed
    print("Model initialized.")
    
    # Test action roundtrip
    batch_size = 1
    chunk_size = 4
    action_dim = 14
    dummy_actions = np.random.rand(batch_size, chunk_size, action_dim).astype(np.float32)
    
    print(f"Original actions (first step): {dummy_actions[0, 0, :5]}")
    
    # Encode
    vlm_tokens = model.encode_actions(dummy_actions)
    print(f"Encoded VLM tokens: {vlm_tokens}")
    
    # Decode
    decoded_actions = model.decode_actions(vlm_tokens)
    print(f"Decoded actions (first step): {decoded_actions[0, 0, :5]}")
    
    # Check if they are similar (quantization loss expected)
    diff = np.abs(dummy_actions - decoded_actions)
    print(f"Mean absolute difference: {np.mean(diff)}")
    
    print("Verification complete.")

if __name__ == "__main__":
    main()
