import torch
import os
from lerobot_policy_smolvla_rl.smolvla_critic import SmolVLACrictic, SmolVLMCriticConfig
from lerobot.configs.types import FeatureType, PolicyFeature

def create_dummy_checkpoint():
    device = "cpu"
    checkpoint_path = "dummy_critic.pt"
    
    print("Creating dummy critic config...")
    config = SmolVLMCriticConfig(
        num_bins=201,
        num_vlm_layers=2, # Small for speed
    )
    
    # Mocking input features
    config.input_features = {
        "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))
    }
    
    print("Initializing model...")
    model = SmolVLACrictic(config).to(device)
    
    print(f"Saving dummy checkpoint to {checkpoint_path}...")
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path

if __name__ == "__main__":
    create_dummy_checkpoint()
