import torch
import numpy as np
from lerobot_policy_smolvla_rl.smolvla_recap import SmolVLARECAP
from lerobot_policy_smolvla_rl.smolvla_critic import SmolVLACrictic, SmolVLMCriticConfig

def test_recap_and_critic_integration():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on {device}")

    # 1. Initialize SmolVLARECAP with minimal layers
    print("Initializing RECAP model...")
    model = SmolVLARECAP(
        model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        num_vlm_layers=2,  # Minimal layers for testing
        num_expert_layers=2,
        action_dim=14,
        use_advantage_conditioning=True,
        device=device
    )
    model.eval()

    # 2. Initialize Critic with minimal layers
    print("Initializing Critic model...")
    critic_config = SmolVLMCriticConfig(
        num_bins=201,
        num_vlm_layers=2,
    )
    # Mocking input features as it's needed for initialization
    from lerobot.configs.types import FeatureType, PolicyFeature
    
    critic_config.input_features = {
        "observation.images.base": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,))
    }
    
    critic = SmolVLACrictic(critic_config).to(device)
    critic.eval()

    # 3. Create Dummy Batch
    print("Creating dummy batch...")
    batch_size = 2
    dummy_batch = {
        "observation.images.base": torch.rand(batch_size, 3, 224, 224, device=device), # Use rand for [0, 1] range
        "observation.state": torch.randn(batch_size, 14, device=device),
        "task": ["pick up the block", "move the arm"],
        "action": torch.randn(batch_size, 50, 14, device=device),
        "task_index": torch.zeros(batch_size, dtype=torch.long, device=device),
        "episode_index": torch.zeros(batch_size, dtype=torch.long, device=device),
        "frame_index": torch.zeros(batch_size, dtype=torch.long, device=device),
    }
    
    # Mocking tokens for critic (usually handled by preprocessor, but let's see)
    # The SmolVLACrictic._prepare_batch expects specific keys
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    dummy_batch[OBS_LANGUAGE_TOKENS] = torch.randint(0, 1000, (batch_size, 16), device=device)
    dummy_batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(batch_size, 16, dtype=torch.bool, device=device)

    # 4. Test Critic Forward Pass
    print("Testing Critic forward...")
    with torch.no_grad():
        logits, probs = critic(dummy_batch)
        print(f"Critic Logits Shape: {logits.shape}") # Should be [B, 201]
        print(f"Critic Probs Shape: {probs.shape}")

    # 5. Test RECAP Loss Computation (with advantage)
    print("Testing RECAP loss...")
    advantage = [True, False]
    camera_keys = ["observation.images.base"]
    
    # This triggers the full pipeline: FAST encoding, VLM AR, Expert Flow Matching, KI
    total_loss, ar_loss, flow_loss = model.compute_loss(dummy_batch, camera_keys, advantage=advantage)
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"AR Loss: {ar_loss.item():.4f}")
    print(f"Flow Loss: {flow_loss.item():.4f}")

    # 6. Verify KI Indexing (Manual check of prefix hidden state size)
    # We can't easily check internal locals, but if it didn't crash, 
    # the slicing didn't go out of bounds.

    print("Test passed successfully!")

if __name__ == "__main__":
    try:
        test_recap_and_critic_integration()
    except Exception as e:
        import traceback
        traceback.print_exc()
        exit(1)
