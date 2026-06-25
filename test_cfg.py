import sys

sys.path.insert(0, "src")
import torch
from lerobot_policy_smolvla_rl import SmolVLARECAPConfig, SmolVLARECAPPolicy
from lerobot.configs.types import FeatureType, PolicyFeature

config = SmolVLARECAPConfig(
    num_vlm_layers=1,
    use_advantage_conditioning=True,
    max_action_dim=7,
    chunk_size=2,
    n_action_steps=1,
    input_features={
        "observation.images.front": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 512, 512)
        ),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
    },
    output_features={
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    },
)
policy = SmolVLARECAPPolicy(config)

device = next(policy.model.parameters()).device
dtype = policy.model.vlm_with_expert.vlm.dtype
batch = {
    "observation.language.tokens": torch.randint(0, 100, (1, 10), device=device),
    "observation.language.attention_mask": torch.ones(
        1, 10, dtype=torch.bool, device=device
    ),
    "observation.state": torch.randn(1, 7, device=device, dtype=dtype),
    "observation.images.front": torch.randn(1, 3, 512, 512, device=device, dtype=dtype),
}

print("Testing cfg_weight=1.0 (standard conditional)...")
action_1 = policy.select_action(batch, cfg_weight=1.0)
print(action_1.shape)

print("\nTesting cfg_weight=0.0 (unconditional)...")
policy.reset()
action_0 = policy.select_action(batch, cfg_weight=0.0)
print(action_0.shape)

print("\nTesting cfg_weight=2.0 (CFG)...")
policy.reset()
action_2 = policy.select_action(batch, cfg_weight=2.0)
print(action_2.shape)

print("\nAll passed!")
