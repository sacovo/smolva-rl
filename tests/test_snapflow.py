import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot_policy_smolvla_rl import SmolVLARECAPConfig, SmolVLARECAP

@pytest.fixture
def base_config():
    config = SmolVLARECAPConfig(
        num_vlm_layers=1,
        chunk_size=4,
        n_action_steps=1,
        max_action_dim=2,
        use_advantage_conditioning=True,
    )
    config.input_features = {
        "observation.images.image": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 224, 224)
        ),
    }
    return config

def test_target_time_mlp_zero_init(base_config):
    model = SmolVLARECAP(base_config)
    # Check if target_time_mlp weights and biases are zero-initialized
    for p in model.target_time_mlp.parameters():
        assert torch.all(p == 0.0)

def test_snapflow_loss_computation(base_config):
    model = SmolVLARECAP(base_config)
    
    # Mock batch
    batch = {
        "observation.images.image": torch.randn(2, 1, 3, 224, 224),
        "observation.language.tokens": torch.zeros(2, 10, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(2, 10, dtype=torch.bool),
        "observation.state": torch.randn(2, 2),
        "action": torch.randn(2, 4, 2),
    }
    
    # Forward pass in snapflow mode
    loss, fm_loss, consistency_loss = model(
        batch,
        mode="snapflow",
        alpha=0.5,
        lambda_c=0.1,
        clamp=20.0,
    )
    
    assert loss is not None
    assert not torch.isnan(loss)
    assert not torch.isnan(fm_loss)
    assert not torch.isnan(consistency_loss)

def test_snapflow_guided_distillation_loss(base_config):
    model = SmolVLARECAP(base_config)

    batch = {
        "observation.images.image": torch.randn(2, 1, 3, 224, 224),
        "observation.language.tokens": torch.zeros(2, 10, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(2, 10, dtype=torch.bool),
        "observation.state": torch.randn(2, 2),
        "action": torch.randn(2, 4, 2),
    }

    # Guidance distillation: teacher blends cond/uncond at w; one frame
    # advantage-negative so the FM mask path is exercised.
    loss, fm_loss, consistency_loss = model(
        batch,
        mode="snapflow",
        alpha=0.5,
        lambda_c=0.1,
        clamp=20.0,
        advantage=[True, False],
        distill_cfg_weight=1.5,
    )

    assert not torch.isnan(loss)
    assert not torch.isnan(fm_loss)
    assert not torch.isnan(consistency_loss)
    assert loss.requires_grad

def test_snapflow_frozen_teacher_guided_loss(base_config):
    model = SmolVLARECAP(base_config)
    teacher = SmolVLARECAP(base_config)
    teacher.load_state_dict(model.state_dict())
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    batch = {
        "observation.images.image": torch.randn(2, 1, 3, 224, 224),
        "observation.language.tokens": torch.zeros(2, 10, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(2, 10, dtype=torch.bool),
        "observation.state": torch.randn(2, 2),
        "action": torch.randn(2, 4, 2),
    }

    loss, fm_loss, consistency_loss = model(
        batch,
        mode="snapflow",
        alpha=0.5,
        lambda_c=0.1,
        clamp=20.0,
        advantage=[True, False],
        distill_cfg_weight=1.5,
        teacher_model=teacher,
    )

    assert not torch.isnan(loss)
    assert not torch.isnan(fm_loss)
    assert not torch.isnan(consistency_loss)
    assert loss.requires_grad

    # A backward pass must leave the frozen teacher without gradients.
    teacher_before = {k: v.clone() for k, v in teacher.state_dict().items()}
    loss.backward()
    assert all(p.grad is None for p in teacher.parameters())
    for k, v in teacher.state_dict().items():
        assert torch.equal(v, teacher_before[k])


def test_snapflow_single_step_inference(base_config):
    base_config.snapflow_enabled = True
    model = SmolVLARECAP(base_config)
    
    images = [torch.randn(2, 3, 384, 384)]
    img_masks = [torch.ones(2, dtype=torch.bool)]
    lang_tokens = torch.zeros(2, 10, dtype=torch.long)
    lang_masks = torch.ones(2, 10, dtype=torch.bool)
    state = torch.randn(2, 32)
    
    actions = model.sample_actions(
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
    )
    
    # actions shape: (bsize, chunk_size, max_action_dim)
    assert actions.shape == (2, base_config.chunk_size, base_config.max_action_dim)

def test_checkpoint_save_load(base_config, tmp_path):
    model = SmolVLARECAP(base_config)
    state_dict = model.state_dict()
    assert "target_time_mlp.0.weight" in state_dict
    
    checkpoint_path = tmp_path / "model.pt"
    torch.save(state_dict, checkpoint_path)
    
    new_model = SmolVLARECAP(base_config)
    loaded_state_dict = torch.load(checkpoint_path, map_location="cpu")
    new_model.load_state_dict(loaded_state_dict)
    
    for p1, p2 in zip(model.target_time_mlp.parameters(), new_model.target_time_mlp.parameters()):
        assert torch.equal(p1, p2)
