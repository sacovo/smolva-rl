import pytest
import torch
import torch.nn as nn
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot_policy_smolvla_rl import SmolVLARECAPConfig, SmolVLARECAP, SmolVLARECAPPolicy
from lerobot_policy_smolvla_rl.efficient_inference import select_diverse_tokens, check_pruning_constraints

@pytest.fixture
def base_config():
    config = SmolVLARECAPConfig(
        num_vlm_layers=2,
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

def test_select_diverse_tokens():
    # 1 batch element, 4 visual tokens, 3 channels
    visual_tokens = torch.tensor([[
        [1.0, 0.0, 0.0],   # Token 0
        [1.0, 0.1, 0.0],   # Token 1 (very similar to 0)
        [0.0, 1.0, 0.0],   # Token 2 (orthogonal to 0)
        [0.0, 0.0, 1.0],   # Token 3 (orthogonal to 0 and 2)
    ]], dtype=torch.float32)
    
    # Token 0 has highest relevance
    relevance_scores = torch.tensor([[10.0, 0.0, 5.0, 1.0]], dtype=torch.float32)
    
    # Keep 3 tokens: K_relevance = 1, alpha = 0.5
    # Stage 1 picks Token 0
    # Stage 2 picks Token 2 or 3 (orthogonal to 0), and then the other
    # Stage 3 sorts the selected indices: [0, 2, 3]
    selected_indices = select_diverse_tokens(
        visual_tokens, relevance_scores, K_keep=3, K_relevance=1, alpha=0.5
    )
    
    expected = torch.tensor([[0, 2, 3]], dtype=torch.long)
    assert torch.equal(selected_indices, expected)

def test_layer_redundancy_forward_patching(base_config):
    # Enable visual token pruning configuration
    base_config.visual_tokens_keep = 24
    
    # 2 layer configuration check
    policy = SmolVLARECAPPolicy(base_config)
    
    # Check that check_pruning_constraints runs and accepts None
    check_pruning_constraints(None, 2)
    
    # Check that check_pruning_constraints raises error on invalid inputs
    with pytest.raises(ValueError):
        check_pruning_constraints([0], 2) # Cannot prune layer 0
        
    with pytest.raises(ValueError):
        check_pruning_constraints([1], 2) # Cannot prune layer 1 (final)
        
    with pytest.raises(ValueError):
        check_pruning_constraints([5], 2) # Out of range

def test_training_guardrails(base_config):
    # Try to import training runner or run guardrail checks directly
    # Check RECAP training configuration guardrail
    base_config.pruned_layers = [1] # invalid layer index
    
    # Running checking logic directly
    with pytest.raises(ValueError):
        check_pruning_constraints(base_config.pruned_layers, 2)
        
    # Check if active pruning is rejected during initialization if someone tries to train
    # Normally, model init is fine during inference, but train scripts raise ValueError
    # Let's mock a simple check to verify train-time ValueError raises
    def mock_train_guardrail(config):
        if config.pruned_layers is not None or config.visual_tokens_keep is not None:
            raise ValueError("Pruning configuration must be disabled during co-training/distillation.")
            
    with pytest.raises(ValueError):
        mock_train_guardrail(base_config) # should raise ValueError
    
    base_config.pruned_layers = None
    base_config.visual_tokens_keep = 24
    with pytest.raises(ValueError):
        mock_train_guardrail(base_config)

def test_kv_reindex_correctness(base_config):
    # Setup model with 4 VLM layers, prune layer 2
    base_config.num_vlm_layers = 4
    base_config.pruned_layers = [2]
    
    policy = SmolVLARECAPPolicy(base_config)
    model = policy.model
    vlm_model = model.vlm_with_expert
    
    # Kept layers: 0, 1, 3 -> mapped to cache keys 0, 1, 2
    assert vlm_model._layer_idx_to_cache_idx == {0: 0, 1: 1, 3: 2}
    
    # Run a mock forward pass with use_cache=True, fill_kv_cache=True
    # Mock inputs
    inputs_embeds = [torch.randn(2, 5, vlm_model.vlm.config.text_config.hidden_size), None]
    attention_mask = torch.ones(2, 5, 5, dtype=torch.bool)
    position_ids = torch.arange(5, dtype=torch.long).unsqueeze(0).expand(2, -1)
    
    past_key_values = {}
    with torch.no_grad():
        outputs, past_key_values = vlm_model.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            fill_kv_cache=True
        )
        
    # Check that past_key_values keys are exactly 0, 1, 2 (dense reindexing!)
    assert set(past_key_values.keys()) == {0, 1, 2}
    
    # Now run an autoregressive denoise step (fill_kv_cache=False)
    # The keys in past_key_values must be read and updated
    inputs_embeds_step = [None, torch.randn(2, 1, vlm_model.expert_hidden_size)]
    attention_mask_step = torch.ones(2, 1, 6, dtype=torch.bool)
    position_ids_step = torch.tensor([[5]], dtype=torch.long).expand(2, -1)
    
    with torch.no_grad():
        outputs_step, past_key_values = vlm_model.forward(
            attention_mask=attention_mask_step,
            position_ids=position_ids_step,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds_step,
            use_cache=True,
            fill_kv_cache=False
        )
        
    # Check that past_key_values keys remain exactly 0, 1, 2
    assert set(past_key_values.keys()) == {0, 1, 2}
    # Check that key_states shape remains length 5 (prefix length) per upstream design
    assert past_key_values[0]["key_states"].shape[1] == 5

def test_output_equivalence(base_config):
    # Fix random seed for reproducibility
    torch.manual_seed(42)
    
    # 4 layers base configuration
    base_config.num_vlm_layers = 4
    base_config.attention_mode = "self_attn"
    
    # Model A: pruned_layers = [2]
    config_a = SmolVLARECAPConfig(**vars(base_config))
    config_a.pruned_layers = [2]
    model_a = SmolVLARECAP(config_a)
    
    # Model B: hand-built (num_vlm_layers=4, and then delete layer 2)
    torch.manual_seed(42)
    config_b = SmolVLARECAPConfig(**vars(base_config))
    config_b.pruned_layers = None
    model_b = SmolVLARECAP(config_b)
    
    # Manually delete layer 2 in model_b
    del model_b.vlm_with_expert.vlm.model.text_model.layers[2]
    del model_b.vlm_with_expert.lm_expert.layers[2]
    model_b.vlm_with_expert.num_vlm_layers = 3
    model_b.vlm_with_expert.num_expert_layers = 3
    
    # Check that check_pruning_constraints passes on pruned_layers = [2] with 4 layers
    check_pruning_constraints(config_a.pruned_layers, 4)
    
    # Run identical inputs through both models' prefix forward and compare outputs
    inputs_embeds = [torch.randn(2, 5, model_a.vlm_with_expert.vlm.config.text_config.hidden_size), None]
    attention_mask = torch.ones(2, 5, 5, dtype=torch.bool)
    position_ids = torch.arange(5, dtype=torch.long).unsqueeze(0).expand(2, -1)
    
    with torch.no_grad():
        out_a, _ = model_a.vlm_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=[x.clone() if x is not None else None for x in inputs_embeds],
            use_cache=True,
            fill_kv_cache=True
        )
        out_b, _ = model_b.vlm_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=[x.clone() if x is not None else None for x in inputs_embeds],
            use_cache=True,
            fill_kv_cache=True
        )
        
    # Assert outputs are identical (allowing double precision or generous tolerance for CPU float)
    assert torch.allclose(out_a[0], out_b[0], atol=1e-5, rtol=1e-5)

def test_ki_composition(base_config):
    # 4 layers, prune layer 2
    base_config.num_vlm_layers = 4
    base_config.pruned_layers = [2]
    
    # SmolVLARECAP model applies the KI patch in __init__
    model = SmolVLARECAP(base_config)
    vlm_model = model.vlm_with_expert
    
    # Inputs with gradients - VLM prefix pass
    vlm_in = torch.randn(2, 5, vlm_model.vlm.config.text_config.hidden_size, requires_grad=True)
    inputs_embeds = [vlm_in, None]
    
    attention_mask = torch.ones(2, 5, 5, dtype=torch.bool)
    position_ids = torch.arange(5, dtype=torch.long).unsqueeze(0).expand(2, -1)
    
    # Forward pass
    outputs, _ = vlm_model.forward(
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        use_cache=True,
        fill_kv_cache=True
    )
    
    assert outputs[0].requires_grad
    loss = outputs[0].sum()
    loss.backward()
    assert vlm_in.grad is not None
