import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForImageTextToText
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from .smolvla_fast import SmolVLAFast
import numpy as np

class SmolVLARECAP(nn.Module):
    """
    SmolVLA model implementing the RECAP / Knowledge Insulation recipe.
    - VLM Backbone trained with AR loss on discretized FAST tokens.
    - Action Expert trained with Flow Matching loss on continuous actions.
    - Knowledge Insulation (KI): No gradients from Action Expert to VLM.
    - Advantage Conditioning: Optional "Advantage: positive/negative" tokens.
    """

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        num_fast_tokens: int = 1024,
        action_dim: int = 14,
        expert_width_multiplier: float = 0.5,
        num_vlm_layers: int = -1,
        num_expert_layers: int = -1,
        use_advantage_conditioning: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype = torch.bfloat16
    ):
        super().__init__()
        self.device = device
        self.use_advantage_conditioning = use_advantage_conditioning
        self.num_fast_tokens = num_fast_tokens
        self.action_dim = action_dim
        
        # 1. Initialize VLM with Expert
        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=model_id,
            num_vlm_layers=num_vlm_layers,
            num_expert_layers=num_expert_layers,
            expert_width_multiplier=expert_width_multiplier,
            device=device,
            load_vlm_weights=True
        ).to(device)
        
        # 2. Setup FAST tokens (reusing SmolVLAFast logic)
        # We wrap the VLM part in SmolVLAFast to reuse its token mapping and initialization
        self.fast_wrapper = SmolVLAFast(
            model_id=model_id,
            num_fast_tokens=num_fast_tokens,
            device=device,
            torch_dtype=torch_dtype
        )
        # Point fast_wrapper to the same VLM instance
        self.fast_wrapper.vlm = self.vlm_with_expert.vlm
        self.fast_wrapper._initialize_action_embeddings()
        
        # 3. Add Advantage Tokens if requested
        if self.use_advantage_conditioning:
            self.tokenizer = self.vlm_with_expert.processor.tokenizer
            self.tokenizer.add_special_tokens({
                "additional_special_tokens": ["<advantage_positive>", "<advantage_negative>"]
            })
            self.vlm_with_expert.vlm.resize_token_embeddings(len(self.tokenizer))
            self.adv_pos_id = self.tokenizer.convert_tokens_to_ids("<advantage_positive>")
            self.adv_neg_id = self.tokenizer.convert_tokens_to_ids("<advantage_negative>")

        # 4. Action Expert Projections (for Flow Matching)
        self.action_in_proj = nn.Linear(action_dim, self.vlm_with_expert.expert_hidden_size).to(device, torch_dtype)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, action_dim).to(device, torch_dtype)
        
        # 5. Apply Knowledge Insulation (KI) Patch
        self._apply_ki_patch()

    def _apply_ki_patch(self):
        """Patch the forward_attn_layer to implement Knowledge Insulation."""
        original_forward_attn = self.vlm_with_expert.forward_attn_layer
        
        def ki_forward_attn_layer(
            model_layers,
            inputs_embeds,
            layer_idx,
            position_ids,
            attention_mask,
            batch_size,
            head_dim,
            **kwargs
        ):
            # If we have both VLM and Expert, we need to ensure Expert doesn't flow grad to VLM
            if len(inputs_embeds) == 2 and inputs_embeds[0] is not None and inputs_embeds[1] is not None:
                # Simplified KI: Detach VLM embeddings for the expert's input
                vlm_hidden = inputs_embeds[0]
                expert_hidden = inputs_embeds[1]
                
                # Run VLM update normally
                vlm_out, _ = original_forward_attn(
                    model_layers, [vlm_hidden, None], layer_idx, position_ids, attention_mask, batch_size, head_dim, **kwargs
                )
                
                # Run Expert update with detached VLM hidden states (Knowledge Insulation)
                expert_out, _ = original_forward_attn(
                    model_layers, [vlm_hidden.detach(), expert_hidden], layer_idx, position_ids, attention_mask, batch_size, head_dim, **kwargs
                )
                
                return [vlm_out[0], expert_out[1]], None
            
            return original_forward_attn(model_layers, inputs_embeds, layer_idx, position_ids, attention_mask, batch_size, head_dim, **kwargs)

        self.ki_forward_attn = ki_forward_attn_layer

    def compute_loss(self, batch, camera_keys, advantage=None):
        """
        Compute combined loss: AR loss on FAST tokens + Flow Matching loss on continuous actions.
        """
        device = self.device
        
        # 1. Prepare inputs with Advantage Conditioning
        prompts = []
        for i, t in enumerate(batch["task"]):
            adv_str = ""
            if advantage is not None:
                adv_str = "<advantage_positive> " if advantage[i] else "<advantage_negative> "
            # Must include <image> token for each camera
            img_tokens = "<image>" * len(camera_keys)
            prompts.append(f"{img_tokens}Task: {t} {adv_str}Action:")
            
        processor = self.vlm_with_expert.processor
        inputs = processor(
            images=[[batch[k][i].cpu() for k in camera_keys] for i in range(len(batch["task"]))],
            text=prompts,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # 2. Encode discrete actions (FAST)
        actions_np = batch["action"].cpu().numpy()
        action_token_ids, action_mask = self.fast_wrapper.encode_actions(actions_np, return_mask=True) # [B, N]
        
        # 3. Concatenate action tokens to input_ids for AR training
        full_input_ids = torch.cat([inputs["input_ids"], action_token_ids], dim=1)
        labels = torch.full_like(full_input_ids, -100)
        action_labels = torch.where(action_mask, action_token_ids, torch.tensor(-100, device=device))
        labels[:, inputs["input_ids"].shape[1]:] = action_labels
        
        # Update attention mask
        full_attention_mask = torch.cat([inputs["attention_mask"], action_mask.long()], dim=1)
        
        # 4. VLM Forward Pass (for AR loss)
        vlm_outputs = self.vlm_with_expert.vlm(
            pixel_values=inputs["pixel_values"],
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        ar_loss = vlm_outputs.loss
        
        # 5. Knowledge Insulation: Get prefix hidden states and detach them for the Expert
        # We use negative indexing because action tokens are not expanded and are at the end.
        action_seq_len = action_token_ids.shape[1]
        vlm_prefix_hidden = vlm_outputs.hidden_states[-1][:, :-action_seq_len, :]
        
        # 6. Action Expert (Flow Matching)
        bsize = full_input_ids.shape[0]
        tau = torch.rand((bsize, 1, 1), device=device)
        omega = torch.randn_like(batch["action"]).to(device)
        noised_actions = tau * batch["action"].to(device) + (1 - tau) * omega
        
        expert_input = self.action_in_proj(noised_actions.to(self.vlm_with_expert.vlm.dtype))
        
        # Expert attends to detached VLM hidden states (KI)
        expert_outputs = self.vlm_with_expert.lm_expert(
            inputs_embeds=expert_input,
            encoder_hidden_states=vlm_prefix_hidden.detach(), # KI
            return_dict=True
        )
        expert_hidden = expert_outputs.last_hidden_state
        
        predicted_flow = self.action_out_proj(expert_hidden)
        target_flow = omega - batch["action"].to(device)
        flow_loss = F.mse_loss(predicted_flow, target_flow)
        
        # Combined loss
        total_loss = ar_loss + flow_loss
        
        return total_loss, ar_loss, flow_loss
