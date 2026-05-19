import math
from collections import deque
from typing import TypedDict, Unpack

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import numpy as np

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.device_utils import get_safe_dtype
from lerobot.policies.smolvla import modeling_smolvla

from .smolvla_fast import SmolVLAFast, SmolVLAFastConfig
from .configuration_smolvla_recap import SmolVLARECAPConfig

class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None

class SmolVLARECAP(modeling_smolvla.VLAFlowMatching):
    """
    SmolVLA model implementing the RECAP / Knowledge Insulation recipe.
    - VLM Backbone trained with AR loss on discretized FAST tokens.
    - Action Expert trained with Flow Matching loss on continuous actions.
    - Knowledge Insulation (KI): No gradients from Action Expert to VLM.
    - Advantage Conditioning: Optional "<advantage_positive>/<advantage_negative>" tokens.
    """

    def __init__(self, config: SmolVLARECAPConfig):
        # Set vlm_model_name from model_id if not present
        if not hasattr(config, "vlm_model_name") or config.vlm_model_name is None:
            config.vlm_model_name = config.model_id
        
        config.compile_model = False
        super().__init__(config)
        self.config = config
        
        # Setup FAST tokens (reusing SmolVLAFast logic)
        fast_config = SmolVLAFastConfig(
            num_fast_tokens=config.num_fast_tokens,
            model_id=config.model_id,
            action_stats=config.action_stats,
            **{k: v for k, v in vars(config).items() if k not in ["num_fast_tokens", "model_id", "action_stats", "use_advantage_conditioning"]}
        )
        self.fast_wrapper = SmolVLAFast(fast_config)
        # Point to the same VLM with Expert instance to avoid redundant weights
        self.fast_wrapper.vlm_with_expert = self.vlm_with_expert
        
        # Add Advantage Tokens if requested
        if config.use_advantage_conditioning:
            self.tokenizer = self.vlm_with_expert.processor.tokenizer
            self.tokenizer.add_special_tokens({
                "additional_special_tokens": ["<advantage_positive>", "<advantage_negative>"]
            })
            self.vlm_with_expert.vlm.resize_token_embeddings(len(self.tokenizer))
            self.adv_pos_id = self.tokenizer.convert_tokens_to_ids("<advantage_positive>")
            self.adv_neg_id = self.tokenizer.convert_tokens_to_ids("<advantage_negative>")

        # Apply Knowledge Insulation (KI) Patch
        self._apply_ki_patch()

        # Ensure projection layers are in the same dtype as the VLM
        self.action_in_proj.to(self.vlm_with_expert.vlm.dtype)
        self.action_out_proj.to(self.vlm_with_expert.vlm.dtype)

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
            # If we have both VLM and Expert, ensure Expert doesn't flow grad to VLM
            if len(inputs_embeds) == 2 and inputs_embeds[0] is not None and inputs_embeds[1] is not None:
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

        self.vlm_with_expert.forward_attn_layer = ki_forward_attn_layer

    def compute_loss(self, batch, camera_keys=None, advantage=None):
        """
        Compute combined loss: AR loss on FAST tokens + Flow Matching loss on continuous actions.
        """
        # 1. Prepare batch components (includes observation state)
        images, img_masks, state, lang_tokens, lang_masks = self.fast_wrapper._prepare_batch(batch)
        
        # 2. Add Advantage conditioning to lang_tokens if active
        if self.config.use_advantage_conditioning and advantage is not None:
            device = lang_tokens.device
            adv_tokens = torch.tensor([self.adv_pos_id if a else self.adv_neg_id for a in advantage], 
                                      device=device).unsqueeze(1)
            lang_tokens = torch.cat([lang_tokens, adv_tokens], dim=1)
            lang_masks = torch.cat([lang_masks, torch.ones_like(adv_tokens, dtype=torch.bool)], dim=1)

        # 3. Embed prefix (images + language + state)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        
        # 4. VLM Backbone (AR loss on FAST tokens)
        actions_np = batch[ACTION].cpu().numpy()
        action_token_ids, action_mask = self.fast_wrapper.encode_actions(actions_np, return_mask=True)
        action_embs = self.vlm_with_expert.embed_language_tokens(action_token_ids)
        
        full_embs = torch.cat([prefix_embs, action_embs], dim=1)
        full_pad_masks = torch.cat([prefix_pad_masks, action_mask], dim=1)
        action_att_masks = torch.ones((prefix_att_masks.shape[0], action_token_ids.shape[1]), 
                                     dtype=prefix_att_masks.dtype, device=prefix_att_masks.device)
        full_att_masks = torch.cat([prefix_att_masks, action_att_masks], dim=1)
        
        full_att_2d_masks = modeling_smolvla.make_att_2d_masks(full_pad_masks, full_att_masks)
        full_position_ids = torch.cumsum(full_pad_masks, dim=1) - 1
        
        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=full_position_ids,
            past_key_values=None,
            inputs_embeds=[full_embs, None],
            use_cache=False,
            fill_kv_cache=True,
        )
        
        vlm_hidden = outputs_embeds[0]
        head = self.vlm_with_expert.vlm.get_output_embeddings()
        logits = head(vlm_hidden.to(head.weight.dtype))
        
        labels = torch.full_like(full_pad_masks.long(), -100)
        prefix_len = prefix_embs.shape[1]
        action_labels = torch.where(action_mask, action_token_ids, torch.tensor(-100, device=action_token_ids.device))
        labels[:, prefix_len:] = action_labels
        
        shift_logits = logits[:, prefix_len-1:-1, :].contiguous()
        shift_labels = labels[:, prefix_len:].contiguous()
        ar_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # 5. Action Expert (Flow Matching loss)
        # Detach VLM hidden states for KI
        vlm_prefix_hidden = vlm_hidden[:, :prefix_len, :]
        expert_dtype = next(self.vlm_with_expert.lm_expert.parameters()).dtype
        
        bsize = full_embs.shape[0]
        device = full_embs.device
        tau = torch.rand((bsize, 1, 1), device=device, dtype=expert_dtype)
        omega = torch.randn_like(batch[ACTION]).to(device=device, dtype=expert_dtype)
        noised_actions = tau * batch[ACTION].to(device=device, dtype=expert_dtype) + (1 - tau) * omega
        
        expert_input = self.action_in_proj(noised_actions.to(next(self.action_in_proj.parameters()).dtype))
        
        expert_outputs = self.vlm_with_expert.lm_expert(
            inputs_embeds=expert_input.to(expert_dtype),
            encoder_hidden_states=vlm_prefix_hidden.detach().to(expert_dtype), # KI
            return_dict=True
        )
        expert_hidden = expert_outputs.last_hidden_state
        
        predicted_flow = self.action_out_proj(expert_hidden.to(next(self.action_out_proj.parameters()).dtype))
        target_flow = omega - batch[ACTION].to(device=device, dtype=expert_dtype)
        
        # Ensure shapes match for MSE loss (handle potential sequence/chunk dimension)
        if predicted_flow.ndim == 3 and target_flow.ndim == 2:
            target_flow = target_flow.unsqueeze(1)
            
        flow_loss = F.mse_loss(predicted_flow.to(torch.float32), target_flow.to(torch.float32))
        
        total_loss = ar_loss + flow_loss
        return total_loss, ar_loss, flow_loss

    def generate_action(self, batch, chunk_size=None):
        """Generate action tokens including observation state."""
        return self.fast_wrapper.generate_action(batch, chunk_size=chunk_size)

    def get_pre_processor(self, dataset):
        return self.fast_wrapper.get_pre_processor(dataset)


class SmolVLARECAPPolicy(PreTrainedPolicy):
    """Wrapper class around SmolVLARECAP model to train and run inference within LeRobot."""

    config_class = SmolVLARECAPConfig
    name = "smolvla_recap"

    def __init__(
        self,
        config: SmolVLARECAPConfig,
        **kwargs,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = SmolVLARECAP(config)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `predict_action_chunk` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `predict_action_chunk` when the
        queue is empty.
        """
        self.eval()

        # Handle robot observation state and images via pre-processor
        # (Already handled by LeRobot pipeline, but ensure queues are updated)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise, **kwargs)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Predict a chunk of actions."""
        self.eval()
        
        # Currently, SmolVLARECAP's generate_action is used for inference.
        # It handles prefix embedding (images + state + language) internally.
        actions = self.model.generate_action(batch, chunk_size=self.config.chunk_size)
        
        # Convert back to torch tensor if it's numpy
        if not isinstance(actions, torch.Tensor):
            actions = torch.from_numpy(actions).to(device=self.model.vlm_with_expert.vlm.device)
            
        return actions
