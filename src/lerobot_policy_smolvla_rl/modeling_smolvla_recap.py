from collections import deque
from typing import TypedDict, Unpack

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION
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
    - SnapFlow for faster inference
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
            **{
                k: v
                for k, v in vars(config).items()
                if k
                not in [
                    "num_fast_tokens",
                    "model_id",
                    "action_stats",
                    "use_advantage_conditioning",
                    "adv_dropout_rate",
                    "cfg_weight",
                    "ar_loss_weight",
                    "fm_loss_weight",
                    "loss_n_action_steps",
                    "snapflow_enabled",
                    "snapflow_alpha",
                    "snapflow_lambda",
                    "snapflow_clamp",
                ]
            },
        )
        self.fast_wrapper = SmolVLAFast(fast_config)
        # Point to the same VLM with Expert instance to avoid redundant weights
        self.fast_wrapper.vlm_with_expert = self.vlm_with_expert
        # Re-initialize action embeddings on the shared VLM instance!
        self.fast_wrapper._initialize_action_embeddings()

        # Setup target time MLP for SnapFlow
        expert_hidden = self.vlm_with_expert.expert_hidden_size
        self.target_time_mlp = torch.nn.Sequential(
            torch.nn.Linear(expert_hidden, expert_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(expert_hidden, expert_hidden),
        )
        # Zero-initialize so model output is identical to pretrained at step 0
        torch.nn.init.zeros_(self.target_time_mlp[0].weight)
        torch.nn.init.zeros_(self.target_time_mlp[0].bias)
        torch.nn.init.zeros_(self.target_time_mlp[2].weight)
        torch.nn.init.zeros_(self.target_time_mlp[2].bias)

        # Add Advantage Tokens if requested
        if config.use_advantage_conditioning:
            self.tokenizer = self.vlm_with_expert.processor.tokenizer
            self.tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": [
                        "<advantage_positive>",
                        "<advantage_negative>",
                    ]
                }
            )
            self.vlm_with_expert.vlm.resize_token_embeddings(len(self.tokenizer))

            # Initialize advantage token embeddings to match the norm of
            # existing trained embeddings.  resize_token_embeddings produces
            # near-zero rows (std ≈ 0.02) that are effectively invisible to
            # attention – the model would need an impractical number of steps
            # to learn them from scratch.
            with torch.no_grad():
                embed = self.vlm_with_expert.vlm.get_input_embeddings()
                mean_norm = embed.weight[:-2].norm(dim=1).mean()
                for row in (-2, -1):
                    embed.weight[row] = (
                        F.normalize(embed.weight[row], dim=0) * mean_norm
                    )

            self.adv_pos_id = self.tokenizer.convert_tokens_to_ids(
                "<advantage_positive>"
            )
            self.adv_neg_id = self.tokenizer.convert_tokens_to_ids(
                "<advantage_negative>"
            )

        # Apply Knowledge Insulation (KI) Patch
        self._apply_ki_patch()

        # Ensure projection layers and expert are in the same dtype as the VLM
        vlm_dtype = self.vlm_with_expert.vlm.dtype
        self.vlm_with_expert.lm_expert.to(vlm_dtype)
        self.action_in_proj.to(vlm_dtype)
        self.state_proj.to(vlm_dtype)
        self.action_time_mlp_in.to(vlm_dtype)
        self.action_time_mlp_out.to(vlm_dtype)
        self.fast_wrapper.to(vlm_dtype)
        self.target_time_mlp.to(vlm_dtype)

        # Unfreeze VLM language model parameters (including embed_tokens and lm_head)
        # for RECAP co-training while keeping the vision encoder frozen.
        for name, param in self.vlm_with_expert.vlm.named_parameters():
            if "vision_model" not in name:
                param.requires_grad = True

    def embed_suffix(self, noisy_actions, timestep, target_time=None):
        # Cast noisy_actions to projection weights dtype to prevent mat1/mat2 dtype mismatch (Float vs BFloat16)
        noisy_actions = noisy_actions.to(dtype=self.action_in_proj.weight.dtype)

        # Standard: project actions + fuse with timestep sinusoidal embedding
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        dtype = action_emb.dtype

        time_emb = modeling_smolvla.create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        ).to(dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # SnapFlow: add target-time embedding
        if target_time is not None:
            s_emb = modeling_smolvla.create_sinusoidal_pos_embedding(
                target_time,
                self.vlm_with_expert.expert_hidden_size,
                self.config.min_period,
                self.config.max_period,
                device=device,
            ).to(dtype)
            s_emb = s_emb[:, None, :].expand_as(action_emb)
            target_emb = self.target_time_mlp(s_emb)
            action_time_emb = action_time_emb + target_emb

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=device
        )
        att_masks = [1] * self.config.chunk_size
        att_masks = torch.tensor(att_masks, dtype=action_time_emb.dtype, device=device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return action_time_emb, action_time_mask, att_masks

    def _apply_ki_patch(self):
        """Patch the forward_attn_layer to implement Knowledge Insulation."""
        original_forward_attn = self.vlm_with_expert.forward_attn_layer

        # pylint: disable=too-many-arguments,too-many-positional-arguments
        def ki_forward_attn_layer(
            model_layers,
            inputs_embeds,
            layer_idx,
            position_ids,
            attention_mask,
            batch_size,
            head_dim,
            **kwargs,
        ):
            # If we have both VLM and Expert, ensure Expert doesn't flow grad to VLM
            if (
                len(inputs_embeds) == 2
                and inputs_embeds[0] is not None
                and inputs_embeds[1] is not None
            ):
                vlm_hidden = inputs_embeds[0]
                expert_hidden = inputs_embeds[1]

                # Run VLM update with no_grad — the FM loss must not update
                # VLM layer weights.  The VLM output serves only as frozen
                # context (KV) for the expert's cross-attention.
                with torch.no_grad():
                    vlm_out, _ = original_forward_attn(
                        model_layers,
                        [vlm_hidden, None],
                        layer_idx,
                        position_ids,
                        attention_mask,
                        batch_size,
                        head_dim,
                        **kwargs,
                    )
                # Detach outputs to sever any residual graph edges
                vlm_out = [v.detach() if v is not None else None for v in vlm_out]

                # Run Expert update with detached VLM hidden states (Knowledge Insulation)
                expert_out, _ = original_forward_attn(
                    model_layers,
                    [vlm_hidden.detach(), expert_hidden],
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    **kwargs,
                )

                if len(vlm_out) == 1:
                    # Self-attention mode: both inputs concatenated in a single attention output
                    vlm_len = vlm_hidden.shape[1]
                    unified_att_output = torch.cat(
                        [
                            vlm_out[0][:, :vlm_len],
                            expert_out[0][:, vlm_len:],
                        ],
                        dim=1,
                    )
                    return [unified_att_output], None

                # Cross-attention mode: separate attention outputs
                return [vlm_out[0], expert_out[1]], None

            return original_forward_attn(
                model_layers,
                inputs_embeds,
                layer_idx,
                position_ids,
                attention_mask,
                batch_size,
                head_dim,
                **kwargs,
            )

        self.vlm_with_expert.forward_attn_layer = ki_forward_attn_layer

    # pylint: disable=arguments-differ
    def forward(
        self,
        batch,
        camera_keys=None,
        advantage=None,
        mode="standard",
        alpha=0.5,
        lambda_c=0.1,
        clamp=20.0,
    ):
        if mode == "snapflow":
            return self.compute_loss_snapflow(
                batch, alpha=alpha, lambda_c=lambda_c, clamp=clamp
            )
        return self.compute_loss(batch, camera_keys=camera_keys, advantage=advantage)

    def compute_loss_snapflow(self, batch, alpha=0.5, lambda_c=0.1, clamp=20.0):
        """
        Compute combined loss for SnapFlow distillation:
        FM loss component (s=t) + Consistency loss component (Shortcut t=1->0.5->s=0 vs student s=0).
        """
        # 1. Prepare batch components (includes observation state)
        # pylint: disable=protected-access
        images, img_masks, state, lang_tokens, lang_masks = (
            self.fast_wrapper._prepare_batch(batch)
        )

        # 2. Add Advantage conditioning to lang_tokens if active (always use positive advantage tokens during SnapFlow)
        if self.config.use_advantage_conditioning:
            device = lang_tokens.device
            adv_tokens = torch.tensor(
                [self.adv_pos_id] * lang_tokens.shape[0],
                device=device,
            ).unsqueeze(1)
            lang_tokens = torch.cat([lang_tokens, adv_tokens], dim=1)
            lang_masks = torch.cat(
                [lang_masks, torch.ones_like(adv_tokens, dtype=torch.bool)], dim=1
            )

        # 3. Embed prefix (images + language + state)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        # 4. Cache VLM prefix KV cache (once per batch)
        prefix_att_2d_masks = modeling_smolvla.make_att_2d_masks(
            prefix_pad_masks, prefix_att_masks
        )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )

        # Helper for expert forward
        def expert_forward_fn(x_t, t, s):
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
                x_t, t, target_time=s
            )

            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch_size, suffix_len, prefix_len
            )

            suffix_att_2d_masks = modeling_smolvla.make_att_2d_masks(
                suffix_pad_masks, suffix_att_masks
            )

            full_att_2d_masks = torch.cat(
                [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
            )
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            outputs_embeds, _ = self.vlm_with_expert.forward(
                attention_mask=full_att_2d_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=True,
                fill_kv_cache=False,
            )
            suffix_out = outputs_embeds[1]
            suffix_out = suffix_out[:, -self.config.chunk_size :]
            v_t = self.action_out_proj(
                suffix_out.to(next(self.action_out_proj.parameters()).dtype)
            )
            return v_t.to(torch.float32)

        # 5. Extract actions
        actions = batch[ACTION]
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)

        expert_dtype = self.vlm_with_expert.vlm.dtype
        bsize = prefix_embs.shape[0]
        device = prefix_embs.device
        actions_fm = actions.to(device=device, dtype=expert_dtype)

        # === Flow Matching Component ===
        tau = torch.rand((bsize, 1), device=device, dtype=expert_dtype)
        omega = torch.randn_like(actions).to(device=device, dtype=expert_dtype)

        tau_expanded = tau[:, :, None]
        noised_actions = tau_expanded * omega + (1 - tau_expanded) * actions_fm
        target_flow = omega - actions_fm

        predicted_flow = expert_forward_fn(
            noised_actions, tau.squeeze(1), tau.squeeze(1)
        )

        pred_fm = predicted_flow
        target_fm = target_flow
        if self.config.loss_n_action_steps > 0:
            n = self.config.loss_n_action_steps
            pred_fm = pred_fm[:, :n]
            target_fm = target_fm[:, :n]
        fm_loss = F.mse_loss(pred_fm, target_fm)

        # === Consistency Component ===
        x_1 = torch.randn_like(actions).to(device=device, dtype=expert_dtype)

        with torch.no_grad():
            t1 = torch.ones(bsize, device=device)
            s1 = t1.clone()
            v_1 = expert_forward_fn(x_1, t1, s1)
            v_1 = v_1.clamp(-clamp, clamp)

            x_half = x_1 + (-0.5) * v_1

            t_half = torch.full_like(t1, 0.5)
            s_half = t_half.clone()
            v_half = expert_forward_fn(x_half, t_half, s_half)
            v_half = v_half.clamp(-clamp, clamp)

            v_shortcut = 0.5 * (v_1 + v_half)

        s_zero = torch.zeros(bsize, device=device)
        v_student = expert_forward_fn(x_1, t1, s_zero)

        pred_student = v_student
        target_shortcut = v_shortcut
        if self.config.loss_n_action_steps > 0:
            n = self.config.loss_n_action_steps
            pred_student = pred_student[:, :n]
            target_shortcut = target_shortcut[:, :n]
        consistency_loss = F.mse_loss(pred_student, target_shortcut)

        loss = alpha * fm_loss + (1.0 - alpha) * lambda_c * consistency_loss
        return loss, fm_loss, consistency_loss

    # pylint: disable=too-many-locals,unused-argument
    def compute_loss(self, batch, camera_keys=None, advantage=None):
        """
        Compute combined loss: AR loss on FAST tokens + Flow Matching loss on continuous actions.
        """
        # 1. Prepare batch components (includes observation state)
        # pylint: disable=protected-access
        images, img_masks, state, lang_tokens, lang_masks = (
            self.fast_wrapper._prepare_batch(batch)
        )

        # 2. Add Advantage conditioning to lang_tokens if active
        if self.config.use_advantage_conditioning and advantage is not None:
            # Dropout: randomly drop the advantage indicator during training
            # so the model also learns to act unconditionally (required for CFG).
            if self.training and torch.rand(1).item() < self.config.adv_dropout_rate:
                advantage = None

        if self.config.use_advantage_conditioning and advantage is not None:
            device = lang_tokens.device
            adv_tokens = torch.tensor(
                [self.adv_pos_id if a else self.adv_neg_id for a in advantage],
                device=device,
            ).unsqueeze(1)
            lang_tokens = torch.cat([lang_tokens, adv_tokens], dim=1)
            lang_masks = torch.cat(
                [lang_masks, torch.ones_like(adv_tokens, dtype=torch.bool)], dim=1
            )

        # 3. Embed prefix (images + language + state)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        # 4. VLM Backbone (AR loss on FAST tokens)
        actions_np = batch[ACTION].cpu().numpy()
        action_token_ids, action_mask = self.fast_wrapper.encode_actions(
            actions_np, return_mask=True
        )
        action_embs = self.vlm_with_expert.embed_language_tokens(action_token_ids)

        full_embs = torch.cat([prefix_embs, action_embs], dim=1)
        full_pad_masks = torch.cat([prefix_pad_masks, action_mask], dim=1)
        action_att_masks = torch.ones(
            (prefix_att_masks.shape[0], action_token_ids.shape[1]),
            dtype=prefix_att_masks.dtype,
            device=prefix_att_masks.device,
        )
        full_att_masks = torch.cat([prefix_att_masks, action_att_masks], dim=1)

        full_att_2d_masks = modeling_smolvla.make_att_2d_masks(
            full_pad_masks, full_att_masks
        )
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
        action_labels = torch.where(
            action_mask,
            action_token_ids,
            torch.tensor(-100, device=action_token_ids.device),
        )
        labels[:, prefix_len:] = action_labels

        shift_logits = logits[:, prefix_len - 1 : -1, :].contiguous()
        shift_labels = labels[:, prefix_len:].contiguous()
        ar_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        # 5. Action Expert (Flow Matching loss)
        # Complete Knowledge Insulation: freeze ALL VLM parameters during the
        # FM forward pass.  This prevents FM gradients from leaking into the
        # VLM backbone through ANY layer — including cross-attention layers
        # and residual/MLP processing — not just the self-attention layers
        # covered by the KI patch on forward_attn_layer.
        prefix_embs_fm = prefix_embs.detach()
        prefix_pad_masks_fm = prefix_pad_masks.detach()

        # Temporarily freeze VLM parameters so PyTorch does not track
        # gradient contributions from VLM weights during FM forward.
        # The AR loss backward (computed earlier in the graph) is unaffected
        # because those operations were recorded when params had grad=True.
        vlm_param_grad_states = {}
        for name, param in self.vlm_with_expert.vlm.named_parameters():
            vlm_param_grad_states[name] = param.requires_grad
            param.requires_grad_(False)

        expert_dtype = self.vlm_with_expert.vlm.dtype

        bsize = prefix_embs_fm.shape[0]
        device = prefix_embs_fm.device

        actions = batch[ACTION]
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)

        tau = torch.rand((bsize, 1), device=device, dtype=expert_dtype)
        omega = torch.randn_like(actions).to(device=device, dtype=expert_dtype)

        tau_expanded = tau[:, :, None]
        actions_fm = actions.to(device=device, dtype=expert_dtype)
        # Flow matching interpolation — MUST match the base VLAFlowMatching
        # convention used by sample_actions() at inference:
        #   x_t = t * noise + (1 - t) * actions   (t=0 → clean, t=1 → noise)
        #   u_t = noise - actions                   (target velocity)
        noised_actions = tau_expanded * omega + (1 - tau_expanded) * actions_fm
        target_flow = omega - actions_fm

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            noised_actions, tau.squeeze(1)
        )

        pad_masks = torch.cat([prefix_pad_masks_fm, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = modeling_smolvla.make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Run unified forward pass with Knowledge Insulation:
        # - KI patch on forward_attn_layer handles self-attention layers
        #   (torch.no_grad for efficiency)
        # - VLM param freeze handles cross-attention layers and residual/MLP
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs_fm, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )

        # Restore VLM parameter grad states (needed for AR loss backward)
        for name, param in self.vlm_with_expert.vlm.named_parameters():
            param.requires_grad_(vlm_param_grad_states[name])

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        predicted_flow = self.action_out_proj(
            suffix_out.to(next(self.action_out_proj.parameters()).dtype)
        )

        pred = predicted_flow.to(torch.float32)
        target = target_flow.to(torch.float32)
        if self.config.loss_n_action_steps > 0:
            n = self.config.loss_n_action_steps
            pred = pred[:, :n]
            target = target[:, :n]
        flow_loss = F.mse_loss(pred, target)

        total_loss = (
            self.config.ar_loss_weight * ar_loss
            + self.config.fm_loss_weight * flow_loss
        )
        return total_loss, ar_loss, flow_loss

    def generate_action(self, batch, chunk_size=None):
        """Generate action tokens including observation state with positive advantage conditioning."""
        if self.config.use_advantage_conditioning:
            batch = dict(batch)
            lang_tokens = batch["observation.language.tokens"]
            lang_masks = batch["observation.language.attention_mask"]
            device = lang_tokens.device
            adv_tokens = torch.full(
                (lang_tokens.shape[0], 1),
                self.adv_pos_id,
                dtype=torch.long,
                device=device,
            )
            batch["observation.language.tokens"] = torch.cat(
                [lang_tokens, adv_tokens], dim=1
            )
            batch["observation.language.attention_mask"] = torch.cat(
                [lang_masks, torch.ones_like(adv_tokens, dtype=torch.bool)], dim=1
            )
        return self.fast_wrapper.generate_action(batch, chunk_size=chunk_size)

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        cfg_weight=None,
        **kwargs,
    ):
        """Override to append advantage conditioning and apply Classifier-Free Guidance during inference."""
        if cfg_weight is None:
            cfg_weight = getattr(self.config, "cfg_weight", 1.0)

        use_adv = self.config.use_advantage_conditioning

        # Temporary state for denoise_step override
        self._cfg_weight = cfg_weight
        self._in_cfg_mode = use_adv and cfg_weight != 1.0

        if self.config.snapflow_enabled:
            # SnapFlow 1-step inference
            device = lang_tokens.device
            bsize = lang_tokens.shape[0]

            if not use_adv:
                pass
            else:
                adv_tokens = torch.full(
                    (bsize, 1),
                    self.adv_pos_id,
                    dtype=torch.long,
                    device=device,
                )

                if cfg_weight == 1.0:
                    # Standard conditional generation (No CFG)
                    lang_tokens = torch.cat([lang_tokens, adv_tokens], dim=1)
                    lang_masks = torch.cat(
                        [lang_masks, torch.ones_like(adv_tokens, dtype=torch.bool)],
                        dim=1,
                    )
                elif cfg_weight == 0.0:
                    self._in_cfg_mode = False  # Ensure no CFG blending
                else:
                    # CFG Mode (cfg_weight != 1.0 and != 0.0)
                    uncond_tokens = torch.cat(
                        [lang_tokens, torch.zeros_like(adv_tokens)], dim=1
                    )
                    uncond_masks = torch.cat(
                        [lang_masks, torch.zeros_like(adv_tokens, dtype=torch.bool)],
                        dim=1,
                    )

                    cond_tokens = torch.cat([lang_tokens, adv_tokens], dim=1)
                    cond_masks = torch.cat(
                        [lang_masks, torch.ones_like(adv_tokens, dtype=torch.bool)],
                        dim=1,
                    )

                    # Double all inputs: [uncond, cond]
                    lang_tokens = torch.cat([uncond_tokens, cond_tokens], dim=0)
                    lang_masks = torch.cat([uncond_masks, cond_masks], dim=0)

                    state = torch.cat([state, state], dim=0)
                    images = [torch.cat([img, img], dim=0) for img in images]
                    img_masks = [torch.cat([mask, mask], dim=0) for mask in img_masks]

                    if noise is None:
                        actions_shape = (
                            bsize,
                            self.config.chunk_size,
                            self.config.max_action_dim,
                        )
                        single_noise = self.sample_noise(actions_shape, device)
                        noise = torch.cat([single_noise, single_noise], dim=0)
                    else:
                        noise = torch.cat([noise, noise], dim=0)

            final_bsize = state.shape[0]
            final_device = state.device
            if noise is None:
                actions_shape = (
                    final_bsize,
                    self.config.chunk_size,
                    self.config.max_action_dim,
                )
                noise = self.sample_noise(actions_shape, final_device)

            # Cache VLM prefix
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            prefix_att_2d_masks = modeling_smolvla.make_att_2d_masks(
                prefix_pad_masks, prefix_att_masks
            )
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            _, past_key_values = self.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=self.config.use_cache,
                fill_kv_cache=True,
            )

            # Single denoise step: t=1 → s=0 (full jump)
            t = torch.ones(final_bsize, device=final_device)
            s = torch.zeros(final_bsize, device=final_device)

            def denoise_step_partial_call(input_x_t, current_timestep=t):
                return self.denoise_step_snapflow(
                    input_x_t, prefix_pad_masks, past_key_values, current_timestep, s
                )

            if (
                getattr(self, "_rtc_enabled", lambda: False)()
                and getattr(self, "rtc_processor", None) is not None
            ):
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                # If CFG is enabled, we need to double prev_chunk_left_over to match batch size
                if (
                    prev_chunk_left_over is not None
                    and use_adv
                    and cfg_weight != 1.0
                    and cfg_weight != 0.0
                ):
                    prev_chunk_left_over = torch.cat(
                        [prev_chunk_left_over, prev_chunk_left_over], dim=0
                    )

                v_t = self.rtc_processor.denoise_step(
                    x_t=noise,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=1.0,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(noise)

            x_0 = noise - v_t  # x_t + dt * v_t where dt = -1.0

            if use_adv and cfg_weight != 1.0 and cfg_weight != 0.0:
                # Return only the first half (since both halves will be identical after blending)
                return x_0[:bsize]
            return x_0

        if not use_adv:
            return super().sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
            )

        device = lang_tokens.device
        bsize = lang_tokens.shape[0]

        adv_tokens = torch.full(
            (bsize, 1),
            self.adv_pos_id,
            dtype=torch.long,
            device=device,
        )

        if cfg_weight == 1.0:
            # Standard conditional generation (No CFG)
            lang_tokens = torch.cat([lang_tokens, adv_tokens], dim=1)
            lang_masks = torch.cat(
                [lang_masks, torch.ones_like(adv_tokens, dtype=torch.bool)], dim=1
            )
            return super().sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
            )

        elif cfg_weight == 0.0:
            # Standard unconditional generation (No CFG)
            self._in_cfg_mode = False  # Ensure no CFG blending
            return super().sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
            )

        # CFG Mode (cfg_weight != 1.0 and != 0.0)
        # We need to double the batch size. First half is unconditional, second half is conditional.
        # Pad unconditional with 0s and mask=False to match sequence length of conditional
        uncond_tokens = torch.cat([lang_tokens, torch.zeros_like(adv_tokens)], dim=1)
        uncond_masks = torch.cat(
            [lang_masks, torch.zeros_like(adv_tokens, dtype=torch.bool)], dim=1
        )

        cond_tokens = torch.cat([lang_tokens, adv_tokens], dim=1)
        cond_masks = torch.cat(
            [lang_masks, torch.ones_like(adv_tokens, dtype=torch.bool)], dim=1
        )

        # Double all inputs: [uncond, cond]
        lang_tokens = torch.cat([uncond_tokens, cond_tokens], dim=0)
        lang_masks = torch.cat([uncond_masks, cond_masks], dim=0)

        state = torch.cat([state, state], dim=0)
        images = [torch.cat([img, img], dim=0) for img in images]
        img_masks = [torch.cat([mask, mask], dim=0) for mask in img_masks]

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            single_noise = self.sample_noise(actions_shape, device)
            noise = torch.cat([single_noise, single_noise], dim=0)
        else:
            noise = torch.cat([noise, noise], dim=0)

        # If RTC is enabled and prev_chunk_left_over is passed, we must double it too
        prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
        if prev_chunk_left_over is not None:
            kwargs["prev_chunk_left_over"] = torch.cat(
                [prev_chunk_left_over, prev_chunk_left_over], dim=0
            )

        actions = super().sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
        )

        # actions has shape (2 * bsize, chunk_size, action_dim)
        # Both halves are identical because denoise_step returns identical blended v_t for both
        return actions[:bsize]

    def denoise_step_snapflow(
        self, x_t, prefix_pad_masks, past_key_values, timestep, target_time
    ):
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            x_t, timestep, target_time=target_time
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        suffix_att_2d_masks = modeling_smolvla.make_att_2d_masks(
            suffix_pad_masks, suffix_att_masks
        )

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(
            suffix_out.to(next(self.action_out_proj.parameters()).dtype)
        )

        # Clamp velocity prediction magnitude
        v_t = v_t.clamp(-self.config.snapflow_clamp, self.config.snapflow_clamp)

        if getattr(self, "_in_cfg_mode", False):
            bsize = v_t.shape[0] // 2
            uncond_v_t = v_t[:bsize]
            cond_v_t = v_t[bsize:]

            # CFG formula: uncond + w * (cond - uncond)
            w = getattr(self, "_cfg_weight", 1.0)
            cfg_v_t = uncond_v_t + w * (cond_v_t - uncond_v_t)

            # Duplicate so x_t remains identical for both halves
            v_t = torch.cat([cfg_v_t, cfg_v_t], dim=0)

        return v_t

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        v_t = super().denoise_step(prefix_pad_masks, past_key_values, x_t, timestep)

        if getattr(self, "_in_cfg_mode", False):
            bsize = v_t.shape[0] // 2
            uncond_v_t = v_t[:bsize]
            cond_v_t = v_t[bsize:]

            # CFG formula: uncond + w * (cond - uncond)
            w = getattr(self, "_cfg_weight", 1.0)
            cfg_v_t = uncond_v_t + w * (cond_v_t - uncond_v_t)

            # Duplicate so x_t remains identical for both halves in the ODE solver
            v_t = torch.cat([cfg_v_t, cfg_v_t], dim=0)

        return v_t

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
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
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
            self._queues[ACTION].extend(
                actions.transpose(0, 1)[: self.config.n_action_steps]
            )

        return self._queues[ACTION].popleft()

    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,  # pylint: disable=unused-argument
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Predict a chunk of actions using the flow-matching expert."""
        self.eval()

        # Handle history tracking queues if needed
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = modeling_smolvla.SmolVLAPolicy.prepare_images(self, batch)
        state = modeling_smolvla.SmolVLAPolicy.prepare_state(self, batch)

        # Ensure correct dtype for inputs
        dtype = self.model.vlm_with_expert.vlm.dtype
        images = [img.to(dtype) for img in images]
        state = state.to(dtype)

        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]

        # Predict actions using the flow-matching expert
        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
        )

        if "action" in self.config.output_features:
            action_dim = self.config.output_features["action"].shape[0]
            actions = actions[:, :, :action_dim]

        return actions

    def forward(self, batch: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        raise NotImplementedError("Use self.model directly for forward/training")

    def get_optim_params(self) -> dict:
        return self.parameters()
