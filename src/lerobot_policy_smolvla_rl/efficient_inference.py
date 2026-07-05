import torch
import torch.nn.functional as F
from lerobot.policies.smolvla import modeling_smolvla
from lerobot_policy_smolvla_rl.analyze.attribution.policy_io import prefix_layout, camera_keys

def select_diverse_tokens(visual_tokens, relevance_scores, K_keep, K_relevance, alpha=0.5):
    """
    Select K_keep tokens per camera using a 3-stage selection procedure:
    Stage 1: Top K_relevance by relevance score.
    Stage 2: Rest by diversity (greedily maximize 1 - max cos_sim(v_j, V_selected)).
    Stage 3: Sort selected indices to preserve spatial order.
    
    Args:
        visual_tokens: (B, N_cam_tokens, D) - typically N_cam_tokens = 64
        relevance_scores: (B, N_cam_tokens)
        K_keep: total tokens to keep per camera
        K_relevance: number of tokens to keep by relevance
        alpha: relevance vs diversity split
    """
    B, N, D = visual_tokens.shape
    device = visual_tokens.device
    
    # L2 normalize the tokens for cosine similarity
    norm_tokens = F.normalize(visual_tokens, p=2, dim=-1) # (B, N, D)
    
    # Stage 1: Get top K_relevance indices by relevance
    _, rel_indices = torch.topk(relevance_scores, K_relevance, dim=-1) # (B, K_relevance)
    
    selected_indices = []
    for b in range(B):
        sel_set = rel_indices[b].tolist()
        tokens_b = norm_tokens[b] # (N, D)
        
        while len(sel_set) < K_keep:
            # Candidates are indices not yet selected
            candidates = [idx for idx in range(N) if idx not in sel_set]
            if not candidates:
                break
                
            cand_tokens = tokens_b[candidates] # (C, D)
            sel_tokens = tokens_b[sel_set]     # (S, D)
            
            # Compute cosine similarity matrix: (C, S)
            sim_matrix = torch.matmul(cand_tokens, sel_tokens.T) # (C, S)
            
            # For each candidate, find its maximum similarity to any selected token
            max_sims, _ = torch.max(sim_matrix, dim=-1) # (C,)
            
            # We want to minimize the maximum similarity (maximize diversity)
            best_cand_idx = torch.argmin(max_sims).item()
            best_global_idx = candidates[best_cand_idx]
            
            sel_set.append(best_global_idx)
            
        selected_indices.append(sel_set)
        
    selected_tensor = torch.tensor(selected_indices, dtype=torch.long, device=device) # (B, K_keep)
    # Stage 3: Sort the indices to preserve spatial layout
    sorted_selected_tensor = torch.sort(selected_tensor, dim=-1).values
    return sorted_selected_tensor

def make_eager_attention_hook(model_wrapper, original_eager_fn):
    """
    Monkeypatch wrapper around eager_attention_forward to record probs at layer 1 (the 2nd layer)
    during C2 scoring pass.
    """
    def wrapped_eager_fn(attention_mask, batch_size, head_dim, query_states, key_states, value_states):
        att_output = original_eager_fn(
            attention_mask, batch_size, head_dim, query_states, key_states, value_states
        )
        
        # If recording is active and we are at layer 1 (the 2nd layer)
        if getattr(model_wrapper, "_record_attention", False) and getattr(model_wrapper, "_current_layer_idx", -1) == 1:
            num_att_heads = model_wrapper.num_attention_heads
            num_key_value_heads = model_wrapper.num_key_value_heads
            num_key_value_groups = num_att_heads // num_key_value_heads
            sequence_length = key_states.shape[1]
            
            k_states = key_states[:, :, :, None, :].expand(
                batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
            ).reshape(batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim)
            
            q_states = query_states.to(dtype=torch.float32)
            k_states = k_states.to(dtype=torch.float32)
            q_states = q_states.transpose(1, 2)
            k_states = k_states.transpose(1, 2)
            
            att_weights = torch.matmul(q_states, k_states.transpose(2, 3)) * (head_dim ** -0.5)
            
            # Mask fill
            if attention_mask.ndim == 2:
                attn_mask = attention_mask[:, None, None, :]
            elif attention_mask.ndim == 3:
                attn_mask = attention_mask[:, None, :, :]
            else:
                attn_mask = attention_mask
                
            big_neg = torch.finfo(att_weights.dtype).min
            masked_att_weights = torch.where(attn_mask, att_weights, big_neg)
            probs = F.softmax(masked_att_weights, dim=-1)
            
            model_wrapper._recorded_attention_probs = probs.detach().cpu()
            
        return att_output
    return wrapped_eager_fn

def select_visual_tokens(policy_model, images, img_masks, lang_tokens, lang_masks, state=None):
    """
    Runs the 2-layer scoring forward pass and returns the dictionary of kept visual token indices.
    """
    model_wrapper = policy_model.vlm_with_expert
    K_keep = policy_model.config.visual_tokens_keep
    alpha = policy_model.config.token_prune_alpha
    K_key = policy_model.config.token_prune_k_key
    K_aug = K_keep - K_key
    K_relevance = K_key + int(alpha * K_aug)
    
    # 1. CFG slice (run scoring on cond half only)
    if getattr(policy_model, "_in_cfg_mode", False):
        B = images[0].shape[0] // 2
        score_images = [img[B:] for img in images]
        score_img_masks = [mask[B:] for mask in img_masks]
        score_lang_tokens = lang_tokens[B:]
        score_lang_masks = lang_masks[B:]
        score_state = state[B:] if state is not None else None
    else:
        score_images = images
        score_img_masks = img_masks
        score_lang_tokens = lang_tokens
        score_lang_masks = lang_masks
        score_state = state
        
    # 2. Get full unpruned prefix embeddings for scoring
    # pass bypass_pruning=True to avoid recursion
    full_embs, full_pad_masks, full_att_masks = policy_model.embed_prefix(
        score_images, score_img_masks, score_lang_tokens, score_lang_masks, score_state, bypass_pruning=True
    )
    
    # 3. Initialize attention recording on wrapper
    model_wrapper._record_attention = True
    model_wrapper._recorded_attention_probs = None
    
    full_att_2d_masks = modeling_smolvla.make_att_2d_masks(full_pad_masks, full_att_masks)
    full_position_ids = torch.cumsum(full_pad_masks, dim=1) - 1
    
    # 4. Forward first 2 VLM layers without filling cache
    with torch.no_grad():
        model_wrapper.forward(
            attention_mask=full_att_2d_masks,
            position_ids=full_position_ids,
            past_key_values=None,
            inputs_embeds=[full_embs, None],
            use_cache=False,
            fill_kv_cache=False,
            limit_layers=2
        )
        
    model_wrapper._record_attention = False
    probs = model_wrapper._recorded_attention_probs
    if probs is None:
        raise RuntimeError("Failed to capture layer-1 attention probabilities during scoring pass.")
        
    # 5. Extract scores per camera
    layout = prefix_layout(policy_model, lang_len=score_lang_tokens.shape[1])
    img_keys = camera_keys(policy_model)
    add_special = getattr(policy_model, "add_image_special_tokens", False)
    
    camera_kept_indices = {}
    for img_idx, cam_key in enumerate(img_keys):
        span = layout.image_spans[cam_key]
        if add_special:
            visual_indices = list(span)[1:-1]
        else:
            visual_indices = list(span)
            
        lang_indices = list(layout.lang_span)
        
        # probs shape: (B_score, num_heads, prefix_len, prefix_len)
        # attn_lang_to_vision shape: (B_score, num_heads, len(lang_indices), len(visual_indices))
        attn_lang_to_vision = probs[:, :, lang_indices][:, :, :, visual_indices]
        relevance = attn_lang_to_vision.mean(dim=(1, 2)) # (B_score, len(visual_indices))
        
        # get visual token embeddings from full_embs
        visual_tokens = full_embs[:, visual_indices, :] # (B_score, len(visual_indices), D)
        
        # Stage 2: Selection
        kept_indices = select_diverse_tokens(visual_tokens, relevance, K_keep, K_relevance, alpha)
        
        # If CFG, duplicate kept_indices to match 2B batch size
        if getattr(policy_model, "_in_cfg_mode", False):
            kept_indices = torch.cat([kept_indices, kept_indices], dim=0)
            
        camera_kept_indices[cam_key] = kept_indices
        
    return camera_kept_indices

def check_pruning_constraints(pruned_layers, num_layers, self_attn_every_n_layers=2):
    if not pruned_layers:
        return
    if any(l < 0 or l >= num_layers for l in pruned_layers):
        raise ValueError(f"Pruned layer indices must be in range [0, {num_layers - 1}].")
    if 0 in pruned_layers or (num_layers - 1) in pruned_layers:
        raise ValueError("Cannot prune the first (0) or final layer.")
        
    kept_layers = [i for i in range(num_layers) if i not in pruned_layers]
    
    # Identify contiguous segments
    segments = []
    current_segment = [kept_layers[0]]
    for x in kept_layers[1:]:
        if x == current_segment[-1] + 1:
            current_segment.append(x)
        else:
            segments.append(current_segment)
            current_segment = [x]
    segments.append(current_segment)
    
    # Validate cross-attention presence in each segment
    for seg in segments:
        has_cross_attn = any(idx % self_attn_every_n_layers != 0 for idx in seg)
        if not has_cross_attn:
            raise ValueError(f"Contiguous kept layer segment {seg} lacks a cross-attention block.")

def make_kv_reindex_wrapper(model_wrapper, original_attn_fn):
    def wrapped_attn_fn(
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache=True,
        fill_kv_cache=True,
        past_key_values=None,
        **kwargs
    ):
        # Translate layer_idx to cache_idx for past_key_values dictionary operations
        cache_idx = layer_idx
        if getattr(model_wrapper, "_layer_idx_to_cache_idx", None) is not None:
            cache_idx = model_wrapper._layer_idx_to_cache_idx.get(layer_idx, layer_idx)

        # Map cache_idx to layer_idx key internally for past_key_values lookup
        mapped_kv = None
        if past_key_values is not None:
            mapped_kv = {}
            if cache_idx in past_key_values:
                mapped_kv[layer_idx] = past_key_values[cache_idx]

        # Call original function (KI patch or original class implementation)
        outputs, updated_kv = original_attn_fn(
            model_layers,
            inputs_embeds,
            layer_idx,  # passed unchanged for weights lookup
            position_ids,
            attention_mask,
            batch_size,
            head_dim,
            use_cache=use_cache,
            fill_kv_cache=fill_kv_cache,
            past_key_values=mapped_kv,
            **kwargs
        )

        # Write back changes from layer_idx key to cache_idx key
        if use_cache and past_key_values is not None and updated_kv is not None:
            if layer_idx in updated_kv:
                past_key_values[cache_idx] = updated_kv[layer_idx]

        return outputs, past_key_values
    return wrapped_attn_fn

def make_patched_forward(model_wrapper, original_forward):
    def patched_forward(
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        limit_layers: int | None = None,
    ):
        if use_cache and past_key_values is None:
            past_key_values = {}

        pruned_layers = getattr(model_wrapper, "pruned_layers", None)
        if pruned_layers is None:
            pruned_layers = []

        models = [model_wrapper.get_vlm_model().text_model, model_wrapper.lm_expert]
        model_layers = model_wrapper.get_model_layers(models)

        batch_size = 1
        for hidden_states in inputs_embeds:
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        num_layers = model_wrapper.num_vlm_layers
        head_dim = model_wrapper.vlm.config.text_config.head_dim

        for layer_idx in range(num_layers):
            if limit_layers is not None and layer_idx >= limit_layers:
                break

            if limit_layers is None and layer_idx in pruned_layers:
                continue

            model_wrapper._current_layer_idx = layer_idx

            if (
                fill_kv_cache
                or "cross" not in model_wrapper.attention_mode
                or (model_wrapper.self_attn_every_n_layers > 0 and layer_idx % model_wrapper.self_attn_every_n_layers == 0)
            ):
                att_outputs, past_key_values = model_wrapper.forward_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            else:
                att_outputs, past_key_values = model_wrapper.forward_cross_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )

            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                att_output = (
                    att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                )
                if hidden_states is not None:
                    if layer is None:
                        outputs_embeds.append(hidden_states)
                        continue
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    att_out = att_output[:, start:end]
                    out_emb = layer.self_attn.o_proj(att_out)

                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()

                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    out_emb += after_first_residual
                    outputs_embeds.append(out_emb)

                    start = end if len(att_outputs) == 1 else 0
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)
        return outputs_embeds, past_key_values

    return patched_forward

