import torch
import torch.nn.functional as F
import types
from pathlib import Path
import click
import pandas as pd
from lerobot_policy_smolvla_rl.analyze.attribution.policy_io import load_recap_policy, iter_eval_batches

def make_recording_forward(original_forward, record_dict):
    def recording_forward(
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=True,
        fill_kv_cache=False,
        **kwargs
    ):
        vlm_hidden, expert_hidden = inputs_embeds
        
        # We loop through layers just like the original forward
        models = [original_forward.__self__.get_vlm_model().text_model, original_forward.__self__.lm_expert]
        model_layers = original_forward.__self__.get_model_layers(models)
        
        for hidden_states in inputs_embeds:
            if hidden_states is not None:
                batch_size = hidden_states.shape[0]
                
        head_dim = original_forward.__self__.vlm.config.text_config.head_dim
        inputs = inputs_embeds
        
        for layer_idx in range(original_forward.__self__.num_vlm_layers):
            # Record input
            if inputs[0] is not None:
                record_dict["vlm_inputs"][layer_idx].append(inputs[0].detach().cpu())
            if inputs[1] is not None:
                record_dict["exp_inputs"][layer_idx].append(inputs[1].detach().cpu())
                
            if (
                fill_kv_cache
                or "cross" not in original_forward.__self__.attention_mode
                or (original_forward.__self__.self_attn_every_n_layers > 0 and layer_idx % original_forward.__self__.self_attn_every_n_layers == 0)
            ):
                att_outputs, past_key_values = original_forward.__self__.forward_attn_layer(
                    model_layers, inputs, layer_idx, position_ids, attention_mask,
                    batch_size, head_dim, use_cache=use_cache, fill_kv_cache=fill_kv_cache, past_key_values=past_key_values
                )
            else:
                att_outputs, past_key_values = original_forward.__self__.forward_cross_attn_layer(
                    model_layers, inputs, layer_idx, position_ids, attention_mask,
                    batch_size, head_dim, use_cache=use_cache, fill_kv_cache=fill_kv_cache, past_key_values=past_key_values
                )
                
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs):
                layer = model_layers[i][layer_idx]
                att_output = att_outputs[i] if i < len(att_outputs) else att_outputs[0]
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
                    
            # Record output
            if outputs_embeds[0] is not None:
                record_dict["vlm_outputs"][layer_idx].append(outputs_embeds[0].detach().cpu())
            if outputs_embeds[1] is not None:
                record_dict["exp_outputs"][layer_idx].append(outputs_embeds[1].detach().cpu())
                
            inputs = outputs_embeds
            
        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)
        return outputs_embeds, past_key_values
    return recording_forward

def run_layer_redundancy_analysis(policy, dataset_repo_id, episodes=None, calibration_frames=256, device="cuda"):
    vlm_model = policy.model.vlm_with_expert
    num_layers = vlm_model.num_vlm_layers
    
    # Initialize record dict
    record_dict = {
        "vlm_inputs": {i: [] for i in range(num_layers)},
        "vlm_outputs": {i: [] for i in range(num_layers)},
        "exp_inputs": {i: [] for i in range(num_layers)},
        "exp_outputs": {i: [] for i in range(num_layers)},
    }
    
    # Monkeypatch the forward method of SmolVLMWithExpertModel
    original_forward = vlm_model.forward
    vlm_model.forward = types.MethodType(make_recording_forward(original_forward, record_dict), vlm_model)
    
    frames_processed = 0
    click.echo(f"Collecting activation hidden states for calibration (target: {calibration_frames} frames)...")
    
    try:
        for batch in iter_eval_batches(dataset_repo_id, policy, episodes=episodes, stride=1, batch_size=8):
            # Put batch on device
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Select action executes the forward pass
            with torch.no_grad():
                policy.select_action(batch_device)
                
            # Each batch has 8 frames (since batch_size=8)
            frames_processed += batch_device["action"].shape[0]
            if frames_processed >= calibration_frames:
                break
    finally:
        # Restore original forward method
        vlm_model.forward = original_forward
        
    click.echo(f"Finished activation collection ({frames_processed} frames). Computing importance scores...")
    
    # Compute similarities per layer
    vlm_cos_sims = {}
    exp_cos_sims = {}
    
    for l in range(num_layers):
        # VLM Similarity
        vlm_in = record_dict["vlm_inputs"][l]
        vlm_out = record_dict["vlm_outputs"][l]
        
        if len(vlm_in) > 0 and len(vlm_out) > 0:
            vlm_in_t = torch.cat(vlm_in, dim=0) # (Total_frames, Seq_len, D)
            vlm_out_t = torch.cat(vlm_out, dim=0) # (Total_frames, Seq_len, D)
            
            # Compute cosine similarity along feature dimension D
            # shape: (Total_frames, Seq_len)
            sim = F.cosine_similarity(vlm_in_t, vlm_out_t, dim=-1)
            vlm_cos_sims[l] = sim.mean().item()
        else:
            vlm_cos_sims[l] = 1.0 # default to 1.0 (no change, i.e., completely redundant)
            
        # Expert Similarity
        exp_in = record_dict["exp_inputs"][l]
        exp_out = record_dict["exp_outputs"][l]
        
        if len(exp_in) > 0 and len(exp_out) > 0:
            exp_in_t = torch.cat(exp_in, dim=0) # (Total_frames, Seq_len, D)
            exp_out_t = torch.cat(exp_out, dim=0) # (Total_frames, Seq_len, D)
            
            sim = F.cosine_similarity(exp_in_t, exp_out_t, dim=-1)
            exp_cos_sims[l] = sim.mean().item()
        else:
            exp_cos_sims[l] = 1.0
            
    # Normalize redundancies to [0, 1] range
    vlm_sims_val = list(vlm_cos_sims.values())
    exp_sims_val = list(exp_cos_sims.values())
    
    vlm_min, vlm_max = min(vlm_sims_val), max(vlm_sims_val)
    exp_min, exp_max = min(exp_sims_val), max(exp_sims_val)
    
    # Normalize cosine similarities: higher similarity = higher redundancy
    # We normalized to [0, 1]
    def normalize_score(val, min_val, max_val):
        if max_val - min_val < 1e-6:
            return 1.0
        return (val - min_val) / (max_val - min_val)
        
    normalized_vlm = {l: normalize_score(vlm_cos_sims[l], vlm_min, vlm_max) for l in range(num_layers)}
    normalized_exp = {l: normalize_score(exp_cos_sims[l], exp_min, exp_max) for l in range(num_layers)}
    
    # Combined score = min of normalized redundancy scores
    # A layer is only redundant if it is redundant in both streams
    combined_redundancy = {
        l: min(normalized_vlm[l], normalized_exp[l]) for l in range(num_layers)
    }
    
    # Build ranked table
    rows = []
    for l in range(num_layers):
        status = "Kept (Constraint)" if (l == 0 or l == num_layers - 1) else "Candidate"
        rows.append({
            "Layer Index": l,
            "VLM Similarity (Redundancy)": f"{vlm_cos_sims[l]:.4f}",
            "Expert Similarity (Redundancy)": f"{exp_cos_sims[l]:.4f}",
            "Normalized VLM Redundancy": f"{normalized_vlm[l]:.4f}",
            "Normalized Expert Redundancy": f"{normalized_exp[l]:.4f}",
            "Combined Redundancy Score": combined_redundancy[l],
            "Pruning Status": status
        })
        
    df = pd.DataFrame(rows)
    # Sort candidate layers by combined redundancy score in descending order (most redundant first)
    # Put non-prunable layers at the bottom or mark them
    candidates_df = df[df["Pruning Status"] == "Candidate"].sort_values(by="Combined Redundancy Score", ascending=False)
    constraints_df = df[df["Pruning Status"] == "Kept (Constraint)"]
    
    final_df = pd.concat([candidates_df, constraints_df], ignore_index=True)
    # Format Combined Redundancy Score column
    final_df["Combined Redundancy Score"] = final_df["Combined Redundancy Score"].map(lambda x: f"{x:.4f}")
    return final_df
