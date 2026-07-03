import torch
import torch.nn as nn
from lerobot_policy_smolvla_rl.analyze.attribution.policy_io import PrefixLayout


class AttentionRecorder:
    """Context manager that wraps eager_attention_forward to record softmax probs per layer."""

    def __init__(self, policy):
        self.policy = policy
        self.layers = []
        self.original_get_attention_interface = None

    def __enter__(self) -> "AttentionRecorder":
        vlm_model = self.policy.model.vlm_with_expert
        self.original_get_attention_interface = vlm_model.get_attention_interface

        def custom_attention_forward(
            attention_mask, batch_size, head_dim, query_states, key_states, value_states
        ):
            num_att_heads = vlm_model.num_attention_heads
            num_key_value_heads = vlm_model.num_key_value_heads
            num_key_value_groups = num_att_heads // num_key_value_heads

            sequence_length = key_states.shape[1]

            key_states = key_states[:, :, :, None, :].expand(
                batch_size,
                sequence_length,
                num_key_value_heads,
                num_key_value_groups,
                head_dim,
            )
            key_states = key_states.reshape(
                batch_size,
                sequence_length,
                num_key_value_heads * num_key_value_groups,
                head_dim,
            )

            value_states = value_states[:, :, :, None, :].expand(
                batch_size,
                sequence_length,
                num_key_value_heads,
                num_key_value_groups,
                head_dim,
            )
            value_states = value_states.reshape(
                batch_size,
                sequence_length,
                num_key_value_heads * num_key_value_groups,
                head_dim,
            )

            query_states = query_states.to(dtype=torch.float32)
            key_states = key_states.to(dtype=torch.float32)

            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

            att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
            att_weights *= head_dim**-0.5

            att_weights = att_weights.to(dtype=torch.float32)
            big_neg = torch.finfo(att_weights.dtype).min
            masked_att_weights = torch.where(
                attention_mask[:, None, :, :], att_weights, big_neg
            )
            probs = nn.functional.softmax(masked_att_weights, dim=-1)

            # Record probs
            self.layers.append(probs.detach().cpu())

            probs = probs.to(dtype=value_states.dtype)
            att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
            att_output = att_output.permute(0, 2, 1, 3)
            att_output = att_output.reshape(
                batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim
            )
            return att_output

        vlm_model.get_attention_interface = lambda: custom_attention_forward
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_get_attention_interface is not None:
            self.policy.model.vlm_with_expert.get_attention_interface = (
                self.original_get_attention_interface
            )


def action_to_image_attention(
    recorder, layout: PrefixLayout, *, reduce_heads: str = "mean", rollout: bool = False
) -> dict[str, torch.Tensor]:
    """Extracts attention mass from expert tokens onto camera image tokens, renormalized."""
    # Determine prefix length
    prefix_len = max(span.stop for span in layout.image_spans.values())
    prefix_len = max(prefix_len, layout.lang_span.stop)
    prefix_len = max(prefix_len, layout.state_span.stop)

    # Filter to get suffix layers
    suffix_layers = [layer for layer in recorder.layers if layer.shape[-1] > prefix_len]

    if not suffix_layers:
        raise ValueError("No recorded attention layers found matching suffix pass.")

    batch_size = suffix_layers[0].shape[0]
    device = suffix_layers[0].device
    policy = recorder.policy
    chunk_size = policy.config.chunk_size

    if rollout:
        L = getattr(
            getattr(policy.model.vlm_with_expert.vlm, "config", None),
            "num_hidden_layers",
            16,
        )
        num_layers = len(suffix_layers)
        if L <= 0 or L > num_layers:
            L = num_layers

        num_steps = num_layers // L
        step_attns = []

        for s in range(num_steps):
            step_layers = suffix_layers[s * L : (s + 1) * L]
            if not step_layers:
                continue

            kv_len = step_layers[0].shape[-1]
            R = torch.eye(kv_len, device=device).unsqueeze(0).expand(batch_size, -1, -1)

            for layer in step_layers:
                # layer: (batch, heads, q_len, kv_len)
                # target device might be CPU if we detached it, let's keep it on same device as layer
                layer_device = layer.device
                R = R.to(layer_device)

                if reduce_heads == "mean":
                    probs_mean = layer.mean(dim=1)  # (batch, q_len, kv_len)
                else:
                    probs_mean = layer.max(dim=1).values

                A = torch.zeros(batch_size, kv_len, kv_len, device=layer_device)
                A[:, :prefix_len, :prefix_len] = torch.eye(
                    prefix_len, device=layer_device
                )
                A[:, prefix_len:, :] = probs_mean

                A_scaled = 0.5 * A + 0.5 * torch.eye(kv_len, device=layer_device)
                R = torch.bmm(A_scaled, R)

            # Extract action queries to prefix
            rollout_attn = R[:, -chunk_size:, :prefix_len]
            attn = rollout_attn.mean(dim=1)
            step_attns.append(attn.cpu())

        if step_attns:
            final_attn = torch.stack(step_attns, dim=0).mean(dim=0)
        else:
            final_attn = torch.zeros(batch_size, prefix_len)

    else:
        attn_list = []
        for layer in suffix_layers:
            attn_action_to_prefix = layer[:, :, -chunk_size:, :prefix_len]
            if reduce_heads == "mean":
                attn_layer = attn_action_to_prefix.mean(dim=1)
            else:
                attn_layer = attn_action_to_prefix.max(dim=1).values

            attn_layer = attn_layer.mean(dim=1)
            attn_list.append(attn_layer.cpu())

        final_attn = torch.stack(attn_list, dim=0).mean(dim=0)

    denom = final_attn.sum(dim=-1, keepdim=True) + 1e-8
    final_attn_norm = final_attn / denom

    camera_attns = {}
    for cam_key, span in layout.image_spans.items():
        camera_attns[cam_key] = final_attn_norm[:, span.start : span.stop]

    return camera_attns
