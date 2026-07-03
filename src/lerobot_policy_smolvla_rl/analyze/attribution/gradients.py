import torch
from lerobot.policies.smolvla import modeling_smolvla
from lerobot_policy_smolvla_rl.analyze.attribution.policy_io import (
    PrefixLayout,
    camera_keys,
)


def grad_x_input_attribution(
    policy, batch, layout: PrefixLayout, *, action_dims: slice | None = None
) -> dict[str, torch.Tensor]:
    """Computes Grad*Input attribution over image token embeddings."""
    device = next(policy.parameters()).device
    recap_batch = batch["recap_batch"]
    original_batch = batch["batch"]

    # 1. Prepare images, state, etc.
    images, img_masks = policy.prepare_images(original_batch)
    state = policy.prepare_state(original_batch)

    dtype = policy.model.vlm_with_expert.vlm.dtype
    images = [img.to(dtype) for img in images]
    state = state.to(dtype)

    lang_tokens = recap_batch["observation.language.tokens"]
    lang_masks = recap_batch["observation.language.attention_mask"]

    # Enable grad and wrap embed_image to catch embeddings
    saved_embeddings = []
    original_embed_image = policy.model.vlm_with_expert.embed_image

    def wrapped_embed_image(img):
        emb = original_embed_image(img)
        emb = emb.detach().requires_grad_(True)
        saved_embeddings.append(emb)
        return emb

    policy.model.vlm_with_expert.embed_image = wrapped_embed_image

    try:
        with torch.enable_grad():
            bsize = state.shape[0]
            actions_shape = (
                bsize,
                policy.config.chunk_size,
                policy.config.max_action_dim,
            )
            noise = policy.model.sample_noise(actions_shape, device)

            prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )

            prefix_att_2d_masks = modeling_smolvla.make_att_2d_masks(
                prefix_pad_masks, prefix_att_masks
            )
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            _, past_key_values = policy.model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=policy.config.use_cache,
                fill_kv_cache=True,
            )

            t_tensor = torch.ones(bsize, device=device)
            v_t = policy.model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=noise,
                timestep=t_tensor,
            )

            if action_dims is not None:
                v_t_slice = v_t[:, :, action_dims]
            else:
                v_t_slice = v_t

            target = (v_t_slice**2).sum(dim=(1, 2))  # (batch,)
            target.sum().backward()

    finally:
        policy.model.vlm_with_expert.embed_image = original_embed_image

    camera_attns = {}
    image_keys = camera_keys(policy)

    for cam_idx, cam_key in enumerate(image_keys):
        num_tokens = len(layout.image_spans[cam_key])
        if cam_idx < len(saved_embeddings):
            emb = saved_embeddings[cam_idx]
            grad = emb.grad
            if grad is not None:
                attr = (grad * emb).sum(dim=-1)
                camera_attns[cam_key] = attr.abs().cpu()
            else:
                camera_attns[cam_key] = torch.zeros(bsize, num_tokens)
        else:
            camera_attns[cam_key] = torch.zeros(bsize, num_tokens)

    return camera_attns


def integrated_gradients_attribution(
    policy,
    batch,
    layout: PrefixLayout,
    *,
    steps: int = 16,
    action_dims: slice | None = None,
) -> dict[str, torch.Tensor]:
    """Computes Integrated Gradients attribution over image token embeddings."""
    device = next(policy.parameters()).device
    recap_batch = batch["recap_batch"]
    original_batch = batch["batch"]

    # Prepare inputs
    images, img_masks = policy.prepare_images(original_batch)
    state = policy.prepare_state(original_batch)

    dtype = policy.model.vlm_with_expert.vlm.dtype
    images = [img.to(dtype) for img in images]
    state = state.to(dtype)

    lang_tokens = recap_batch["observation.language.tokens"]
    lang_masks = recap_batch["observation.language.attention_mask"]

    bsize = state.shape[0]
    actions_shape = (bsize, policy.config.chunk_size, policy.config.max_action_dim)
    noise = policy.model.sample_noise(actions_shape, device)

    original_embeddings = []
    original_embed_image = policy.model.vlm_with_expert.embed_image

    with torch.no_grad():
        for img in images:
            original_embeddings.append(original_embed_image(img))

    accumulated_grads = [torch.zeros_like(emb) for emb in original_embeddings]

    for step in range(1, steps + 1):
        alpha = step / steps
        saved_embeddings = []

        def wrapped_embed_image(img):
            cam_idx = len(saved_embeddings)
            orig_emb = original_embeddings[cam_idx]

            # Linear interpolation from baseline (zeros) to orig_emb
            interpolated = alpha * orig_emb
            interpolated = interpolated.detach().requires_grad_(True)
            saved_embeddings.append(interpolated)
            return interpolated

        policy.model.vlm_with_expert.embed_image = wrapped_embed_image

        try:
            with torch.enable_grad():
                prefix_embs, prefix_pad_masks, prefix_att_masks = (
                    policy.model.embed_prefix(
                        images, img_masks, lang_tokens, lang_masks, state=state
                    )
                )
                prefix_att_2d_masks = modeling_smolvla.make_att_2d_masks(
                    prefix_pad_masks, prefix_att_masks
                )
                prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

                _, past_key_values = policy.model.vlm_with_expert.forward(
                    attention_mask=prefix_att_2d_masks,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=policy.config.use_cache,
                    fill_kv_cache=True,
                )

                t_tensor = torch.ones(bsize, device=device)
                v_t = policy.model.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=noise,
                    timestep=t_tensor,
                )

                if action_dims is not None:
                    v_t_slice = v_t[:, :, action_dims]
                else:
                    v_t_slice = v_t

                target = (v_t_slice**2).sum(dim=(1, 2))
                target.sum().backward()

                for cam_idx, emb in enumerate(saved_embeddings):
                    if emb.grad is not None:
                        accumulated_grads[cam_idx] += emb.grad

        finally:
            policy.model.vlm_with_expert.embed_image = original_embed_image

    # Compute attribution: (orig_emb - baseline) * average_grad
    camera_attns = {}
    image_keys = camera_keys(policy)

    for cam_idx, cam_key in enumerate(image_keys):
        num_tokens = len(layout.image_spans[cam_key])
        if cam_idx < len(original_embeddings):
            orig_emb = original_embeddings[cam_idx]
            avg_grad = accumulated_grads[cam_idx] / steps
            attr = (orig_emb * avg_grad).sum(dim=-1)
            camera_attns[cam_key] = attr.abs().cpu()
        else:
            camera_attns[cam_key] = torch.zeros(bsize, num_tokens)

    return camera_attns
