import math
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

C_FAIL = 10000

"""
Train a separate, generalized distributional critic to estimate the expected steps-to-success for any given state.

C. Reward definition and value function training
Since our aim is to develop a general and broadly applicable
method for training VLAs from experience, we use a general
sparse reward definition that can be applied to essentially any
task. For each episode, we obtain a label indicating whether
that episode was successful. We derive the reward from
this episode-level success label such that the value function
corresponds to the (negative) number of steps until successful
completion of the episode. This is equivalent to the following
reward function, where Tcorresponds to the last step in the
episode, and Cfailis a large constant that is chosen so as to
ensure that failed episodes have low values:
0 if t = T and success
-C_FAIL if t = T and failure
-1 otherwise.

With this reward function, we train the value function to
predict the (negative of the) number of remaining steps until
success for successful episodes, and a large negative value for
failed episodes. In practice, we normalize the values predicted
to be between (−1,0). Since we train on diverse tasks that
have very different typical lengths, we normalize the values
per task based on the maximum episode length of the task.
The value function takes as input the same language inputs
as the π∗0.6VLA, and uses the same architecture design, with a
smaller 670M parameter VLM backbone that is also initialized
from Gemma 3 (see Figure 3).

To-Do Items:
    [x] Instantiate a completely separate, compact VLM backbone for the critic (e.g., a 670M parameter VLM).
    [x] Attach a C51 categorical binning head to predict a distribution over normalized steps-to-completion.
    [x] Define your sparse reward (e.g., 0 for success, -C_FAIL for failure, -1 for intermediate steps).
    [x] Train on a dataset of trajectories with known outcomes, using the defined reward to compute target values for the critic.
    
Notes:
- For pretraining the critic use a dataset of trajectories and assume they all are successfull: lerobot/droid_1.0.1
"""

"""_summary_
        "observation.state": {
            "dtype": "float32",
            "shape": [
                8
            ],
            "names": {
                "axes": [
                    "joint_0",
                    "joint_1",
                    "joint_2",
                    "joint_3",
                    "joint_4",
                    "joint_5",
                    "joint_6",
                    "gripper"
                ]
            }
        },
"""


def pad_tensor(tensor, max_len, pad_value=0):
    b, d = tensor.shape[:2]
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded_tensor[:, :d] = tensor
    return padded_tensor


class SmolVLMWithCriticModel(nn.Module):
    """
    Critic for estimating expected steps-to-success, built on a compact VLM backbone.

    Input is a LeRobot observation ("observation.state", "observation.images.wrist_left", "observation.images.exterior_1_left")

    Output is a distribution over normalized steps-to-completion, represented as a C51 categorical distribution.

    Load backbone from model_id, reduce size by layer skipping, attach C51 head, implement forward pass.

    """

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        num_vlm_layers: int = -1,
        max_state_dim: int = 8,
        num_bins: int = 51,
        vmin: float = -1.0,
        vmax: float = 0.0,
        add_image_special_tokens: bool = True,
        prefix_length: int = 0,
        freeze_vision_encoder: bool = True,
    ):
        super().__init__()
        self.model_id = model_id
        self.max_state_dim = max_state_dim
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.add_image_special_tokens = add_image_special_tokens
        self.prefix_length = prefix_length
        self.freeze_vision_encoder = freeze_vision_encoder

        print(f"Loading {model_id} weights for Critic...")
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

        if num_vlm_layers > 0:
            print(
                f"Reducing the number of VLM layers to {num_vlm_layers} for Critic..."
            )
            self.vlm.model.text_model.layers = self.vlm.model.text_model.layers[
                :num_vlm_layers
            ]

        if freeze_vision_encoder:
            self.vlm.model.vision_model.eval()
            for params in self.vlm.model.vision_model.parameters():
                params.requires_grad = False

        hidden_size = self.vlm.config.text_config.hidden_size

        self.state_proj = nn.Linear(max_state_dim, hidden_size).to(dtype=self.vlm.dtype)
        self.c51_head = nn.Linear(hidden_size, num_bins).to(dtype=self.vlm.dtype)

        self.fake_image_token = self.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.vlm.model.vision_model.eval()

    def embed_image(self, image: torch.Tensor):
        # We assume image has shape (batch_size, num_frames, channels, height, width)
        # or (batch_size, channels, height, width). We'll use the VLM's processor mechanism implicitly,
        # but to be safe, let's use the forward pass of the Vision Model.

        # However, it's easier and safer to use the exact same embedding path as the main SmolVLM.
        # So we just pass the pixel_values to the vision encoder.

        # If it's 5D (video/multi-image), we need to reshape for the connector
        if image.ndim == 5:
            bsize, n_frames, c, h, w = image.shape
            pixel_values = image.view(bsize * n_frames, c, h, w)
        else:
            pixel_values = image
            bsize = pixel_values.shape[0]
            n_frames = 1

        pixel_values = pixel_values.to(dtype=self.vlm.model.vision_model.dtype)

        # Get sequence from the vision encoder
        image_hidden_states = self.vlm.model.vision_model(
            pixel_values=pixel_values,
            patch_attention_mask=None,
        ).last_hidden_state
        # Modality projection & resampling
        image_hidden_states = self.vlm.model.connector(image_hidden_states)

        # Reshape back to handle frames if it was flattened
        if image.ndim == 5:
            # connector might change sequence length per image, e.g., via pixel_shuffle
            _, num_patches, dim = image_hidden_states.shape
            image_hidden_states = image_hidden_states.view(
                bsize, n_frames * num_patches, dim
            )

        return image_hidden_states

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.vlm.model.text_model.get_input_embeddings()(tokens)

    def embed_inputs(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embs = []
        pad_masks = []

        for img, img_mask in zip(images, img_masks, strict=False):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0],
                    dtype=torch.bool,
                    device=image_start_token.device,
                )
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.embed_image(img)
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(
                img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device
            )

            # Important: The vision model processor we invoked produces (B*Frames, C, H, W).
            # If we didn't unflatten it in `embed_image`, `img_emb` is (B*Frames, SeqLen, Dim).
            # To concatenate with `lang_emb` and `state_emb` (which are [B, SeqLen, Dim]),
            # we MUST ensure `img_emb` is unflattened back to [B, Frames*SeqLen, Dim].

            # Use original img.shape[0] as true batch size.
            actual_bsize = img.shape[0]
            if img_emb.shape[0] != actual_bsize:
                # It was flattened. Reshape it back.
                _, seq_l, dim = img_emb.shape
                # Assume batch dimension is first, frames is second in original img
                # So if img_emb is (B*F, S, D), we view it as (B, F*S, D)
                img_emb = img_emb.view(actual_bsize, -1, dim)

            emb_bsize, num_img_embs = img_emb.shape[:2]

            # The mask provided from the dataset is likely [B, 1] or similar
            img_mask = img_mask[:emb_bsize, None].expand(emb_bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            if self.add_image_special_tokens:
                image_end_token = (
                    self.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(emb_bsize, -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0],
                    dtype=torch.bool,
                    device=image_end_token.device,
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)

        lang_emb = self.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        if state is not None:
            state_emb = self.state_proj(state.to(dtype=self.vlm.dtype))
            state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
            embs.append(state_emb)
            bsize = state_emb.shape[0]
            states_seq_len = state_emb.shape[1]
            state_mask = torch.ones(
                bsize, states_seq_len, dtype=torch.bool, device=state_emb.device
            )
            pad_masks.append(state_mask)

        # Before concatenating, ensure all components have the same batch dimension.
        max_bsize = max(e.shape[0] for e in embs)
        for i in range(len(embs)):
            if embs[i].shape[0] < max_bsize:
                embs[i] = embs[i].expand(max_bsize, -1, -1)
                pad_masks[i] = pad_masks[i].expand(max_bsize, -1)

        embs = torch.cat(embs, dim=1).to(dtype=self.vlm.dtype)
        pad_masks = torch.cat(pad_masks, dim=1)

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)

        return embs, pad_masks

    def forward(self, images, img_masks, lang_tokens, lang_masks, state=None) -> Tensor:
        inputs_embeds, attention_mask = self.embed_inputs(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        # Pass through the VLM text model
        outputs = self.vlm.model.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state

        # Extract the representation of the last valid token for each sequence in the batch
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]

        last_token_hidden_states = hidden_states[
            torch.arange(batch_size, device=hidden_states.device), sequence_lengths
        ]

        last_token_hidden_states = last_token_hidden_states.to(dtype=self.vlm.dtype)

        logits = self.c51_head(last_token_hidden_states)
        return logits


def compute_c51_target_distribution(
    returns: torch.Tensor, num_bins: int = 51, vmin: float = -1.0, vmax: float = 0.0
) -> torch.Tensor:
    """
    Computes the C51 target categorical distribution.

    Args:
        returns (torch.Tensor): Expected returns (normalized between vmin and vmax). Shape: (batch_size,)
        num_bins (int): Number of bins.
        vmin (float): Minimum value of the support.
        vmax (float): Maximum value of the support.

    Returns:
        torch.Tensor: Target distribution. Shape: (batch_size, num_bins)
    """
    returns = torch.clamp(returns, vmin, vmax)

    batch_size = returns.shape[0]
    device = returns.device

    delta_z = (vmax - vmin) / (num_bins - 1)

    # Compute the projection of the target
    tz = returns

    # Compute bin indices
    b = (tz - vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    # Clamp bounds
    l = torch.clamp(l, 0, num_bins - 1)
    u = torch.clamp(u, 0, num_bins - 1)

    m = torch.zeros(batch_size, num_bins, dtype=torch.float32, device=device)

    # Distribute probability mass
    dl = u.float() - b
    du = b - l.float()

    # Fix exact match (when l == u)
    exact_match = l == u
    dl[exact_match] = 1.0
    du[exact_match] = 0.0

    offset = torch.arange(0, batch_size * num_bins, num_bins, device=device)

    m.view(-1).index_add_(0, l + offset, dl)
    m.view(-1).index_add_(0, u + offset, du)

    return m
