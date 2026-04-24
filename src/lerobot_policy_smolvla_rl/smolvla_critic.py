from dataclasses import dataclass
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import LeRobotDataset
from lerobot.policies import SmolVLAConfig

from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
import torch
from torch import nn

from lerobot.policies.smolvla import modeling_smolvla

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

C_FAIL = 10000



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



@PreTrainedConfig.register_subclass("smolvla_critic")
@dataclass
class SmolVLMCriticConfig(SmolVLAConfig):
    num_bins: int = 201
    vmin: float = -1.0
    vmax: float = 0.0


class SmolVLACrictic(modeling_smolvla.VLAFlowMatching):
    def __init__(self, config: SmolVLMCriticConfig):
        config.compile_model = False  # maybe implement this actually
        super().__init__(config)
        self.config = config

        del self.action_in_proj
        del self.action_out_proj

        hidden_size = self.vlm_with_expert.expert_hidden_size
        hidden_size = 960

        self.c51_head = nn.Sequential(
            nn.Linear(hidden_size, config.num_bins).to(
                dtype=self.vlm_with_expert.vlm.dtype
            ),
        )


    def _prepare_batch(
        self, batch: dict[str, torch.Tensor] 
    ):

        images, img_masks = modeling_smolvla.SmolVLAPolicy.prepare_images(self, batch)
        state = modeling_smolvla.SmolVLAPolicy.prepare_state(self, batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        return images, img_masks, state, lang_tokens, lang_masks


    def forward(self, batch: dict[str, torch.Tensor]):
        images, img_masks, state, lang_tokens, lang_masks = self._prepare_batch(batch)

        return self._forward(images, img_masks, lang_tokens, lang_masks, state)


    def _forward(self, images, img_masks, lang_tokens, lang_masks, state):
        """
        Evaluates the C51 target distribution based purely on the environment
        and language prefix, without action/timestep suffixes.
        """

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        prefix_att_2d_masks = modeling_smolvla.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=True,
        )

        prefix_out = outputs_embeds[0]
        prefix_out = prefix_out.to(dtype=torch.float32)

        # Slice the final token(s) to represent the aggregated state.
        # (Adjust this slice if you need a specific chunk_size or pooling method)
        state_out = prefix_out[:, -1:]

        # Pass through the new sequential head
        c51_logits = self.c51_head(state_out)

        # Compute the target distribution across the bins
        c51_distribution = torch.softmax(c51_logits, dim=-1)

        return c51_logits, c51_distribution

    def get_pre_processor(self, dataset: LeRobotDataset):
        pre, _ = make_smolvla_pre_post_processors(
            self.config,
            dataset.meta.stats,
        )
        return pre


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
