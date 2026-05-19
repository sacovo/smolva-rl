from dataclasses import dataclass
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import LeRobotDataset
from lerobot.policies import SmolVLAConfig

from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
import torch
from torch import nn
import torch.nn.functional as F

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
    state_dropout: float = 0.0


class SmolVLACrictic(modeling_smolvla.VLAFlowMatching):
    def __init__(self, config: SmolVLMCriticConfig):
        config.compile_model = False  # maybe implement this actually
        super().__init__(config)
        self.config = config

        del self.action_in_proj
        del self.action_out_proj

        hidden_size = self.vlm_with_expert.vlm.config.text_config.hidden_size
        # hidden_size = 960

        self.c51_head = nn.Sequential(
            nn.Linear(hidden_size, config.num_bins).to(
                dtype=self.vlm_with_expert.vlm.dtype
            ),
        )

    def _prepare_batch(self, batch: dict[str, torch.Tensor]):

        images, img_masks = modeling_smolvla.SmolVLAPolicy.prepare_images(self, batch)
        state = modeling_smolvla.SmolVLAPolicy.prepare_state(self, batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        return images, img_masks, state, lang_tokens, lang_masks

    def forward(self, batch: dict[str, torch.Tensor]):
        images, img_masks, state, lang_tokens, lang_masks = self._prepare_batch(batch)

        if self.training and self.config.state_dropout > 0:
            dropout_mask = (
                torch.rand(state.shape[0], 1, device=state.device)
                > self.config.state_dropout
            ).to(state.dtype)
            state = state * dropout_mask

        return self._forward(images, img_masks, lang_tokens, lang_masks, state)

    def _forward(self, images, img_masks, lang_tokens, lang_masks, state):

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        prefix_att_2d_masks = modeling_smolvla.make_att_2d_masks(
            prefix_pad_masks, prefix_att_masks
        )
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

        state_out = prefix_out[:, -1:]

        logits = self.c51_head(state_out).squeeze(1)  # [B, num_bins]

        distribution = torch.softmax(logits, dim=-1)

        return logits, distribution

    def compute_loss(self, batch, time_to_completion, weights=None):
        """
        Compute cross-entropy loss for value function training.
        """
        logits, _ = self.forward(batch)

        targets = torch.clamp(time_to_completion, 0, self.config.num_bins - 1)

        if weights is not None:
            loss = F.cross_entropy(logits, targets, reduction="none")
            return (loss * weights).mean()

        return F.cross_entropy(logits, targets)

    def get_pre_processor(self, dataset: LeRobotDataset):
        pre, _ = make_smolvla_pre_post_processors(
            self.config,
            dataset.meta.stats,
        )
        return pre
