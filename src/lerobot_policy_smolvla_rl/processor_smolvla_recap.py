from typing import Any
import torch
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from .configuration_smolvla_recap import SmolVLARECAPConfig


def make_smolvla_recap_pre_post_processors(
    config: SmolVLARECAPConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    # Reuse the standard SmolVLA pre/post processors
    return make_smolvla_pre_post_processors(config, dataset_stats)
