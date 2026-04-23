from transformers import AutoModelForImageTextToText
from typing import Any

from torch import nn


class SmolVLMFast(nn.Module):
    """SmolVLM Backbone with FAST Tokens replacing the 1024 least used tokens

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self, model_id: str, num_vlm_layers: int = -1, num_fast_tokens: int = 1024
    ) -> None:
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype="bfloat16",
            low_cpu_mem_usage=True,
        )

        if num_vlm_layers > 0:
            self.num_vlm_layers = num_vlm_layers
        else:
            self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)

    def get_vlm_model(self):
        return self.vlm.model
