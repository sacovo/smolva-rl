from dataclasses import dataclass
import numpy as np
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("smolvla_recap")
@dataclass
class SmolVLARECAPConfig(SmolVLAConfig):
    num_fast_tokens: int = 1024
    use_advantage_conditioning: bool = True
    model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    action_stats: dict[str, np.ndarray] | None = None
