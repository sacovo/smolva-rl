from dataclasses import dataclass
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("smolvla_recap")
@dataclass
class SmolVLARECAPConfig(SmolVLAConfig):
    num_fast_tokens: int = 1024
    use_advantage_conditioning: bool = True
    model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    action_stats: dict[str, list[float]] | None = None
    # Probability of dropping advantage conditioning per batch during training.
    # Enables classifier-free guidance at inference. RECAP paper uses 0.3.
    adv_dropout_rate: float = 0.3
    # Classifier-free guidance weight for inference.
    # 1.0 = no guidance, >1.0 = amplify advantage-conditioned direction.
    cfg_weight: float = 1.0
    # Loss weights for the combined objective.
    ar_loss_weight: float = 1.0
    fm_loss_weight: float = 1.0
