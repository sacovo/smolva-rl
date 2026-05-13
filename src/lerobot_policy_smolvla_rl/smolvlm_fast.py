import torch
from torch import nn
from transformers import AutoModelForImageTextToText, AutoTokenizer
from .smolvla_fast import SmolVLAFast

class SmolVLMFast(SmolVLAFast):
    """Alias for SmolVLAFast with potentially different defaults or layer pruning."""
    
    def __init__(
        self, 
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", 
        num_fast_tokens: int = 1024,
        num_vlm_layers: int = -1,
        **kwargs
    ) -> None:
        super().__init__(model_id=model_id, num_fast_tokens=num_fast_tokens, **kwargs)
        
        if num_vlm_layers > 0:
            # Prune layers if requested (experimental)
            if hasattr(self.vlm.model.text_model, 'layers'):
                self.vlm.model.text_model.layers = self.vlm.model.text_model.layers[:num_vlm_layers]
