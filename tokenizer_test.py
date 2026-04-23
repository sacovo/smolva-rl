import numpy as np
from transformers import AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

# 1. Load the underlying text tokenizer in isolation
# This succeeds because it doesn't get confused by the processor's config files
bpe_tokenizer = AutoTokenizer.from_pretrained(
    "physical-intelligence/fast", trust_remote_code=True
)

# 2. Pass the loaded tokenizer directly into the processor
# We use get_class_from_dynamic_module because AutoProcessor.from_pretrained
# fails to load the bpe_tokenizer when it is not in a 'bpe_tokenizer' subfolder.
processor_class = get_class_from_dynamic_module(
    "processing_action_tokenizer.UniversalActionProcessor",
    "physical-intelligence/fast",
)
processor = processor_class(bpe_tokenizer=bpe_tokenizer)

# 3. Test your action chunks
# Ensure your dummy data is float32 and maximum 3 dimensions
action_data = np.random.rand(1, 50, 14).astype(np.float32)

print(f"Original Action: {action_data[:, :5, :]}")  # Print first 5 tokens for brevity
# Encode
tokens = processor(action_data)
print(f"Encoded Tokens (first 5): {tokens[0][:5]}")

# Decode
decoded_actions = processor.decode(tokens)
print(f"Decoded Actions Shape: {decoded_actions.shape}")

print(f"Decoded Action (first 5): {decoded_actions[:, :5, :]}")
