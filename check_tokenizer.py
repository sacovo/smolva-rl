from transformers import AutoTokenizer

model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(f"Vocab size: {len(tokenizer)}")
print(f"Special tokens: {tokenizer.all_special_tokens}")
print(f"Special token IDs: {tokenizer.all_special_ids}")
