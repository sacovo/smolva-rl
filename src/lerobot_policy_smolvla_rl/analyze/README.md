# `analyze` Module

This module provides path-free, tested library code and a single CLI entry point for run history comparison, loss curve statistics, dataset metadata inspection, evaluation results aggregation, and camera-level/token-level attribution for trained SmolVLA ReCap checkpoints.

## CLI Usage

Run any command using:
```bash
python -m lerobot_policy_smolvla_rl.analyze <command>
```

Available commands:

### 1. Loss Analysis & Comparison
* **Summary statistics (block stats, initial/tail/SMA means):**
  ```bash
  python -m lerobot_policy_smolvla_rl.analyze loss-summary --csv outputs/wandb_export_*.csv --run scratch
  ```
* **Plotting comparisons:**
  ```bash
  python -m lerobot_policy_smolvla_rl.analyze loss-compare --csv outputs/wandb_export_1.csv --run scratch --csv outputs/wandb_export_2.csv --run finetune --out comparison.png
  ```

### 2. WandB API Utilities
* **Find and filter runs in a project:**
  ```bash
  python -m lerobot_policy_smolvla_rl.analyze wandb-runs --project entity/project --dataset-contains libero
  ```
* **Pull run metric history directly:**
  ```bash
  python -m lerobot_policy_smolvla_rl.analyze wandb-history --run entity/project/id --keys total_loss,ar_loss --out history.csv
  ```

### 3. Dataset & Eval Metadata
* **Dataset shapes and statistics:**
  ```bash
  python -m lerobot_policy_smolvla_rl.analyze dataset --repo-id HuggingFaceVLA/libero --full
  ```
* **Compile eval success rates:**
  ```bash
  python -m lerobot_policy_smolvla_rl.analyze eval-results --eval-dir outputs/eval --pivot checkpoint,suite
  ```

### 4. Saliency & Token Attribution (XAI)
* **Camera ablation:**
  ```bash
  python -m lerobot_policy_smolvla_rl.analyze ablate --checkpoint outputs/recap_libero_exported --dataset-repo-id HuggingFaceVLA/libero --episodes 0:5
  ```
* **Attention capture:**
  ```bash
  python -m lerobot_policy_smolvla_rl.analyze attention --checkpoint outputs/recap_libero_exported --dataset-repo-id HuggingFaceVLA/libero --episodes 0:5 --rollout
  ```
* **Gradient attribution (Grad*Input / Integrated Gradients):**
  ```bash
  python -m lerobot_policy_smolvla_rl.analyze gradients --checkpoint outputs/recap_libero_exported --dataset-repo-id HuggingFaceVLA/libero --method ig --episodes 0:5
  ```

---

## Manual Smoke Verification / GPU Tests

Since real-checkpoint evaluations and backpropagation through attention require a GPU, these are not run in CI. You can manually verify the attribution pipeline using the script template below:

### `scripts/verify_attribution.py`
```python
import torch
from pathlib import Path
from lerobot_policy_smolvla_rl.analyze.attribution.policy_io import load_recap_policy, prefix_layout
from lerobot_policy_smolvla_rl.analyze.attribution.ablation import run_camera_ablation
from lerobot_policy_smolvla_rl.analyze.attribution.attention import AttentionRecorder, action_to_image_attention
from lerobot_policy_smolvla_rl.analyze.attribution.gradients import grad_x_input_attribution

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = Path("outputs/recap_libero_exported")
dataset_repo_id = "HuggingFaceVLA/libero"

if not checkpoint.exists():
    print(f"Skipping verification: checkpoint {checkpoint} not found.")
    exit(0)

print(f"Loading policy on {device}...")
policy = load_recap_policy(checkpoint, device)
layout = prefix_layout(policy)

print("\n1. Testing Ablation...")
ab_res = run_camera_ablation(policy, dataset_repo_id, episodes=[0])
print(ab_res.per_task)

print("\n2. Testing Attention...")
# Iterate batch and record attention during the denoising (sample_actions) pass
from lerobot_policy_smolvla_rl.analyze.attribution.policy_io import iter_eval_batches
batch_dict = next(iter_eval_batches(dataset_repo_id, policy, episodes=[0], batch_size=1))
images, img_masks = policy.prepare_images(batch_dict["batch"])
state = policy.prepare_state(batch_dict["batch"])
lang_tokens = batch_dict["recap_batch"]["observation.language.tokens"]
lang_masks = batch_dict["recap_batch"]["observation.language.attention_mask"]
with AttentionRecorder(policy) as recorder:
    with torch.no_grad():
        policy.model.sample_actions(
            images=images, img_masks=img_masks,
            lang_tokens=lang_tokens, lang_masks=lang_masks, state=state,
        )
attns = action_to_image_attention(recorder, layout)
for cam, val in attns.items():
    print(f"  {cam}: {val.mean().item():.4f}")

print("\n3. Testing Gradients...")
gxi = grad_x_input_attribution(policy, batch_dict, layout)
for cam, val in gxi.items():
    print(f"  {cam} (mean grad*input): {val.mean().item():.4f}")

print("\nAll verification checks passed successfully!")
```
