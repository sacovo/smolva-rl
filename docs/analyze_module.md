# `analyze` Module

Path-free, tested library code plus a single CLI for analysing SmolVLA ReCap
training and checkpoints: wandb/CSV loss-curve statistics, dataset metadata
inspection, evaluation aggregation, and camera-/token-level input attribution.

Everything is reachable from one entry point:

```bash
python -m lerobot_policy_smolvla_rl.analyze <command> [options]
```

Outputs are written under `outputs/analysis/<name>/` (tables as CSV/JSON,
figures as PNG) and the destination path is printed at exit. Nothing is written
outside the repo, and no paths are hardcoded — inputs are always CLI arguments.

## Package layout

```
analyze/
├── __main__.py           # `python -m lerobot_policy_smolvla_rl.analyze`
├── cli.py                # click group; subcommands live next to their logic
├── runs.py               # wandb run / CSV loading → tidy DataFrame
├── loss_curves.py        # block stats, rolling means, run comparison
├── datasets.py           # dataset metadata inspection (shapes, stats, lengths)
├── eval_results.py       # eval_info.json aggregation
├── plotting.py           # shared mpl style, figure helpers, save conventions
└── attribution/
    ├── policy_io.py      # checkpoint loading, batch prep, prefix layout
    ├── ablation.py       # camera-level input ablation via img_masks
    ├── attention.py      # attention capture + action→image attribution
    ├── gradients.py      # grad×input / integrated gradients on image tokens
    └── report.py         # importance tables, saliency overlays, method agreement
```

## CLI reference

### Loss curves & wandb

| Command | Purpose |
|---|---|
| `loss-summary` | Block stats + initial/tail/SMA summary for one run in a CSV |
| `loss-compare` | Overlay raw trace + SMA for multiple runs and save a PNG |
| `wandb-runs` | List/filter runs in a project |
| `wandb-history` | Pull a run's metric history to CSV |

```bash
# Summary statistics for a single run column (substring match on the header)
python -m lerobot_policy_smolvla_rl.analyze loss-summary \
    --csv outputs/wandb_export.csv --run scratch --max-step 50000 --block 500

# Compare runs across CSVs; --step-offset aligns a resumed checkpoint and draws
# a "Resumed Checkpoint" marker.
python -m lerobot_policy_smolvla_rl.analyze loss-compare \
    --csv a.csv --run scratch --csv b.csv --run finetune --out comparison.png

# Read-only wandb API access
python -m lerobot_policy_smolvla_rl.analyze wandb-runs --project entity/project --dataset-contains libero
python -m lerobot_policy_smolvla_rl.analyze wandb-history --run entity/project/id --keys total_loss,ar_loss
```

`load_wandb_csv` owns the `Step * log_interval` alignment (default
`log_interval=10`) and resolves run columns by substring, so full
`"<run_name> - loss"` headers do not need to be typed exactly; `__MIN`/`__MAX`
columns are ignored unless explicitly requested.

### Datasets & evaluation

```bash
# Metadata only (downloads meta/info.json + meta/stats.json, no episodes):
python -m lerobot_policy_smolvla_rl.analyze dataset --repo-id HuggingFaceVLA/libero
# --full additionally loads the dataset and reports episode-length stats:
python -m lerobot_policy_smolvla_rl.analyze dataset --repo-id HuggingFaceVLA/libero --full

# Aggregate every outputs/eval/**/eval_info.json into one table; --pivot builds
# a success-rate matrix (last column pivots, the rest form the index):
python -m lerobot_policy_smolvla_rl.analyze eval-results --pivot checkpoint,suite
```

`eval-results` parses `key=value`-style fields (checkpoint, suite, cfg) out of
the job folder name, so it is no longer tied to a fixed checkpoint/suite/cfg
matrix. `scripts/compile_results.py` is now a thin caller of
`collect_eval_results`.

### Attribution (XAI)

All three attribution commands take `--checkpoint`, `--dataset-repo-id`,
`--episodes` (`0:50` or `0,1,2`), `--seed`, and `--device`, and write to
`outputs/analysis/attribution/<checkpoint-name>/<method>/`.

```bash
python -m lerobot_policy_smolvla_rl.analyze ablate     --checkpoint CKPT --dataset-repo-id DS --episodes 0:5 [--cameras k1,k2]
python -m lerobot_policy_smolvla_rl.analyze attention  --checkpoint CKPT --dataset-repo-id DS --episodes 0:5 [--rollout]
python -m lerobot_policy_smolvla_rl.analyze gradients  --checkpoint CKPT --dataset-repo-id DS --episodes 0:5 [--method gxi|ig] [--action-dims 0:3]
```

## Attribution methodology

The attribution code decides which camera images can be dropped, down-resolved,
or token-pruned. It builds on these architecture facts (verified against the
vendored `lerobot` 0.5.x code):

- The prefix layout is `[img tokens per camera | language | state]`
  (`modeling_smolvla.embed_prefix`). Each 512×512 camera image becomes 64 tokens
  (8×8 grid) via SigLIP + a pixel-shuffle connector.
- `img_masks` is a per-camera boolean mask already threaded through
  `embed_prefix` → attention, so camera ablation needs no model surgery.
- Attention is eager (`eager_attention_forward`, explicit softmax), so weights
  are capturable with a hook.

`PrefixLayout` (`policy_io.prefix_layout`) is the single source of truth mapping
*token index → camera/patch*. It resolves the image-token count dynamically,
accounts for image special tokens when enabled, and assumes the state projects
to a single token.

**Level 1 — ablation (ground truth).** For each frame, `run_camera_ablation`
runs the policy once with full input and once per camera with that camera's
`img_masks` entry set to `False`, reusing the **same seeded noise** so the
flow-matching comparison is not dominated by sampling variance. It reports, per
task, MSE of the predicted action chunk against ground truth (`action_mse`) and
against the full-input prediction (`policy_divergence_mse`).

**Level 2 — attention.** `AttentionRecorder` is a context manager that wraps the
eager attention path and stores softmax probabilities per layer while
`sample_actions` runs. `action_to_image_attention` renormalises the attention
mass from action-expert query tokens onto each camera's image tokens.
`--rollout` enables attention-rollout across layers as a secondary estimator;
plain per-layer mean is the default. Attention scores should be **validated
against ablation** (see `report.method_agreement`) before being trusted.

**Level 3 — gradients.** `grad_x_input_attribution` and
`integrated_gradients_attribution` take gradients of the expert's predicted
velocity field (sum of squared components, or a `--action-dims` slice such as
the gripper only) with respect to the **image-token embeddings** (output of
`embed_image`, 64 tokens/camera), not pixels. These paths run with grad enabled
in eval mode at small batch size.

**Reporting.** `report.py` provides `importance_table`,
`saliency_overlay` (upsamples the 8×8 score grid over the camera frame), and
`method_agreement` (Spearman rank correlation between methods, per task).

## Outputs

| Command(s) | Location |
|---|---|
| `loss-compare` | `outputs/analysis/loss-compare/<out>.png` |
| `wandb-history` | `outputs/analysis/wandb-history/<out>.csv` |
| `eval-results` | `outputs/eval/sweep_summary.csv` (+ `_pivoted.csv` with `--pivot`) |
| `ablate` / `attention` / `gradients` | `outputs/analysis/attribution/<ckpt>/<method>/{table.csv, overlays/*.png}` |

## Testing & verification

`tests/test_analyze.py` runs CPU-only and covers CSV loading/step-alignment,
loss-curve stats, mocked wandb/HF access, eval aggregation, `PrefixLayout`
resolution, the `AttentionRecorder` and gradient paths on toy modules, and the
camera-ablation loop (asserting `img_masks` is flipped per camera and noise is
reused).

```bash
pytest tests/test_analyze.py
```

Real-checkpoint smoke tests need a GPU and are not part of CI. See
[`src/lerobot_policy_smolvla_rl/analyze/README.md`](../src/lerobot_policy_smolvla_rl/analyze/README.md)
for a manual verification script template.

## Scope & limitations

- Read-only wandb usage; missing runs raise `RunNotFoundError` with the searched
  scope in the message.
- Attribution operates on trained checkpoints only — no training-loop changes.
- This module produces the *measurements* that a later token-pruning
  implementation would consume; it does not perform online/runtime pruning.
- `visualize_critic.py` remains standalone for now; relocating it into
  `analyze/critic.py` is a possible follow-up.
