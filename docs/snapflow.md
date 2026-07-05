# SnapFlow: One-Step Distillation of the RECAP Action Expert

SnapFlow distills the multi-step Flow-Matching action expert of a trained
**RECAP + Knowledge-Insulation** checkpoint into a **single-step** generator,
so inference no longer runs an iterative ODE solve over the action chunk. It
follows the shortcut/consistency distillation idea of SnapFlow (Luan et al.,
arXiv:2604.05656), adapted here to condition on a *target time* and to run on
top of the frozen SmolVLA VLM backbone.

The method text below doubles as the paper's SnapFlow section.

## What it is

A standard Flow-Matching policy generates an action chunk by integrating the
learned velocity field from `t=1` (noise) to `t=0` (clean action) over many
Euler steps. SnapFlow trains the expert to make the **whole jump `t=1 → s=0` in
one forward pass**, by teaching a *target-time-conditioned* velocity field to
agree with a short multi-step "teacher" trajectory of the same (frozen-VLM)
model.

Two ingredients make this work:

1. **Target-time conditioning.** The suffix embedding is extended with a
   `target_time` `s` in addition to the flow time `t`
   (`embed_suffix(..., target_time=s)`,
   `modeling_smolvla_recap.py:319`). The velocity the expert predicts is
   interpreted as "the average velocity that carries `x_t` from time `t` to
   time `s`". At `s = t` this reduces to the ordinary instantaneous
   Flow-Matching velocity; at `s = 0` it is the one-step jump used at
   inference.
2. **Zero-initialized `target_time_mlp`.** The MLP that injects `s` is
   initialized to all-zeros (both layers' weights *and* biases,
   `modeling_smolvla_recap.py:88-97`). At the start of distillation the target
   time therefore contributes nothing, so the model is exactly the pre-trained
   FM expert and training starts from a known-good point.

## The loss (`compute_loss_snapflow`)

Implemented in `modeling_smolvla_recap.py:461-616`. Advantage conditioning is
active during distillation and **always uses the positive advantage token**
(`<advantage_positive>`), so the distilled one-step model reproduces the
"good-action" branch of the policy. All VLM parameters are frozen for the
duration of the forward/backward pass (complete Knowledge Insulation), so the
continuous losses only update the expert and projections.

The total loss mixes two components:

```
loss = alpha * fm_loss + (1 - alpha) * lambda_consistency * consistency_loss
```

**1. Flow-Matching component (`fm_loss`, keeps the base velocity field intact).**
Standard flow matching with `s = t`:

```
tau            ~ U(0, 1)
omega          ~ N(0, I)
x_tau          = tau * omega + (1 - tau) * action     # tau=0 -> clean, tau=1 -> noise
target_flow    = omega - action
fm_loss        = MSE( f(x_tau, t=tau, s=tau), target_flow )
```

**2. Consistency component (`consistency_loss`, the actual distillation).**
A two-step-Euler "teacher" trajectory from pure noise `x_1` is computed under
`torch.no_grad()` (the teacher is the same model, not a separate network) and
the one-step "student" must match its averaged velocity:

```
x_1        ~ N(0, I)
# --- teacher (no_grad, two half-steps of Euler) ---
v_1        = clamp( f(x_1,    t=1,   s=1),   ±clamp )
x_half     = x_1 - 0.5 * v_1
v_half     = clamp( f(x_half, t=0.5, s=0.5), ±clamp )
v_shortcut = 0.5 * (v_1 + v_half)
# --- student (one full jump) ---
v_student  = f(x_1, t=1, s=0)
consistency_loss = MSE( v_student, v_shortcut )
```

The `clamp` (default ±20) bounds the teacher velocities so a poorly-scaled
prediction early in training cannot blow up the shortcut target. When
`config.loss_n_action_steps > 0`, both losses are computed only over the first
`n` action steps.

## What is trainable

The VLM backbone is fully frozen; only the action expert and its projections
train (`train_snapflow.py:255-269`):

- `target_time_mlp` (the new `s` conditioning)
- `vlm_with_expert.lm_expert` (the action expert transformer)
- `action_in_proj`, `action_out_proj`
- `action_time_mlp_in`, `action_time_mlp_out`

## CLI

Distillation is launched with `src/lerobot_policy_smolvla_rl/train_snapflow.py`.
`--recap_checkpoint` and `--dataset_repo_id` are **required**; the checkpoint
must be a trained RECAP+KI model.

SnapFlow-specific flags (defaults in parentheses):

| Flag | Default | Meaning |
| :-- | :-- | :-- |
| `--recap_checkpoint` | *required* | Trained RECAP+KI checkpoint to distill. |
| `--alpha` | `0.5` | FM / consistency mixing ratio (`snapflow_alpha`). |
| `--lambda_consistency` | `0.1` | Consistency-loss weight (`snapflow_lambda`). |
| `--clamp` | `20.0` | Teacher velocity clamp range (`snapflow_clamp`). |

Common training flags (shared with the other trainers): `--steps` (30000),
`--batch_size` (4), `--accumulation_steps` (4), `--lr` (2.5e-5), `--min_lr`
(2.5e-7), `--warmup_steps` (500), `--max_grad_norm` (1.0), `--save_dir`
(`outputs/snapflow`), plus the standard dataloader flags. The LR follows the
shared linear-warmup + cosine-to-`min_lr` schedule
(`train_common.build_warmup_cosine_scheduler`).

Example:

```bash
PYTHONPATH=src uv run accelerate launch \
  src/lerobot_policy_smolvla_rl/train_snapflow.py \
  --recap_checkpoint outputs/recap_libero_final \
  --dataset_repo_id sancov/smolvla_recap_libero_spatial \
  --steps 30000 --batch_size 4 --accumulation_steps 4 \
  --alpha 0.5 --lambda_consistency 0.1 --clamp 20.0
```

On Slurm, use `scripts/submit_snapflow.py` / `scripts/submit_snapflow.sh`
(see `README_SLURM.md`).

## Export and inference

The distilled behavior is gated entirely by `snapflow_enabled` in the exported
policy config (`configuration_smolvla_recap.py:30-33`, alongside
`snapflow_alpha`, `snapflow_lambda`, `snapflow_clamp`). When
`snapflow_enabled=true`, `sample_actions` takes the **single-step branch**
(`modeling_smolvla_recap.py:829`): it caches the VLM prefix once and performs
one denoise step `t=1 → s=0` instead of the iterative solver. Advantage
conditioning and Classifier-Free Guidance still apply (see
`docs/policy_evaluation_and_cfg.md`) — with `cfg_weight != 0, 1` the batch is
duplicated into `[uncond, cond]` and blended.

## Evaluation

`scripts/eval_snapflow.sh` runs LIBERO-spatial eval of an exported SnapFlow
policy at `cfg_weight=0.0` (no CFG) and `cfg_weight=1.5` (with CFG), then
summarizes with `scripts/compare_eval_runs.py`. See
`docs/policy_evaluation_and_cfg.md` for the CFG sweep methodology.
