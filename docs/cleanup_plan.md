# Cleanup Plan

Status: planned (2026-07-04). Companion to `docs/paper_plan.md`.

Each item is small, isolated, and independently committable — one commit
per item (or per group where noted). Ordered by priority. Nothing here
changes training behavior except item 9 (a real bug fix).

## P1 — actively misleading / broken (do first, all trivial)

### 1. Fix wrong file path in the methods doc
- **File:** `docs/smolvla_rl_methods.md:56`
- **Change:** `smolvla_recap_bap.py` → `modeling_smolvla_recap.py`.
  While there, spot-check the code snippets in §2 against the current
  `compute_loss` and update line-level drift.
- **Verify:** every `src/...` path mentioned in the doc exists.

### 2. Fix the advantage-threshold percentile statement
- **Files:** `docs/smolvla_rl_methods.md:139`, `advantage_utils.py:112-146`
- **Change:** code computes the **70th** percentile
  (`positive_fraction=0.3` → top 30% positive); doc says 30th. Confirm 70th
  is the intent (it matches "top 30% of frames labeled positive"), then fix
  the doc sentence.
- **Also:** add a unit test `tests/test_advantage_utils.py` pinning the
  percentile behavior on a tiny synthetic advantage array (this is
  paper-method-critical and currently untested).

### 3. Remove machine-specific paths from committed eval scripts
- **Files:** `scripts/eval_cfg.sh:27`, `scripts/eval_snapflow.sh:35`,
  `docs/policy_evaluation_and_cfg.md:71-79`
- **Change:** both scripts call
  `/home/sandro/.gemini/antigravity/.../compare_eval_runs.py` (outside the
  repo, does not exist for anyone else). Either (a) copy that comparison
  logic into `scripts/compare_eval_runs.py`, or (b) drop the call and point
  at `scripts/compile_results.py` / the `analyze` module. Update the doc to
  match whichever is chosen.
- **Verify:** `grep -r "antigravity\|/home/sandro" scripts/ docs/` is clean.

### 4. Gitignore stray root artifact
- **Files:** `.gitignore`, `local_verification.json`
- **Change:** add `local_verification.json` (or `*_verification.json`) to
  `.gitignore`; delete the file if not needed.

## P2 — dead code removal

### 5. Delete unused `fast_tokenizer.py` and stop documenting it as live
- **Files:** `src/lerobot_policy_smolvla_rl/fast_tokenizer.py` (127 lines,
  imported by nothing), `docs/smolvla_rl_methods.md` §1 (lines ~22-45)
- **Change:** delete the module. Rewrite the doc's FAST section to describe
  what actually runs: the remote HF `UniversalActionProcessor` loaded via
  `AutoTokenizer` in `smolvla_fast.py:46-53`.
- **Verify:** `grep -r fast_tokenizer src tests scripts` is empty;
  `pytest tests/` passes.
- **Alternative:** if the local DCT/BPE implementation is wanted for the
  paper's reproducibility story, wire it in instead — but that is a
  feature, not cleanup; decide explicitly.

### 6. Remove unused imports / dead locals (single commit)
- `src/lerobot_policy_smolvla_rl/efficient_inference.py:2` (`torch.nn`)
- `scripts/submit_utils.py:2` (`argparse`)
- `scripts/compile_results.py:2-3` (`sys`, `os`)
- `scripts/record_eval.py:5,9,10,18,26,28` (several) + dead `headless`
  local at `:226`
- `scripts/cache_vision_embeddings.py:168` (dead `cache` local)
- `scripts/overfit_test.py:13` (shadowed import)
- `scripts/test_loss.py:1`, `scripts/verify_pipeline.py:6,12`
- **Verify:** re-run pyflakes/prospector; `pytest tests/`.

### 7. Delete stale scratch forks (local only, not committed)
- `scratch/run_array_task.py`, `scratch/compile_results.py` are diverged
  copies of the `scripts/` versions. Delete the scratch copies so there is
  one source of truth. (scratch/ is gitignored — hygiene only.)

## P3 — deduplication (behavior-preserving refactors)

### 8. Move `patch_lerobot_dataset_reader` into `dataloader_utils.py`
- **Files:** `train_recap.py:30-116`, `train_snapflow.py:30-116`
  (byte-identical ~87 lines), `dataloader_utils.py`
- **Change:** single shared function; both trainers import it.
- **Verify:** `diff` of the moved block against both originals before
  deleting; `pytest tests/test_dataloader_utils.py tests/test_snapflow.py`;
  a few hundred steps of `scripts/overfit_test.py` as a smoke test.

### 9. Shared cosine-warmup scheduler + fix critic `--min_lr` no-op
- **Files:** `train_recap.py:507-523`, `train_snapflow.py:367-376`,
  `train_critic.py:70,363`
- **Change:** extract the LambdaLR warmup+cosine+floor into a helper (e.g.
  in a new `train_common.py` or `dataloader_utils.py` sibling). Switch
  `train_critic.py` to it — it currently uses
  `diffusers.get_scheduler("cosine")`, which **silently ignores
  `--min_lr`** (latent bug; this item changes critic LR behavior late in
  training, intentionally).
- **Verify:** plot the three schedules for a dummy run of 1000 steps and
  compare against the previous recap/snapflow curves (must be identical);
  critic curve now floors at `min_lr`.
- Also fold in the trivial `import math` moves
  (`train_recap.py:510`, `train_snapflow.py:367` → module top).

### 10. (Optional, larger) `train_common.py` for trainer boilerplate
- Accelerator setup, checkpoint save/resume loop, wandb init are
  re-implemented in all three trainers. Worth doing only after the paper
  deadline — it touches resume logic that in-flight runs depend on.
  **Do not do this while `expert_libero` / `plain_smolvla_libero` are
  training with `--resume_from auto`.**

## P4 — documentation additions

### 11. Write the SnapFlow training/eval doc (doubles as paper method text)
- **New file:** `docs/snapflow.md`
- **Contents:** what it is (one-step distillation per Luan et al.,
  arXiv:2604.05656, adapted to the RECAP+KI checkpoint); the loss
  (`compute_loss_snapflow`, `modeling_smolvla_recap.py:454-609`: alpha-mixed
  FM + two-step-Euler shortcut consistency, clamp); the zero-init
  `target_time_mlp`; what is trainable (expert + projections only, VLM
  frozen); full CLI (`--recap_checkpoint` required, `--alpha`,
  `--lambda_consistency`, `--clamp`, defaults); export path
  (`snapflow_enabled=true` in exported config is what triggers the 1-step
  `sample_actions`); how to eval (`scripts/eval_snapflow.sh`).

### 12. Add SnapFlow to `README_SLURM.md`
- One §Scripts entry each for `scripts/submit_snapflow.py` /
  `submit_snapflow.sh`, mirroring the existing `submit_recap.py` entry.

### 13. Merge the two efficientvla docs and mark implemented
- **Files:** `docs/efficientvla_spec.md` (291 lines),
  `docs/efficientvla_implementation_plan.md` (380 lines) — ~70% overlap,
  both future-tense "draft" although C1/C2 are implemented and tested.
- **Change:** collapse into one `docs/efficientvla.md` with status
  "implemented (C1 layer pruning, C2 visual-token pruning; C3 deliberately
  dropped in favor of SnapFlow)"; fix broken cross-refs
  (`analyze_module_spec.md` → `analyze_module.md`; remove
  `research/snapflow_analysis.md`); add a short "how to run a pruned
  config" usage snippet (`pruned_layers`, `visual_tokens_keep`).

### 14. Short usage docs for the critic pipeline
- One page covering `train_critic.py`, `compute_thresholds.py`,
  `advantage_utils.py`, `visualize_critic.py`: the C51 head (201 bins,
  vmin/vmax = [-1, 0]), `C_FAIL`, and the phase-3 order of operations.
  Can be a new section in `docs/smolvla_rl_methods.md` instead of a new
  file.

## P5 — small quality items

### 15. Rename `SmolVLACrictic` → `SmolVLACritic`
- **Files:** `smolvla_critic.py:29`, `train_critic.py:29` (+ any other refs)
- **Caution:** check whether the class name is pickled inside existing
  critic checkpoints (`outputs/libero_critic_final.pt`) before renaming; if
  `torch.load` needs the old name, keep an alias
  `SmolVLACrictic = SmolVLACritic`.

### 16. pyproject.toml hygiene
- Replace placeholder `description = "Add your description here"`.
- Move `marimo[lsp]` and `weasyprint` out of core `dependencies` into a
  dev/docs optional group.
- Add pytest config: `[tool.pytest.ini_options] pythonpath = ["src", "."]`
  (fixes the implicit-rootdir dependence of
  `tests/test_submit_scripts.py:9`).

### 17. Name the visual-token magic number
- **File:** `modeling_smolvla_recap.py:1136` — hardcoded `64` tokens per
  camera in `(64 - config.visual_tokens_keep)`. Derive from the vision
  config or name it as a constant with a comment stating the SigLIP grid
  assumption.

## Suggested batching

- **Batch 1 (30 min, zero risk):** items 1, 2 (doc part), 3, 4, 7, 12, 16.
- **Batch 2 (dead code):** items 5, 6 + the test from item 2.
- **Batch 3 (refactors, after current training runs finish):** 8, 9, 15, 17.
- **Batch 4 (docs for the paper):** 11, 13, 14 — item 11 first, it feeds
  the paper's method section.
- Item 10 explicitly deferred until after the paper.
