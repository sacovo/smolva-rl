# Knowledge Insulation experiment, LIBERO eval, and the RECAP rollout iteration

Status: **in progress** (2026-07-06). Log of the KI ablation, the LIBERO
evaluation and what it told us, a review of the advantage-conditioning
implementation, and the RECAP rollout-iteration pipeline now underway.

Companion to `docs/smolvla_rl_methods.md` (method), `docs/efficientvla.md`,
`docs/snapflow.md`. Raw eval artifacts: `outputs/eval_comparison_cfg1.csv`,
`outputs/eval_downloaded/`.

---

## 1. The `knowledge_insulation` toggle (feature)

Motivation: mirror the SmolVLA-paper training setup — **no AR/FAST loss and the
VLM trained by flow matching** — rather than RECAP's default (VLM trained only
by the AR/FAST loss, insulated from the FM loss by KI).

Added `SmolVLARECAPConfig.knowledge_insulation: bool = True` (default preserves
RECAP+KI). `train_recap --disable_ki` sets it False. When disabled
(`modeling_smolvla_recap.compute_loss`):

- `_apply_ki_patch` (the self-attention `no_grad` wrapper) is **not** installed;
- the VLM prefix is **not** detached and VLM params are **not** frozen during
  the FM forward → the FM loss trains the VLM backbone.
- The AR/FAST forward is **skipped entirely** when `ar_loss_weight == 0`
  (avoids a wasted full VLM pass). The vision encoder stays frozen either way
  (`freeze_vision_encoder=True`).

Verified on GPU (`ar_loss_weight=0`): KI on → 0/148 VLM-language params get
gradient (insulated); KI off → 140/148 get FM gradient (trained). `fm_loss` is
identical either way, i.e. only the backward graph changes.

Commit: `7b51d50` (feat: knowledge_insulation toggle to train the VLM via flow
matching).

---

## 2. Trained models

All on LIBERO (`HuggingFaceVLA/libero`), SmolVLM2 backbone truncated to 16
layers, 2×H200. Configs read from each `recap_model/migrated/config.json`.

| Model (dir) | adv-cond | KI | AR loss | notes |
|---|---|---|---|---|
| `libero_recap_250000` / `libero/recap_model` | **on** (critic-based, real ±) | on | on | the full RECAP+KI model (250k / 350k) |
| `libero_expert` | on (all-positive, `expert_mode`) | on | on | phase-2 imitation baseline |
| `libero_expert_no_ki` | **off** | on | on | ⚠️ name is misleading — KI is **on** per its config (predates the `--disable_ki` flag) |
| `libero_smolvla_noki` (wandb `kloi9fle`) | off | **off** | **0** | the genuine SmolVLA-paper-style run: FM trains the VLM, 250k steps |

> Caution: `libero_expert_no_ki` is *not* a KI-off model. The only genuine
> KI-off run is `libero_smolvla_noki`. The two "expert" models differ only in
> advantage conditioning (on vs off), both KI-on + AR-on.

---

## 3. LIBERO evaluation (cfg=1.0, n=50/suite)

We first ran a full CFG sweep, then realized **CFG is meaningless for these
models**: `expert_no_ki`/`smolvla_noki` have `use_advantage_conditioning=False`
so `cfg_weight` is ignored entirely, and `expert` was trained all-positive
(`expert_mode`) so the guidance direction is ≈ null. Reduced to `cfg=1.0`.

| Model | spatial | object | goal | libero_10 | **Avg\*** |
|---|---:|---:|---:|---:|---:|
| recap+ki 250k | 84 | 0 | 68 | 34 | **62.0** |
| recap+ki 350k final | 72 | 0 | 72 | 38 | **60.7** |
| expert | 70 | 0 | 76 | 40 | **62.0** |
| expert_no_ki | 86 | 0 | 72 | 38 | **65.3** |
| smolvla_noki (real KI-off) | 62 | 0 | 58 | 36 | **52.0** |

\* Avg over the three working suites (excludes `libero_object`).

- **`libero_object` = 0% for every model**, including the full RECAP+KI
  baseline → a systematic eval failure on that suite (rendering/normalization),
  not model quality. All averages exclude it. **TODO: investigate.**
- The genuine SmolVLA-paper run (`smolvla_noki`, KI off + AR off) is the
  **weakest** (52.0), lower on all three working suites. Disabling KI + dropping
  AR co-training **hurt** on LIBERO.
- KI+AR variants cluster ~60–65. The full model's 250k checkpoint (62.0)
  slightly beats its 350k "final" (60.7) — mild degradation past 250k.
- Caveat: n=50/suite is noisy (~±7%); only `smolvla_noki` being *consistently*
  lowest is a robust signal.

### Successful-episode length (execution speed)

Measured from recorded-video frame counts (frames = env steps to completion).

| Model | median | spatial | goal | libero_10 |
|---|---:|---:|---:|---:|
| recap+ki 250k | 110 | 118 | 110 | 236 |
| recap+ki 350k | 120 | 117 | 106 | 283 |
| expert | 113 | 108 | 113 | 262 |
| expert_no_ki | 117 | 114 | 116 | 316 |
| smolvla_noki | 112 | 109 | 119 | 273 |

On the short-horizon suites everything sits at **~107–119 steps median** —
within noise. **No policy completes tasks meaningfully faster.** KI/advantage
conditioning affect success rate (marginally), not execution speed.

---

## 4. Why advantage conditioning underwhelms (implementation review)

We traced the advantage path end-to-end. **Nothing is backward / inverted:**

| Stage | Code | Verdict |
|---|---|---|
| Advantage sign | `A = R_t − V(s_t)` (higher = better than expected) | ✅ |
| Threshold | 70th pct of successful advantages → top 30% positive | ✅ |
| Label | `advantage_bool = advantage > threshold` → positive | ✅ |
| Token | `adv_pos_id if bool else adv_neg_id`; distinct special tokens, embeddings resized + norm-matched | ✅ |
| CFG | `uncond + w·(cond − uncond)` (w>1 amplifies positive conditional) | ✅ |

Two structural reasons it has little effect on LIBERO — **not bugs**:

1. **Data.** LIBERO is expert-demo-only; even "below-threshold" frames are
   still expert actions. There is no genuinely *bad* behavior to contrast
   against, so the positive/negative signal is inherently weak. RECAP advantage
   conditioning is designed for **mixed-quality** (RL rollout) data.
2. **KI insulates the advantage token from the action objective.** The
   advantage token lives in the prefix; under KI the FM forward detaches the
   prefix and freezes the VLM, so the FM loss (which trains the deployed
   continuous action expert) puts **zero gradient** on the advantage-token
   embedding. It is trained *only* by the AR/FAST loss and reaches the action
   expert only indirectly via VLM hidden states. So KI (which helps overall)
   simultaneously dampens advantage conditioning's effect on the actual actions.

**Takeaway:** on this data/complexity, KI is the lever that helps (cleaner
training signal for the VLM); advantage conditioning barely moves the needle —
expected, because expert demos lack the good/bad contrast it needs.

---

## 5. Next step — RECAP rollout iteration (in progress)

To give advantage conditioning something real to condition on, generate
**messy** (mixed success/failure) rollout data with the best model and retrain,
per the paper's rollout phase.

Decisions: ~600 rollout episodes (20/task) over spatial/goal/libero_10
(`libero_object` excluded — it is broken); finetune the existing critic
(warm start); policy retrain on a rollout+demo mix.

Pipeline:

1. **Rollout collection** — `scripts/record_eval.py --headless` runs the 250k
   RECAP policy (`outputs/libero_recap_250000`, `cfg_weight=1.0`) in LIBERO and
   records per-episode success/failure into a LeRobotDataset. Launched via
   `scripts/submit_rollout.sh` (new sbatch wrapper), one job per suite →
   `outputs/rollout_{spatial,goal,libero10}`. Canary (10 ep) validated the loop:
   7 success / 3 failure, successes 76–133 steps, failures time out at 280.
2. **Merge** — `lerobot.datasets.aggregate.aggregate_datasets` → one dataset
   `outputs/rollout_merged`.
3. **Critic finetune** — `train_critic.py` warm-started from
   `outputs/libero/critic/checkpoint_final.pt` on the merged rollout data;
   failed episodes get the `C_FAIL` penalty, so the critic finally learns to
   separate good vs bad states.
4. **Advantages** — `compute_thresholds.py` → per-frame advantages + per-task
   thresholds from the new critic.
5. **Policy retrain** — `train_recap.py --precomputed_advantages` (NOT
   `--expert_mode`), finetuning from the 250k checkpoint on a rollout+demo mix
   with real ± advantage conditioning.

### Cluster environment notes (calculon)

The LIBERO eval / rollout jobs need, in the sbatch script:

- `source /opt/conda/etc/profile.d/conda.sh && conda activate ~/conda12`
  (PATH-only prepend is **not** enough — the env supplies EGL for headless
  rendering).
- `export MUJOCO_GL=egl`.
- `export HF_LEROBOT_HOME=...` (the deprecated `LEROBOT_HOME` now hard-errors).
- **Do not** set `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` — compute nodes have
  internet, and the flags block model-init HF fetches.
- `EGLError` messages in `__del__`/`eglDestroyContext` at teardown are harmless
  noise; check job exit state, not those.
- Do not run long compute on the login node.
