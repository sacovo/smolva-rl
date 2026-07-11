# SmolVLA-RL: Critic-Guided Rollout Post-Training for a Compact VLA

RECAP-style reinforcement post-training for [SmolVLA](https://arxiv.org/abs/2506.01844) (450M), evaluated on the LIBERO benchmark: a distributional critic scores the policy's own rollouts, per-frame advantages are thresholded into positive/negative tokens, and the policy is retrained with advantage conditioning so classifier-free guidance can steer it at inference. Pre-training uses knowledge insulation (KI) with FAST-token co-training of the VLM backbone; SnapFlow one-step distillation removes the iterative flow-matching solve for real-time inference on edge hardware (Jetson Orin Nano).

Best configuration (critic-labeled rollouts co-trained with expert demonstrations): **67.8%** average success across three LIBERO suites, vs. 62.9% for the base policy and 52.4% for plain SmolVLA — with rollouts collected fully autonomously, no human interventions.

**Project page:** https://sacovo.github.io/smolva-rl/ · **Paper:** [`paper/`](paper/) (PDF built by CI, [download](https://sacovo.github.io/smolva-rl/paper.pdf)) · **Findings:** [`docs/recap_findings_overview.md`](docs/recap_findings_overview.md)

## Repository layout

| Path | Contents |
| --- | --- |
| `src/lerobot_policy_smolvla_rl/` | Policy (`modeling_smolvla_recap.py`), critic (`smolvla_critic.py`), FAST co-training (`smolvla_fast.py`), training entry points (`train_recap.py`, `train_critic.py`, `train_snapflow.py`), advantage tooling (`compute_thresholds.py`) |
| `scripts/` | Rollout recording (`record_eval.py`), CFG evaluation sweeps (`eval_cfg.sh`), result compilation, Jetson latency benchmark (`bench_jetson.py`), SLURM submission scripts |
| `paper/` | IEEE draft, figure scripts, and figure data — see [`paper/README.md`](paper/README.md) |
| `page/` | Project page deployed to GitHub Pages by CI |
| `analysis_videos/` | Evaluation episodes used in the paper figures and on the project page |
| `docs/` | Method notes, experiment plans, and findings write-ups |
| `tests/` | Pytest suite |

## Setup

Python ≥ 3.12, managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra libero   # lerobot[smolvla] + LIBERO simulation
```

A CUDA Docker image is built from `docker/Dockerfile` and published to `ghcr.io/sacovo/smolva-rl` on every push to main. For cluster training (multi-GPU, checkpoint/resume, config-file submission), see [`README_SLURM.md`](README_SLURM.md).

## Training pipeline

The RECAP recipe runs in three stages:

1. **Pre-training.** Train the critic on expert demonstrations (`train_critic.py`) — it bins normalized time-to-completion returns in `[-1, 0]` as a C51 distribution:

   ```bash
   python src/lerobot_policy_smolvla_rl/train_critic.py \
       --dataset_repo_id HuggingFaceVLA/libero \
       --steps 5000 \
       --state_dropout 0.2 \
       --end_weight 3.0 --end_threshold -0.1 \
       --save_dir outputs/checkpoints_critic
   ```

   `--state_dropout` randomly zeroes the proprioceptive state so the critic must judge progress from the images instead of shortcutting on the gripper state (keep it high); `--end_weight`/`--end_threshold` upweight the loss on end-of-episode frames.

   Then train the policy with KI, FAST co-training, and advantage conditioning (`train_recap.py`). With `--expert_mode`, the critic is bypassed and every demonstration frame is labeled positive:

   ```bash
   python src/lerobot_policy_smolvla_rl/train_recap.py \
       --dataset_repo_id <dataset_id> \
       --expert_mode \
       --steps 100000
   ```

2. **Rollout collection and critic fine-tuning.** Collect policy rollouts with `scripts/record_eval.py` (episode success is stored in `meta/episodes.parquet`). Fine-tune the critic on the rollout data with the same `train_critic.py` command, starting from the pre-trained checkpoint via `--pretrained_critic_path`; failed episodes receive a terminal penalty that pushes them into the lowest value bin. Then pre-compute N-step TD advantages and per-task labeling thresholds:

   ```bash
   python src/lerobot_policy_smolvla_rl/compute_thresholds.py \
       --dataset_repo_id <dataset_id> \
       --critic_checkpoint <critic.pt> \
       --save_dir outputs/recap_phase1
   ```

   This writes `task_advantages_<repo>.npy` and `task_thresholds_<repo>.json` into `--save_dir`. The threshold's target positive fraction matters: aggressive settings label necessary actions negative and degrade the policy (see the paper's ablation).

3. **Advantage-conditioned fine-tuning.** Run `train_recap.py` without `--expert_mode`, pointing `--save_dir` (or `--precomputed_advantages` / `--thresholds_path`) at the stage-2 outputs. The trainer only consumes pre-computed advantages — it never runs the critic itself. Mixing expert demonstration batches into this fine-tune (co-training) gives the best results and is required to fix long-horizon regressions. Optionally distill the result into a one-step policy with `train_snapflow.py`.

## Evaluation

`scripts/eval_cfg.sh` sweeps the classifier-free guidance weight on LIBERO suites; `scripts/compile_results.py` and `scripts/compare_eval_runs.py` aggregate results. `scripts/bench_jetson.py` measures action-chunk latency on a Jetson Orin Nano (the paper's 922 ms → 255 ms SnapFlow numbers).

## Paper

`paper/` contains the full LaTeX source; every figure regenerates from data committed in this repo (`paper/make_figures.py`, `paper/make_filmstrips.py`). CI builds the PDF and deploys the project page on every push to main (`.github/workflows/paper.yml`).
