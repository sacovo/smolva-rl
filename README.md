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

The pipeline is one critic → advantages → policy cycle, run twice: first on the expert demonstrations (pre-training), then on the policy's own rollouts (fine-tuning). Advantage conditioning is in place from the very first policy training step — the pre-trained policy already interprets the ± tokens before it ever sees rollout data.

**Round 1 — pre-training on expert demonstrations**

1. **Train the critic** (`train_critic.py`) — it bins normalized time-to-completion returns in `[-1, 0]` as a C51 distribution:

   ```bash
   python src/lerobot_policy_smolvla_rl/train_critic.py \
       --dataset_repo_id HuggingFaceVLA/libero \
       --steps 5000 \
       --state_dropout 0.2 \
       --end_weight 3.0 --end_threshold -0.1 \
       --save_dir outputs/checkpoints_critic
   ```

   `--state_dropout` randomly zeroes the proprioceptive state so the critic must judge progress from the images instead of shortcutting on the gripper state (keep it high); `--end_weight`/`--end_threshold` upweight the loss on end-of-episode frames.

2. **Compute advantages and thresholds** (`compute_thresholds.py`) — N-step TD advantages from the critic, plus the per-task labeling thresholds:

   ```bash
   python src/lerobot_policy_smolvla_rl/compute_thresholds.py \
       --dataset_repo_id HuggingFaceVLA/libero \
       --critic_checkpoint <critic.pt> \
       --save_dir outputs/recap_phase1
   ```

   This writes `task_advantages_<repo>.npy` and `task_thresholds_<repo>.json` into `--save_dir`. The threshold's target positive fraction matters: aggressive settings label necessary actions negative and degrade the policy (see the paper's ablation).

3. **Pre-train the policy** (`train_recap.py`) with KI, FAST co-training, and advantage conditioning, pointing `--save_dir` (or `--precomputed_advantages` / `--thresholds_path`) at the step-2 outputs — the trainer only consumes pre-computed advantages, it never runs the critic itself:

   ```bash
   python src/lerobot_policy_smolvla_rl/train_recap.py \
       --dataset_repo_id HuggingFaceVLA/libero \
       --save_dir outputs/recap_phase1 \
       --steps 250000
   ```

   (`--expert_mode` instead bypasses the critic entirely and labels every frame positive — plain advantage-free imitation, used for ablations.)

**Round 2 — fine-tuning on the policy's own rollouts**

4. **Collect rollouts** with `scripts/record_eval.py`; episode success is stored in `meta/episodes.parquet`.
5. **Fine-tune the critic** on the rollout data — same `train_critic.py` command, starting from the round-1 checkpoint via `--pretrained_critic_path`. This step is necessary: trained on successes only, the critic is over-optimistic on failed rollouts. Failed episodes receive a terminal penalty that pushes them into the lowest value bin.
6. **Recompute advantages and thresholds** on the rollout dataset with the fine-tuned critic (same `compute_thresholds.py` command, new `--save_dir`).
7. **Fine-tune the policy** from the round-1 checkpoint on the labeled rollouts; mixing expert demonstration batches back in (co-training) gives the best results and is required to fix long-horizon regressions:

   ```bash
   python src/lerobot_policy_smolvla_rl/train_recap.py \
       --dataset_repo_id <rollout_dataset> \
       --pretrained_policy_path <round1_checkpoint> \
       --save_dir outputs/recap_phase2 \
       --demo_dataset_repo_id HuggingFaceVLA/libero --demo_mix_ratio 0.5 \
       --steps 20000
   ```

   Optionally distill the result into a one-step policy with `train_snapflow.py`.

## Evaluation

`scripts/eval_cfg.sh` sweeps the classifier-free guidance weight on LIBERO suites; `scripts/compile_results.py` and `scripts/compare_eval_runs.py` aggregate results. `scripts/bench_jetson.py` measures action-chunk latency on a Jetson Orin Nano (the paper's 922 ms → 255 ms SnapFlow numbers).

## Paper

`paper/` contains the full LaTeX source; every figure regenerates from data committed in this repo (`paper/make_figures.py`, `paper/make_filmstrips.py`). CI builds the PDF and deploys the project page on every push to main (`.github/workflows/paper.yml`).
