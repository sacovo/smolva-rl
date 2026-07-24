# SmolVLA-RL: Critic-Guided Rollout Post-Training for a Compact VLA

RECAP-style reinforcement post-training for [SmolVLA](https://arxiv.org/abs/2506.01844) (450M), evaluated on the LIBERO benchmark: a distributional critic scores the policy's own rollouts, per-frame advantages are thresholded into positive/negative tokens, and the policy is retrained with advantage conditioning so classifier-free guidance can steer it at inference. Pre-training uses knowledge insulation (KI) with FAST-token co-training of the VLM backbone; SnapFlow one-step distillation removes the iterative flow-matching solve for real-time inference on edge hardware (Jetson Orin Nano).

Best configuration (critic-labeled rollouts co-trained with expert demonstrations, positive fraction 0.8): **68.1%** average success across three LIBERO suites, vs. 62.9% for the base policy and 52.4% for plain SmolVLA — with rollouts collected fully autonomously, no human interventions.

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

## Reproducing the paper's results

Every number in the paper comes out of the pipeline above; the differences between table rows are only in which config you run. In short:

1. **Base policy and no-KI baseline (Table I, first block)** — round-1 pre-training on `HuggingFaceVLA/libero` with pre-computed advantages, 250k steps on 2×H200 (~19 h). The base config is [`scripts/cluster/configs/recap_libero_aug.json`](scripts/cluster/configs/recap_libero_aug.json) (add `--no_augmentation` to match the paper's original base exactly); the no-KI baseline is [`scripts/cluster/configs/smolvla_noki_nodrop.json`](scripts/cluster/configs/smolvla_noki_nodrop.json).
2. **Rollout fine-tunes (Table I, remaining rows)** — round 2 as above: collect rollouts with `scripts/record_eval.py`, fine-tune the critic, recompute advantages at the row's positive fraction (pf = 0.4 or 0.8), then fine-tune with or without demo co-training (`--demo_mix_ratio 0.5`). The outcome-only row skips the critic and labels whole episodes by success.
3. **Evaluation sweeps** — each cell is 500 episodes per suite per guidance weight w ∈ {0, 0.5, 1, 1.5, 2}, run as a SLURM array ([`scripts/cluster/rat_aug.py`](scripts/cluster/rat_aug.py) + [`scripts/cluster/sub_aug_eval.sh`](scripts/cluster/sub_aug_eval.sh) are a working pair; edit the three constants at the top for a new sweep) and harvested with [`scripts/cluster/harvest_eval.py`](scripts/cluster/harvest_eval.py). Suite-level noise at n = 500 is about ±2 points — treat smaller differences as ties.
4. **SnapFlow distillation ablation (Table "one-step distillation")** — `train_snapflow.py` from the co-trained pf 0.8 checkpoint, 15k steps on 1×H200; the deployed student (v5) uses `--demo_mix_ratio 0.5`, the frozen-teacher guidance-baking variant (v7) is [`scripts/cluster/configs/snapflow_ctpf08_v7.json`](scripts/cluster/configs/snapflow_ctpf08_v7.json).
5. **Edge latency** — `scripts/bench_jetson.py` on a Jetson Orin (bf16), teacher vs. SnapFlow student, giving the 922 ms → 255 ms (3.6×) chunk-latency numbers.
6. **`libero_object` exclusion control (Sec. IV)** — the base pre-training config run as-is (augmentation is on by default) reproduces the augmented control: `object` goes from 0% to 53.6% while the three paper suites stay within noise (63.5 vs. 62.9 avg), showing the exclusion was a visual-domain gap, not bad data. Evaluate with the same sweep as step 3 — `rat_aug.py` already includes all four suites.

[`docs/cluster_runbook.md`](docs/cluster_runbook.md) has the full cluster workflow (access, submission, chaining, monitoring) plus the current state of all experiments — every result table, the run-by-run SnapFlow lessons, and a checkpoint map.

## Paper

`paper/` contains the full LaTeX source; every figure regenerates from data committed in this repo (`paper/make_figures.py`, `paper/make_filmstrips.py`). CI builds the PDF and deploys the project page on every push to main (`.github/workflows/paper.yml`).
