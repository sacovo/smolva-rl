# Cluster runbook and experiment state

Last updated: **2026-07-15**. How we start, monitor, and evaluate runs on the
FHNW SLURM cluster (`fhnw-hpc` = calculon.informatik.fhnw.ch), plus the full
current state of the experiments. Companion to
`docs/ki_experiment_and_rollout.md` (method log) and `docs/snapflow.md`.

Committed cluster artifacts live in `scripts/cluster/`:

| file | purpose |
|---|---|
| `configs/recap_libero_aug.json` | augmented base pre-training (active run) |
| `configs/snapflow_ctpf08_v7.json` | frozen-teacher guidance distillation |
| `configs/smolvla_noki_nodrop.json` | clean no-KI baseline pre-training |
| `rat_aug.py` | eval array worker (checkpoint × suite × w per task) |
| `sub_aug_eval.sh` | sbatch wrapper for the eval array |
| `harvest_eval.py` | collect all `eval_info.json` into a results table |

---

## 1. Access

```bash
ssh fhnw-hpc                      # alias for calculon.informatik.fhnw.ch (needs FHNW VPN)
```

Login node is for submission and file work only — no compute. Every job script
needs the following environment (PATH-only activation is **not** enough; the
conda env supplies EGL for headless MuJoCo):

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate ~/conda12
export MUJOCO_GL=egl
unset LEROBOT_HOME                       # deprecated var now hard-errors
export HF_LEROBOT_HOME="$HOME/.cache/huggingface/lerobot"
# do NOT set HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE — compute nodes have internet
```

Code lives at `~/smolvla-rl` (synced from this repo via `rsync`, not git).
After local changes:

```bash
rsync -a src/lerobot_policy_smolvla_rl/ fhnw-hpc:~/smolvla-rl/src/lerobot_policy_smolvla_rl/
```

## 2. Starting training runs

All training goes through the config-driven submitters on the cluster
(`scripts/submit_recap.py`, `scripts/submit_snapflow.py`,
`scripts/submit_critic.py`). Each takes `--config <json>` with a `slurm` and a
`training` section; the training dict is serialized to CLI flags for the
corresponding `train_*.py` (booleans `true` become bare flags, `false`/`null`
are dropped). Avoid `scripts/train_slurm.sh` — its `TRAIN_ARGS=$@` quoting is
a known hazard; the Python submitters are the supported path.

```bash
ssh fhnw-hpc
source /opt/conda/etc/profile.d/conda.sh && conda activate ~/conda12
cd ~/smolvla-rl
python scripts/submit_recap.py    --config <config.json>   # pre-train / fine-tune
python scripts/submit_snapflow.py --config <config.json>   # distillation
```

Key conventions (see `scripts/cluster/configs/*.json` for full examples):

- **Long runs across 24 h walls**: set `slurm.num_jobs: N` (submits N chained
  jobs with `afterany`) **and** `training.resume_from: "auto"`. The trainer
  saves state shortly before the wall (`duration` is auto-set from
  `slurm.time`) and the next job resumes. A follow-up job that finds the run
  already complete just re-exports — harmless.
- **Pre-training** (250 k steps): `gpu:h200:2`, batch 32 × accum 8,
  `fm_loss_weight 4.0`, `num_vlm_layers 16`, chunk 20. Advantage-conditioned
  runs add `precomputed_advantages` + `thresholds_path`
  (`outputs/libero/task_*_HuggingFaceVLA_libero.*` for the demo dataset).
  Image augmentation (photometric) is **on by default** since commit
  `aa56ee5`; disable with `no_augmentation: true`, add geometric with
  `augmentation_geom: true`.
- **Fine-tunes** (20 k steps, ~90 min on one H200): same submitter, warm-start
  via `resume_from`/checkpoint args, rollout advantages from
  `outputs/rollout_advantages*` (see §5).
- **Distillation** (15 k steps, ~2–3 h on one H200): `submit_snapflow.py`;
  `precomputed_advantages`/`thresholds_path` for true ± labels,
  `demo_dataset_repo_id` + `demo_mix_ratio 0.5` for the demo mix,
  `distill_cfg_weight` + `frozen_teacher` for guidance baking.

Every submission archives its resolved config to
`runs/configuration/config_<jobname>_<timestamp>.json` — the authoritative
record of what a run actually used.

## 3. Evaluation

Evals are SLURM arrays: one cell = one (checkpoint, suite, guidance weight),
50 episodes/task → n=500/suite. The worker (`rat_*.py`, template:
`scripts/cluster/rat_aug.py`) reads `SLURM_ARRAY_TASK_ID`, runs `lerobot-eval`,
and logs to W&B; the wrapper (`sub_*.sh`, template:
`scripts/cluster/sub_aug_eval.sh`) sets the environment and array bounds.

To evaluate a new checkpoint: copy both templates, edit `CHECKPOINTS`,
`SUITES`, `CFG_SCALES` in the worker and `--array=0-(cells-1)` + job name in
the wrapper, then chain it to training:

```bash
sbatch --dependency=afterok:<train_jobid> scripts/sub_<name>.sh
```

`afterok` (not `afterany`) so a failed training cancels the eval instead of
evaluating a broken checkpoint. Workers skip cells whose `eval_info.json`
already exists, so re-running an array after stragglers is free.

Conventions: sweep w ∈ {0, 0.5, 1, 1.5, 2} for advantage-conditioned models;
single point w=1 for models where guidance is undefined (constant-token
baseline) or already baked in (guided students). Timing on p4500: ~1–1.5 h for
spatial/goal, ~2.5–6 h for long/object. If a 14 h p4500 cell risks the wall,
resubmit the cell on h200.

## 4. Monitoring and harvesting

```bash
squeue -u $USER -o "%.9i %.22j %.9T %.11M %.20E"      # queue incl. dependencies
sacct -j <ids> -X --format=JobID,JobName%30,State,Elapsed,End
tail -f logs/<jobname>_<jobid>.out                     # submitter/stdout
tail -c 2000 logs/<jobname>_<jobid>.err | tr '\r' '\n' | tail -5   # tqdm lives in .err
python3 scripts/cluster/harvest_eval.py "2026-07-1*"   # results table
```

W&B projects: `smolvla-recap` / `smolvla-snapflow` (training),
`smolvla-recap-eval` (per-cell eval results with per-task breakdown).
Queue note: fairshare weight is 0 on this cluster — priority is age-based and
backfill favors short walltimes; don't cancel+resubmit to "fix" priority.

## 5. Rollout collection, critic, advantages (the RECAP loop)

1. **Rollouts**: `scripts/record_eval.py --headless` via
   `scripts/submit_rollout.sh`, one job per suite → `outputs/rollout_*`;
   merge with `lerobot.datasets.aggregate` → e.g. `sancov/rollout_merged`.
2. **Critic fine-tune**: `scripts/submit_critic.py`, warm-started from
   `outputs/libero/critic/checkpoint_final.pt` on the merged rollouts.
3. **Advantages/thresholds**: `compute_thresholds.py` →
   `outputs/rollout_advantages*` (`_pf08`, `_pf30`, `_outcome` variants).
4. **Policy fine-tune**: `submit_recap.py` with the advantage files (never
   `--expert_mode` on rollout data).

## 6. Edge benchmarks (Jetson)

`scripts/bench_jetson.py` on `jetson@192.168.31.26` (Orin Nano 8 GB) and
`jetson@172.16.10.164` (Orin NX 16 GB). Working stack: cp312 venv, PyPI torch
cu13, `LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat_orin`, **bf16 + autocast**
(fp32 OOMs the Nano). Full setup and dead ends: memory note
`jetson-orin-bench-setup` / Appendix of the paper.

---

## 7. Experiment state (2026-07-15)

Benchmark: LIBERO, n=500/suite, percent success at best guidance weight
(best-of-sweep) unless noted. `libero_object` is 0 %/timeout for every policy
so far (paper excludes it; the active augmentation run tests whether that is a
visual-domain gap). Suite-level SE ≈ ±2 points.

### Pre-training / baselines (250 k steps, 2×H200)

| model | spatial | goal | long | avg | notes |
|---|---:|---:|---:|---:|---|
| no-KI baseline (`libero_smolvla_noki_nodrop`) | 65.0 | 70.6 | 30.2 | 55.3 | SmolVLA-style: FM trains VLM, no AR, constant token, no dropout; w=1 only |
| **base +KI** (`outputs/libero`, 250 k ckpt) | 71.2 | 75.8 | 41.6 | 62.9 | π_base of the paper; KI + FAST AR + adv-conditioning, token dropout 0.3 |

(The older `libero_smolvla_noki` run — degenerate token *with* dropout — is
superseded by `noki_nodrop` and no longer used in the paper.)

### Fine-tunes on rollout data (20 k steps from π_base)

| config | spatial | goal | long | avg |
|---|---:|---:|---:|---:|
| rollout-only, pf 0.3 | 67.4 | 76.6 | 42.4 | 62.1 |
| rollout-only, pf 0.4 | 72.8 | 76.0 | 36.6 | 61.8 |
| rollout-only, pf 0.8 | 77.8 | 80.0 | 35.8 | 64.5 |
| outcome-only labels | 72.8 | 81.8 | 40.0 | 64.9 |
| co-train (mix 0.5), pf 0.4 | 75.4 | 84.2 | **43.8** | 67.8 |
| **co-train (mix 0.5), pf 0.8** | **80.0** | **86.2** | 38.2 | **68.1** |

Key findings: KI recipe +5–11 pts/suite over the clean baseline; rollout-only
degrades long; co-training gives the two best configs but only pf 0.4 fully
repairs long; CFG adds ~1 pt for rollout-only policies but +6.6 spatial for
co-train pf 0.8 (73.4→80.0 at w=1.5). Negative-token probes (w=1): pf 0.8
77.4/78.2/34.8, pf 0.4 69.2/76.2/29.4, pf 0.3 66.4/76.4/40.4 — small
separation everywhere.

### SnapFlow distillation (students of co-train pf 0.8; teacher 80.0/86.2/38.2, avg 68.1)

| student | steps | spatial | goal | long | avg | lesson |
|---|---|---:|---:|---:|---:|---|
| v1 all frames positive | 15 k | 43.8 | – | – | – | label bug: FM imitates neg. rollout actions |
| v2 critic ± labels | 15 k | 67.8 | 71.4 | 30.4 | 56.5 | true labels recover most of it |
| v3 ± labels | 60 k | 62.8 | 51.8 | 26.4 | 47.0 | over-distillation: loss 0.106→0.047, success ↓ |
| v4 ± labels | 120 k | 61.2 | 49.6 | 27.2 | 46.0 | worse still — stop at 15 k |
| **v5 + demo mix 0.5** | 15 k | 74.2 | 84.2 | 40.0 | **66.1** | **deployed student**; within 2.0 of teacher |
| v6 guidance baked (self-teacher) | 15 k | 67.0 | 72.4 | 32.6 | 57.3 | self-teacher blend collapses toward v_u; uncond drifts |
| v7 guidance baked (frozen teacher) | 15 k | 76.8 | 75.2 | 25.2 | 59.1 | fixes spatial (best single-pass, within 3.2 of guided teacher); overshoots on goal/long |

v5 single-pass operating points: 64.7 avg at w=0, 64.5 at w=1 (w∉{0,1}
doubles the expert batch). Edge latency (bf16): Orin Nano 922→255 ms (3.6×),
Orin NX 850→240 ms (3.5×); NX fp32 1143→575 ms (2.0×).

### Active runs (submitted 2026-07-14)

| job(s) | what | status at last check |
|---|---|---|
| 179460 → 179461 | base pre-training **with photometric augmentation**, 250 k, 2×H200, `outputs/libero_aug/` | 179460 running, ETA ~18 h, augmentation + advantages confirmed in logs |
| 179462 (`afterok:179461`) | eval array 0–19: **all four suites incl. object**, w ∈ {0,…,2}, n=500 | pending |

Decision when it lands: if the augmented base beats 71.2/75.8/41.6 (and/or
unbreaks object), redo the full rollout → critic → advantages(pf 0.8) →
co-train round on top of it and update paper Table I.

### Paper (`paper/main.tex`)

6 pages, compiles clean, all results above are in. Remaining red markers:
second author, acknowledgments. Open items: augmentation-run outcome (may
trigger a second self-improvement round and a rewrite of the object-suite
exclusion paragraph in Sec. IV).

### Checkpoint map (cluster, `~/smolvla-rl/outputs/`)

| path | contents |
|---|---|
| `libero/` | π_base pre-training + demo critic + demo advantages/thresholds |
| `libero_aug/` | augmented pre-training (active) |
| `libero_smolvla_noki_nodrop/` | clean no-KI baseline |
| `recap_cotrain_pf08_policy/` | best fine-tune = SnapFlow teacher |
| `rollout_advantages*/` | rollout advantages/thresholds (pf 0.8 / pf 0.3 / outcome) |
| `snapflow_ctpf08_v*_policy/` | distillation students v1–v7 (`.../snapflow_model/migrated`) |
| `eval/<date>/<time>_<jobname>/eval_info.json` | all eval results |
