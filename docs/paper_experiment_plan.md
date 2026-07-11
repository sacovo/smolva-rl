# Paper Experiment Plan — RECAP Rollout Iteration for SmolVLA on LIBERO

Goal: collect a complete, systematic dataset to support the paper. Written to run over a
(less-crowded) weekend. Builds on `docs/recap_findings_overview.md`.

---

## 0. Claims → evidence required

| # | Claim | Evidence needed |
|---|---|---|
| C1 | RECAP rollout iteration improves a strong SmolVLA policy **without forgetting** | Model-comparison table (baseline → 250k → rollout-iterated), all suites, n=500, best-cfg + cfg 0; per-suite forgetting deltas |
| C2 | **Knowledge Insulation** is the dominant pretrain lever | KI on/off at pretrain (have) + KI on/off at the rollout finetune |
| C3 | Advantage-conditioning / **CFG value is suite-difficulty dependent** (≈0 on easy suites where base is near ceiling) | CFG curves per suite × model; plot "cfg gain" vs base accuracy / suite |
| C4 | **`positive_fraction` (negative-labeling aggressiveness) shapes CFG behavior** — the methodological contribution | **Two-point contrast**: pf=0.3 (aggressive negatives, ~8% positive) vs pf=0.8 (relaxed, 51.5% positive) — cfg curves + neg-token gap for each. If pf=0.8 makes cfg help → result + "find optimum" is future work; if not → clean negative result. No full sweep. |
| C5 | Diagnostic: negative-token probing quantifies learned pos/neg separation | `RECAP_FORCE_ADV=neg` eval across variants × suites |
| C6 | **SnapFlow one-step distillation** — speed↔accuracy tradeoff of the best policy | Distill best RECAP checkpoint → SnapFlow; eval accuracy (n=500) + **inference-latency benchmark** (1-step vs N-step FM) on fixed GPU |

---

## 1. Fixed protocol (apply everywhere for comparability)

- **Suites:** all 4 LIBERO — `spatial`, `object`, `goal`, `libero_10`. (Re-add **object**, which we dropped.)
- **CFG grid:** `{0.0, 0.5, 1.0, 1.5, 2.0}`.
- **Episodes:** **50 / task → n=500 / suite** for all *final* numbers. Report success % + 95% binomial CI.
  - Use **n=50 (5/task)** only for exploration / smoke; never mix n in a reported table.
  - **Fix the `N_EPISODES` bug**: set it explicitly per script (the quick-check `5` silently propagated into the "full" sweeps → current rollout variants are only n=50).
- **Seeds:** 3 training seeds for the **headline** configs (250k, rollout-only, best-pf); single seed for the pf-sweep interior; 2 seeds at the pf endpoints. Eval is ~deterministic per policy, so seed variance comes from *training*.
- **Right-sized SLURM** (from `seff`): eval `--mem=24G --cpus=8`; train `--mem=96G`. Prefer `h200`; fan out to `performance`/`top6`/`p4500` when idle.

---

## 2. What we already have (reuse — don't rerun)

- **250k (KI)** and **smolvla_noki (no-KI)**: n=500 on spatial + goal (250k also libero_10). *Missing: object; baseline libero_10.*
- Trained finetune checkpoints: `rollout-only` (pf0.4), `co-train` (pf0.4+demos), `pf0.8` — but their evals are **n=50 only** → must re-eval at n=500.
- Critic + advantages (`outputs/rollout_advantages*`) — reusable; thresholds recompute per pf in seconds (CPU).
- Neg-token diagnostic on rollout-only (n=50).

---

## 3. Experiment matrix

### Trainings (finetune from `libero_recap_250000`, 20k steps, ~90 min / GPU)

**FINALIZED scope — only 2 new training runs:**

| Run | Purpose | Notes |
|---|---|---|
| **`recap_pf03`** | pf=0.3 low endpoint (C4) | recompute pf0.3 thresholds (CPU) → finetune. Have pf0.4/0.8 already. |
| **SnapFlow-of-best** | C6 tradeoff | `submit_snapflow.sh sancov/rollout_merged <best_ckpt>` — distil the winning n=500 variant to 1-step |

Reuse (no retrain): `recap_rollout_p3` (pf0.4), `recap_pf08` (pf0.8), `recap_cotrain`, `250k`, `smolvla_noki`.
Single seed (no seed multiplication). Threshold recompute: `scripts/recompute_thresholds.py` (CPU, ~2 min).

### Evals (n=500, cfg×5, per suite)

| Model set | Suites | cfg | n | tasks |
|---|---|---|---|---|
| Baseline completion (250k, no-KI) | object (+ no-KI libero_10) | 0.5,1 or full | 500 | ~15 |
| **pf sweep** (6 models incl 0.4) | 4 | 5 | 500 | 120 |
| co-train | 4 | 5 | 500 | 20 |
| **Neg-token** (all key variants) | 4 | 1.0 only | 500 | ~28 |
| Seeds (headline models) | 4 | full | 500 | ~120 |

---

## 4. FINALIZED experiment set (locked scope)

Scope: 3 suites (spatial/goal/libero_10), n=500, full cfg {0,0.5,1,1.5,2}, single seed.

**Models to evaluate at n=500** (all 3 suites × 5 cfg = 15 tasks each):
| Model | Checkpoint | Status |
|---|---|---|
| 250k (KI) | `libero_recap_250000` | have n=500 all 3 suites ✓ |
| smolvla_noki (baseline) | — | have spatial/goal; **need libero_10** |
| rollout-only (pf0.4) | `recap_rollout_policy` | re-eval n=500 |
| **pf0.3** (aggressive neg) | `recap_pf03` (train) | train → eval n=500 |
| **pf0.8** (relaxed neg) | `recap_pf08_policy` | re-eval n=500 |
| co-train (pf0.4+demos) | `recap_cotrain_policy` | re-eval n=500 (ablation: positive anchor) |
| **SnapFlow-of-best** | distil winner | after main evals → eval n=500 |

**Diagnostics:** neg-token (`RECAP_FORCE_ADV=neg`, cfg 1.0, 3 suites, n=500) for pf0.3, pf0.8 (± rollout-only) — quantifies how pf changes the learned pos/neg gap (C4/C5).

**Compute:** ~7 models × 15 + baseline-fill (~5) + neg-token (~9) ≈ **~120 eval tasks** at n=500 + **2 trainings** + 1 SnapFlow distil + latency benchmark.

---

## 5. Compute budget & weekend schedule

Rough eval cost at n=500: spatial/object/goal ≈ **3 h**, libero_10 ≈ **6 h** per (suite,cfg) task.
Finetune ≈ **1.5 h**.

| Tier | eval tasks | ≈ GPU-h | on 8 GPUs |
|---|---:|---:|---:|
| A | ~75 | ~280 | ~35 h |
| A+B | ~120 | ~450 | ~56 h |
| A+B+C | ~270 | ~1000 | ~125 h |

Sequencing (dependency-aware; trainings gate their evals):
1. **Fri PM:** recompute thresholds for all pf; launch **all finetunes** (pf0.3/0.5/0.7/1.0 + seeds) — they're 90 min each and cheap; get them done first while eval GPUs are busy.
2. **Sat:** fan out **eval sweeps** as checkpoints land. Prioritize `libero_10` early (longest). Spread across partitions.
3. **Sun:** neg-token evals + object-suite completion + any reruns; collect + aggregate.

Compute-savers if the weekend is short: drop cfg grid to `{0,1,2}` for the **pf interior** (0.5/0.7), keep full grid for endpoints + headline; or use **n=250** (CI ±6%) for the interior.

---

## 6. Concrete ordered job list (ready to script)

Naming: threshold dirs `rollout_advantages_pf{03,05,07,10}`; finetunes `recap_pf{03,05,07,10}` →
`outputs/recap_pf{XX}_policy`; eval labels `rollout_pf{XX}`; **all evals `N_EPISODES=50` (n=500)**.

1. `thresholds_pfXX.py` (CPU) → 4 new threshold jsons  *(mirror the pf0.8 recompute)*
2. `submit_recap … --thresholds_path … --job_name recap_pfXX` → 4 finetunes  *(+ seeds via `--seed`)*
3. `run_array_task_pfXX.py` + `submit_eval_sweep_pfXX.sh` (24G/8cpu, N_EPISODES=50, 4 suites × 5 cfg) → eval
4. Re-eval existing: rollout-only, co-train, pf0.8 at n=500 (new labels `*_n500` to avoid the n=50 dirs)
5. Baseline completion: 250k+no-KI on object; no-KI on libero_10
6. Neg-token: copy eval scripts with `export RECAP_FORCE_ADV=neg`, cfg 1.0, 4 suites, per variant
7. Aggregate: one collector → CSV of (model, pf, suite, cfg, n, pc, CI) + neg/pos gaps → tables/figures

**Automation:** a single driver `run_pf_experiment.sh XX` doing steps 1→3 per pf makes the sweep one-command and reproducible for the paper appendix.

---

## 7. SnapFlow + compute benchmark (C6)

**Distillation:** once the n=500 comparison picks the **best RECAP checkpoint**, distil it:
`./scripts/submit_snapflow.sh sancov/rollout_merged <best_ckpt> --partition h200` → `train_snapflow.py`
(target-time-conditioned, zero-init `target_time_mlp` so it starts = the FM teacher; advantage token = positive during distillation). Produces `outputs/snapflow_*/…/migrated`.

**Accuracy:** eval SnapFlow model at n=500, 3 suites, cfg {0,1} (SnapFlow supports CFG) → accuracy delta vs the multi-step teacher.

**Compute benchmark (the tradeoff figure):** on a single fixed GPU, measure **action-chunk generation latency**, teacher (N-step FM Euler solve) vs SnapFlow (1-step jump), same batch=1, warm cache, ≥100 timed calls, report mean ± std:
- ms / action-chunk and effective control Hz
- separate the **VLM-prefix** cost (shared) from the **denoise** cost (where SnapFlow wins) → shows the real end-to-end speedup, not just the solver step count
- report the FM step count used (`num_steps` / inference default) so the ratio is interpretable
- → Figure: accuracy (y) vs latency (x), teacher vs SnapFlow.

Small standalone script (`scripts/bench_inference.py`): load both checkpoints, `torch.cuda.synchronize()` around `select_action`/`sample_actions`, no env, GPU-only timing.

## 8. Decisions (LOCKED)

- **Suites:** 3 (spatial, goal, libero_10) — object dropped.
- **n:** 500 everywhere.
- **Seeds:** single seed (no multiplication).
- **pf:** two-point contrast — **pf=0.3 (train fresh)** vs **pf=0.8 (have)**; pf=0.4 rollout-only kept as mid reference.
- **cfg:** 5 points {0,0.5,1,1.5,2}.
- **+ SnapFlow** on the best run + latency benchmark (C6).

**Next:** I'll (1) recompute pf0.3 thresholds + queue `recap_pf03`, (2) prep n=500 eval scripts (with `N_EPISODES=50` — the bug fixed) + neg-token scripts + `bench_inference.py`, staged to fire when the cluster frees. SnapFlow distil waits until the best variant is known.
