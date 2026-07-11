# RECAP Rollout-Iteration — Findings Overview

_Status snapshot. LIBERO (SmolVLA + RECAP). Suites: spatial, goal, libero_10 (object dropped)._
_Metric = `overall pc_success` (% task success)._

> **Sample-size caveat (important):** `250k` and `smolvla_noki` baselines were run at **50 ep/task (n=500)**.
> All **rollout-derived variants** (rollout-only, co-train, pf0.8, neg-test) were run at **5 ep/task (n=50)** —
> the `N_EPISODES=5` from the quick forgetting-check propagated through the eval-script copies.
> So rollout-variant numbers carry ~±7% noise; their **cfg-curve _shapes_** are indicative, not final.
> A proper n=500 re-run of the winning variant is the obvious follow-up.

---

## 1. Models / Training runs

All trained on 2×/1× H200, `num_vlm_layers=16`, `batch=32`, `accum=8`. Finetunes warm-start from `outputs/libero_recap_250000`.

| Run (wandb name) | Job | Role | Dataset | Steps | ar_w / fm_w | KI | Thresholds (pf) | Demos | ~Runtime |
|---|---|---|---|---|---|---|---|---|---|
| `plain_smolvla_libero` | — | **250k base** (`libero_recap_250000`) | HuggingFaceVLA/libero | 250k | 0 / 4 | **on** | — | — | — |
| `smolvla_libero_noki` | — | **baseline** (no-KI) | HuggingFaceVLA/libero | 250k | 0 / 4 | **off** | — | — | — |
| `recap_rollout_p3` | 169009 | **rollout-only** | sancov/rollout_merged | 20k | 1 / 1 | on | pf0.4 | — | 89 min |
| `recap_cotrain` | 170032 | **co-training** (+expert positives) | rollout_merged + HuggingFaceVLA/libero | 20k | 1 / 1 | on | pf0.4 | mix 0.5 | 90 min |
| `recap_pf08` | 170166 | **clean-negatives** (pf0.8) | sancov/rollout_merged | 20k | 1 / 1 | on | **pf0.8** | — | 90 min |

Notes:
- The "250k" model is **flow-matching + KI** (ar_loss=0, fm_loss=4) — not a full AR-cotrained RECAP. The finetunes turn AR back on (ar=1, fm=1).
- All finetunes ran the full 20k steps (step 19990), warm-start clean (490 "missing keys" is a benign shared-parameter alias — the VLM appears under both `vlm_with_expert.*` and `fast_wrapper.vlm_with_expert.*`).

## 2. RECAP data pipeline (the rollout iteration)

| Stage | Output | Notes |
|---|---|---|
| 1. Messy rollouts | `sancov/rollout_merged` (HF) | 600 eps, **386 success (64.3%)**, 107,410 frames, 30 tasks, image dtype |
| 2. Critic finetune | `outputs/critic_rollout/critic/checkpoint_final.pt` | warm-started; loss 11→2.25 |
| 3a. Advantages | `outputs/rollout_advantages/*.npy` | all 107,410 frames; min −35.7 (C_FAIL), max +0.98, mean −3.55 |
| 3b. Thresholds pf0.4 | `outputs/rollout_advantages/*.json` | ε −0.13..0.29 → **12% of frames positive** |
| 3c. Thresholds pf0.8 | `outputs/rollout_advantages_pf08/*.json` | ε −0.26..0.12 → **51.5% of frames positive** |
| 4. Policy retrain | `outputs/recap_{rollout,cotrain,pf08}_policy/…/migrated` | 3 variants above |

---

## 3. Eval results

### 3a. Baselines (n=500)

**250k (RECAP / flow-matching + KI)** — cfg 0.0 / 0.5 / 1.0 / 1.5 / 2.0
| suite | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | best |
|---|---:|---:|---:|---:|---:|---|
| spatial | 69.4 | 71.2 | 71.0 | 69.2 | 68.2 | 71.2 |
| goal | 69.4 | 73.8 | 72.6 | 75.8 | 72.6 | 75.8 |
| libero_10 | 35.4 | 39.6 | 41.6 | 36.0 | 36.8 | 41.6 |

**smolvla_noki (baseline, no KI, no RECAP)** — n=500
| suite | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 |
|---|---:|---:|---:|---:|---:|
| spatial | 58.0 | 58.6 | 59.4 | 57.6 | 61.0 |
| goal | 61.2 | 65.8 | 63.0 | 66.0 | 60.2 |

### 3b. Rollout-iteration variants (n=50 — noisy)

| variant | suite | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 |
|---|---|---:|---:|---:|---:|---:|
| **rollout-only** (pf0.4) | spatial | 76.0 | 74.0 | 70.0 | 72.0 | 62.0 |
| | goal | 78.0 | 78.0 | — | — | — |
| | libero_10 | — | — | 38.0 | — | — |
| **co-train** (pf0.4 + demos) | spatial | 76.0 | 70.0 | 72.0 | 66.0 | 74.0 |
| **pf0.8** (clean negatives) | spatial | 74.0 | — | — | — | — _(GPU-starved)_ |

### 3c. Negative-token diagnostic (n=50, cfg 1.0)

Force the conditioning token to `<advantage_negative>` (`RECAP_FORCE_ADV=neg`) vs normal positive:

| suite | negative | positive | gap |
|---|---:|---:|---:|
| spatial | 72.0 | 76.0 | 4.0 |
| goal | 74.0 | 78.0 | 4.0 |
| libero_10 | 26.0 | 38.0 | 12.0 |

---

## 4. Findings

1. **KI is the big win at pretrain.** 250k (KI on) beats the no-KI baseline by **~10–12 pts** everywhere
   (spatial 71 vs 59, goal 74 vs 63). This is the largest single effect measured.

2. **The rollout iteration raises peak accuracy** (tentative, n=50). Rollout-only spatial hits **76** (cfg 0)
   vs 250k's best **71** (+5), goal **78** vs **75.8**, with **no catastrophic forgetting**. libero_10 ≈ flat
   (38 vs 41.6). So retraining the policy on its own messy rollouts + advantage labels helped the base policy.

3. **CFG / advantage-conditioning does _not_ help on these (easy) suites.** For every rollout variant the cfg
   curve **peaks at cfg 0** (unconditional) and is flat-to-declining. On the well-sampled 250k it's also flat
   (~70 across cfg). The base policy is already good enough that steering toward "positive" adds nothing.

4. **Why (neg-test):** the model _does_ use the token (negative < positive everywhere), but the gap is **tiny on
   the easy suites (4 pts)** — a negative-conditioned policy still succeeds 72–74%. The positive/negative/uncond
   span is compressed into ~72–76, so CFG has almost nothing to amplify and hurts when extrapolated (cfg 2 → 62).
   Bigger gap on libero_10 (12 pts), whose threshold is lower / less polluted.

5. **Two candidate causes tested:**
   - **Weak positive branch → co-training** (add expert demos as guaranteed-positive): **did not flip cfg**
     (spatial peak still at cfg 0), but **softened the high-cfg collapse** (cfg 2.0: 74 vs rollout-only's 62).
   - **Over-aggressive negative labeling → pf0.8** (raise positive_fraction 0.4→0.8, so only the worst 20% of
     successes + failures are negative; overall positive frames 12% → 51.5%): **incomplete** — only spatial cfg 0.0
     (=74) landed before the cluster GPU-starved the eval. **Decisive curve still pending.**

6. **Mechanistic read:** on LIBERO's easy suites the bottleneck isn't the conditioning mechanism — the
   unconditional policy is already near-ceiling, so advantage-conditioning/CFG can't add lift. The labeling
   fixes reshape the tails (esp. the high-cfg collapse) but don't make cfg > 0 win. libero_10 (harder, lower
   accuracy, less-polluted negatives) is where conditioning has the most signal and is the suite worth pushing.

## 5. Open / follow-ups

- **Finish the pf0.8 cfg curve** (GPU-starved) — the one missing decisive data point.
- **Re-run the best rollout variant at n=500** for publishable numbers (current rollout comparisons are n=50).
- **Fill libero_10** for the rollout variants (only rollout-only cfg 1.0 = 38 exists) — likely the most
  informative suite for conditioning.
- Held/queued: cotrain goal+libero_10 (`170360_[5-14]`), rollout-only goal/libero_10 n=500 (`169826_[7-14]`).

## 6. wandb

- Training: project **`smolvla-recap`** (run = job_name). Eval: **`smolvla-recap-eval`**.

## 7. Per-task failure analysis (n=500, cfg 1.0)

Per-task success % (50 ep/task). Policies: 250k (KI), noki (no-KI baseline), rollout (pf0.4),
pf08 (pf0.8), cotrain (pf0.4+demos), pf03 (pf0.3). `min`/`spread` across the policies present.
(spatial = all "pick up the black bowl **[X]** and place it on the plate".)

| task | description | 250k | noki | rollout | pf08 | cotrain | pf03 | min | spread |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spatial t0 | bowl **between the plate and the ramekin** | 68 | 50 | 82 | 84 | 46 | 78 | 46 | 38 |
| spatial t1 | bowl **next to the ramekin** | 84 | 76 | 66 | 62 | 88 | 72 | 62 | 26 |
| spatial t2 | bowl **from table center** | 88 | 84 | 96 | 98 | 92 | 96 | 84 | 14 |
| spatial t3 | bowl **on the cookie box** | 90 | 86 | 86 | 90 | 86 | 72 | 72 | 18 |
| spatial t4 | bowl **in the top drawer of the wooden cabinet** | 66 | 46 | 40 | 62 | **92** | 52 | 40 | 52 |
| spatial t5 | bowl **on the ramekin** | 18 | 18 | **64** | **70** | 18 | 52 | 18 | 52 |
| spatial t6 | bowl **next to the cookie box** | 94 | 54 | 92 | 78 | 86 | 78 | 54 | 40 |
| spatial t7 | bowl **on the stove** | 74 | 60 | 74 | 76 | 76 | 72 | 60 | 16 |
| spatial t8 | bowl **next to the plate** | 78 | 64 | 48 | 66 | 70 | 30 | 30 | 48 |
| spatial t9 | bowl **on the wooden cabinet** | 50 | 56 | 56 | 68 | 72 | 58 | 50 | 22 |
| goal t0 | **open the middle drawer of the cabinet** | 10 | 42 | 12 | 20 | **92** | 4 | 4 | **88** |
| goal t1 | put the bowl on the stove | 94 | 62 | 96 | 100 | 94 | 98 | 62 | 38 |
| goal t2 | put the wine bottle on top of the cabinet | 76 | 60 | 92 | 86 | 88 | 80 | 60 | 32 |
| goal t3 | open the top drawer and put the bowl inside | 64 | 48 | 78 | 78 | 56 | 68 | 48 | 30 |
| goal t4 | put the bowl on top of the cabinet | 96 | 86 | 94 | 94 | 94 | 96 | 86 | 10 |
| goal t5 | push the plate to the front of the stove | 82 | 60 | 98 | 90 | 82 | 100 | 60 | 40 |
| goal t6 | put the cream cheese in the bowl | 62 | 46 | 54 | 36 | 52 | 36 | 36 | 26 |
| goal t7 | turn on the stove | 100 | 100 | 100 | 100 | 100 | 98 | 98 | 2 |
| goal t8 | put the bowl on the plate | 88 | 74 | 90 | 90 | 90 | 98 | 74 | 24 |
| goal t9 | put the wine bottle on the rack | 54 | 52 | 46 | 82 | **94** | 58 | 46 | 48 |
| libero_10 t0 | **put both the alphabet soup and the tomato sauce in the basket** | 14 | — | 22 | 12 | 8 | — | 8 | 14 |
| libero_10 t1 | put both the cream cheese box and the butter in the basket | 48 | — | 24 | 22 | 42 | — | 22 | 26 |
| libero_10 t2 | turn on the stove and put the moka pot on it | 62 | — | 52 | 64 | 74 | — | 52 | 22 |
| libero_10 t3 | put the black bowl in the bottom drawer of the cabinet and close it | 78 | — | 76 | 74 | 68 | — | 68 | 10 |
| libero_10 t4 | **put the white mug on the left plate and the yellow/white mug on the right plate** | 34 | — | 8 | 10 | 26 | — | 8 | 26 |
| libero_10 t5 | pick up the book and place it in the back compartment of the caddy | 76 | — | 68 | 80 | 66 | — | 66 | 14 |
| libero_10 t6 | put the white mug on the plate and the chocolate pudding to the right | 36 | — | 34 | 18 | 38 | — | 18 | 20 |
| libero_10 **t7** | **put both the alphabet soup and the cream cheese box in the basket** | 12 | — | 0 | 2 | 16 | — | 0 | 16 |
| libero_10 t8 | **put both moka pots on the stove** | 14 | — | 8 | 10 | 20 | — | 8 | 12 |
| libero_10 t9 | put the yellow and white mug in the microwave and close it | 42 | — | 40 | 66 | 70 | — | 40 | 30 |

### Findings

**(a) Universally-hard = the two-object "put both A and B" tasks in libero_10.** The hard core is semantically
coherent — the tasks requiring **two sequential pick-and-places** are near-unsolvable for everyone:
- t7 "put both the alphabet soup **and** the cream cheese box in the basket" — **~0–16% (unsolvable)**
- t0 "put both the alphabet soup **and** the tomato sauce in the basket" — 8–22%
- t8 "put **both** moka pots on the stove" — 8–20%
- t4 "put the white mug on the left plate **and** the yellow/white mug on the right plate" — 8–34%

So libero_10's ~35–44% ceiling is **dual-object long-horizon manipulation**, not diffuse difficulty. No
labeling/co-training change touches these → a **capability ceiling** (likely error-compounding over two subgoals),
not a method gap. (Single-subgoal libero_10 tasks like t3, t5 are ~70–80%, comparable to spatial/goal.)

**(b) Models specialize; gains aren't uniform.** Aggregate differences come from a few "swing" tasks:
- **goal t0 = "open the middle drawer of the cabinet"** (spread 88): **only co-training solves it** (92% vs
  ≤42% for all others). It's an **articulated-object** skill — the expert demos taught drawer-opening that the
  self-rollout policies never learned. This one task largely explains cotrain's goal lead.
- **spatial t5 = "bowl on the ramekin"** (spread 52): the *rollout-only* variants learn it (rollout 64, pf08 70)
  but **co-training regresses to baseline (18%)** — adding demos *lost* this specific bowl-grasp config.
- **spatial t4 = "bowl in the top drawer"** (92) and **goal t9 = "wine bottle on the rack"** (94): co-training
  uniquely masters these (t4 also a drawer/articulated skill → consistent with the demos-teach-articulation story).

**(c) Only 1/30 tasks is universally easy** (≥90% for all): goal t7.

**Takeaway for the paper:** report this as a failure-analysis / task-breakdown — the headline gaps are driven by
task specialization + a hard libero_10 core, not uniform improvement. Follow-up: inspect the BDDL/videos of
libero_10 t7/t0/t8 to characterize the ceiling.
