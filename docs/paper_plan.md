# Paper Plan: Report, Brief, and Experiment Guide

Status: working plan (2026-07-04)

This document lays out (1) the paper brief with verified citations, and
(2) the experiment plans needed to fill the results tables. Repo cleanup
items live separately in `docs/cleanup_plan.md`.

---

## 1. Framing

Working title idea: *"Recipe Transfer to Small VLAs: Advantage-Conditioned
RL and One-Step Flow Distillation for SmolVLA."*

The defensible framing: Physical Intelligence demonstrated FAST co-training
with Knowledge Insulation (KI) and RECAP on large proprietary models; this
work provides an **open implementation and empirical study of whether the
recipe transfers to a 450M open model (SmolVLA)**, plus a latency study of
one-step flow distillation. A real contribution for a short paper without
overclaiming novelty.

**SnapFlow attribution:** the distillation implemented here follows the
SnapFlow paper — *SnapFlow: One-Step Action Generation for Flow-Matching
VLAs via Progressive Self-Distillation* (Luan et al., arXiv:2604.05656) —
which already evaluates on SmolVLA. It is a method dependency, not a
contribution: cite it as the source, and state the delta plainly (adapting
it to a RECAP+KI-trained policy and quantifying the speedup in that
pipeline). Do not claim novelty for one-step SmolVLA itself.

## 2. Section outline (budget for ~5 pages)

**Abstract (~150 words).** Small open VLAs make robot learning affordable,
but the recent training-recipe advances (FAST/KI co-training,
advantage-conditioned RL, few-step distillation) were shown on large
proprietary models. We implement the full recipe on SmolVLA-450M, evaluate
on LIBERO, and show: (a) whether AR co-training and advantage conditioning
improve success, (b) one-step distillation retains X% of success at ~N×
lower action-generation latency.

**1. Introduction (~0.75 page).** VLAs; cost problem; SmolVLA as the
affordable baseline [shukor2025smolvla]; the π-family recipe [black2024pi0,
intelligence2025pi05, driess2025knowledge, intelligence2025pistar06]; open
question of transfer to small models; contributions list (implementation,
ablation study, latency study). RT-2/OpenVLA as background [brohan2023rt2,
kim2024openvla].

**2. Related work (~0.5 page, three short paragraphs).**
- *VLAs:* RT-2, OpenVLA, π0, SmolVLA (+ SmolVLM backbone
  [marafioti2025smolvlm], LeRobot [cadene2026lerobot], OXE [oneill2024oxe]).
- *RL for VLAs:* RECAP/π*0.6; one online-RL representative (SimpleVLA-RL
  [li2025simplevla] is the best fit — LIBERO-based); advantage/return
  conditioning precursors: Decision Transformer [chen2021decision],
  AWR [peng2019awr].
- *Fast action generation:* Consistency Policy [prasad2024consistency],
  Shortcut Models [frans2024shortcut], MeanFlow [geng2025meanflow],
  SnapFlow [luan2026snapflow].

**3. Method (~1.25 pages).** Four subsections mirroring the code:
- *Architecture:* SmolVLM2 backbone truncated to 16 layers + flow-matching
  action expert [lipman2023flow], action chunk size 20.
- *Co-training with KI:* AR loss on FAST tokens [pertsch2025fast] trains the
  backbone; FM loss trains only the expert (detached prefix / frozen VLM
  during the FM pass) [driess2025knowledge].
- *RECAP:* C51 distributional critic [bellemare2017distributional] over 201
  bins on normalized time-to-completion returns in [-1, 0]; N-step TD
  advantages; per-task 70th-percentile thresholds (**verify — the methods
  doc says 30th, the code computes 70th; see cleanup plan**);
  `<advantage_positive/negative>` conditioning; CFG at inference
  [intelligence2025pistar06].
- *One-step distillation:* zero-init target-time MLP, mixed FM +
  shortcut-consistency loss (alpha=0.5, lambda=0.1), frozen backbone,
  1 NFE at inference [luan2026snapflow].

**4. Experiments (~1.25 pages).** See §4/§5 below for the runs. Contents:
setup (LIBERO 4 suites, episodes/task, CFG sweep); **Table 1** — main
comparison (plain / +co-training / +RECAP across suites); **Table or
Figure 2** — CFG weight sweep; **Table 3** — latency & success vs NFE
(10-step teacher, naive few-step, 1-step distilled).

**5. Conclusion + limitations (~0.25 page).** Limitations to state honestly:
sim-only single benchmark; single seed per config unless more are added;
the frozen-backbone caveat on the plain baseline; expert-mode (Phase-2)
rather than full rollout-based Phase-3 RL where applicable.

## 3. Verified citations (arXiv IDs checked 2026-07-04)

Method dependencies:

| Key | Paper | Ref |
|---|---|---|
| shukor2025smolvla | SmolVLA: A VLA Model for Affordable and Efficient Robotics | arXiv:2506.01844 |
| pertsch2025fast | FAST: Efficient Action Tokenization for VLA Models (RSS 2025) | arXiv:2501.09747 |
| driess2025knowledge | Knowledge **Insulating** VLA Models: Train Fast, Run Fast, Generalize Better | arXiv:2505.23705 |
| intelligence2025pistar06 | π*_0.6: a VLA That Learns From Experience (introduces RECAP) | arXiv:2511.14759 |
| black2024pi0 | π_0: A VLA Flow Model for General Robot Control | arXiv:2410.24164 |
| intelligence2025pi05 | π_0.5: a VLA Model with Open-World Generalization | arXiv:2504.16054 |
| liu2023libero | LIBERO (NeurIPS 2023 D&B) | arXiv:2306.03310 |
| lipman2023flow | Flow Matching for Generative Modeling (ICLR 2023) | arXiv:2210.02747 |
| bellemare2017distributional | A Distributional Perspective on RL / C51 (ICML 2017) | arXiv:1707.06887 |
| marafioti2025smolvlm | SmolVLM (covers SmolVLM2; no separate SmolVLM2 paper) | arXiv:2504.05299 |
| cadene2026lerobot | LeRobot library paper (listed ICLR 2026 — verify acceptance) | arXiv:2602.22818 |
| oneill2024oxe | Open X-Embodiment (ICRA 2024) | arXiv:2310.08864 |
| luan2026snapflow | SnapFlow: One-Step Action Generation for Flow-Matching VLAs (the distillation method implemented here) | arXiv:2604.05656 |

Related work:

| Key | Paper | Ref |
|---|---|---|
| prasad2024consistency | Consistency Policy (RSS 2024) | arXiv:2405.07503 |
| frans2024shortcut | One Step Diffusion via Shortcut Models (ICLR 2025) | arXiv:2410.12557 |
| geng2025meanflow | Mean Flows for One-step Generative Modeling | arXiv:2505.13447 |
| li2025simplevla | SimpleVLA-RL | arXiv:2509.09674 |
| chen2025conrft | ConRFT (alternative to SimpleVLA-RL) | arXiv:2502.05450 |
| chen2021decision | Decision Transformer (NeurIPS 2021) | arXiv:2106.01345 |
| peng2019awr | Advantage-Weighted Regression | arXiv:1910.00177 |
| schmidhuber2019udrl | Upside-Down RL (only if space) | arXiv:1912.02875 |
| brohan2023rt2 | RT-2 (CoRL 2023) | arXiv:2307.15818 |
| kim2024openvla | OpenVLA (CoRL 2024) | arXiv:2406.09246 |

Notes: the KI paper title uses "Insulating" (gerund) — use the exact title
in the bib entry. π*0.6 authorship is team-attributed ("Physical
Intelligence"). SmolVLA, KI, π0, π0.5, SmolVLM, π*0.6, MeanFlow,
SimpleVLA-RL, and SnapFlow are arXiv-only.

## 4. Experiment plan A — three-policy comparison

What is actually being compared (KI is hardwired in
`modeling_smolvla_recap.py`; there is no flag to disable it — see
cleanup plan):

| Run | Advantage cond. | AR/FAST loss | VLM backbone trained? | Status |
|---|---|---|---|---|
| RECAP + co-training (`libero_recap_250000`) | yes, critic-derived | yes | yes (via AR only) | done, has eval |
| Expert-mode + co-training (`expert_libero`) | tokens present, always positive | yes | yes (via AR only) | training |
| "Plain" (`plain_smolvla_libero`) | no | off → backbone frozen | no | training |

**Framing caveat:** the third run is not vanilla SmolVLA — it is "FM expert
on a frozen pretrained backbone" (with `--ar_loss_weight 0` and hardwired
KI, nothing trains the VLM). This still cleanly isolates *"does AR
co-training help?"* since FM is insulated in all three runs. In the paper,
call the ablation "AR co-training", not "KI". Optionally cite published
vanilla SmolVLA LIBERO numbers as an external reference row. A true KI
ablation (FM gradients into the backbone) would require a code change and
another 250k-step run — out of scope for now.

**`libero_object` = 0%:** every RECAP checkpoint/CFG scores 0% on
`libero_object` in `outputs/eval/sweep_summary_pivoted.csv`, while a plain
SmolVLA eval run (2026-06-15) scored 93.3% on the same suite. Working
hypothesis: **training-data issue** (not an eval-config bug). The two
in-flight runs (no-recap frozen-VLM, no-recap + co-training) train on the
same `HuggingFaceVLA/libero` data — evaluating them will confirm or refute
this: if they also score 0% on object, the data path is implicated; if they
score well, the issue is specific to the RECAP training/advantage pipeline.
Treat the object column as unresolved until then.

Steps when a checkpoint completes:

1. Locate the final `.pt` under `outputs/libero_expert/` and
   `outputs/libero_expert_no_ki/` on the cluster (`--resume_from auto`
   naming).
2. Export + migrate with `scripts/export_recap_checkpoint.py` — must pass
   `--num_vlm_layers 16 --action_chunk_size 20
   --dataset_repo_id HuggingFaceVLA/libero` (defaults are wrong: chunk
   defaults to 50), and for the plain run the flag setting
   `use_advantage_conditioning=false` (its state dict has no
   advantage-token embeddings). Match `--n_action_steps` to what the 250k
   evals used (`n_action_steps=1` in the existing exports).
3. Extend the sweep: add both exports to `CHECKPOINTS` in
   `scripts/run_array_task.py` and update the array size in
   `scripts/submit_eval_sweep.sh` (currently `0-47` = 3 ckpts × 4 suites ×
   4 cfg). For the plain checkpoint only run `cfg_weight=0.0` — CFG
   contrasts advantage tokens the model doesn't have.
4. Run `libero_object` early (both new checkpoints) to settle the
   training-data question before burning the full grid.
5. Statistical power: the sweep's 5 episodes/task (50/suite) gives ±13pp
   95% CI at 70% success. Use ≥10 episodes/task (200/suite), ideally 20
   for the headline table; report Wilson binomial CIs. Keep 5-episode runs
   only for the CFG-sweep ablation figure.
6. Compile with `scripts/compile_results.py` →
   `outputs/eval/sweep_summary_pivoted.csv`; wandb project
   `smolvla-recap-eval` has per-task breakdowns.

## 5. Experiment plan B — SnapFlow speed

Current state: quality signal exists on `libero_spatial` only (64.0% /
63.0% at cfg 0 / 1.5 over 100 episodes, vs 66–70% for the 10-step teacher).
The only timing on disk is `eval_info.json:overall.eval_ep_s` (17.5 s vs
23.6 s), which is **confounded**: it includes LIBERO simulation time, and
the two numbers differ mainly because cfg 1.5 doubles the batch. Not
publishable as a latency claim. `outputs/snapflow_migrated` is not on the
local machine — locate it on the cluster first.

Benchmark harness to build (`docs/efficientvla.md` §10 "Evaluation &
calibration protocol" sketches the latency table; the harness itself is not
implemented):

- Small script that loads a migrated policy, feeds a fixed batch of real
  LIBERO observations (batch size 1 — deployment-realistic), and times
  `select_action`/`sample_actions` with `torch.cuda.synchronize()` +
  `time.perf_counter()`: ~20 warmup calls, then ≥100 timed calls; report
  median and IQR.
- Instrument two segments separately: (a) prefix/VLM encode (once per
  chunk, identical for both models) and (b) expert denoising (10 Euler
  steps vs 1). The headline speedup lives in (b); end-to-end alone would
  understate it since the shared VLM prefix dominates.
- Derive: ms per action chunk, effective control rate in Hz (chunk of 20
  actions at 10 Hz LIBERO control), speedup ×.

Conditions matrix:

| Condition | Purpose |
|---|---|
| RECAP teacher, 10 steps, cfg 0 / 1.5 | baseline latency + quality |
| Teacher naively at 1, 2, 4 steps (override `num_steps` in a copy of the exported `config.json`) | shows naive step-skipping collapses success — justifies distillation |
| SnapFlow, 1 step, cfg 0 / 1.5 | the adapted method (cite luan2026snapflow) |
| (Optional) SnapFlow + C2 visual-token pruning (`visual_tokens_keep`) | attacks the prefix segment SnapFlow doesn't touch |

Quality side: run the full 4-suite SnapFlow eval (after the object question
is settled) at the same episode count as the main table. The paper figure
is **success rate vs NFE (or latency)**: teacher@10, teacher@4/2/1
(degrading), SnapFlow@1 (holding).

Hardware: report the eval GPU (H200) explicitly. If the rover deployment
narrative matters, an extra measurement on target/edge hardware (or the
TensorRT path in the separate ros repo) would strengthen the claim —
optional.

## 6. Order of operations

1. Evaluate the two in-flight checkpoints when done (settles both the
   ablation table and the `libero_object` training-data question).
2. Write the SnapFlow training doc (becomes paper method text) — see
   cleanup plan.
3. Build the latency harness; run the NFE/latency matrix.
4. Full 4-suite SnapFlow eval.
5. Write the paper.

Cleanup items (doc fixes, dead code, dedup) are tracked in
`docs/cleanup_plan.md` and can be done alongside.
