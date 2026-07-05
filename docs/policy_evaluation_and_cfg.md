# Policy Evaluation and Classifier-Free Guidance (CFG) Sweep Guide

This document provides a guide on how to evaluate the trained RECAP / SmolVLA-RL policies inside the LIBERO simulation suites, along with key experimental findings regarding the impact of Classifier-Free Guidance (CFG) weights.

---

## 1. How to Evaluate Policies using LeRobot

Policy evaluation is performed using the `lerobot-eval` command-line tool. It connects the trained VLA policy to the simulated environment, runs rollouts, and logs task success rates.

### Core CLI Command

To run evaluation on a policy checkpoint, execute:

```bash
PYTHONPATH=src uv run lerobot-eval \
  --policy.path="outputs/recap_libero_exported_migrated" \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --env.max_parallel_tasks=1 \
  --policy.device=cuda \
  --policy.cfg_weight=1.5
```

### Parameter Description

* `--policy.path`: Path to the exported policy directory (which must contain the preprocessing/postprocessing statistics and configurations).
* `--env.type`: The simulation environment manager (use `libero`).
* `--env.task`: The task suite to run. Available LIBERO suites:
  - `libero_spatial` (10 spatial tasks)
  - `libero_object` (10 object manipulation tasks)
  - `libero_goal` (10 goal-directed tasks)
  - `libero_10` (10 general tasks)
* `--eval.n_episodes`: Number of episodes to run **per task**. For a suite with 10 tasks, setting `n_episodes=10` runs 100 total episodes.
* `--policy.cfg_weight`: Overrides the default Classifier-Free Guidance weight used by the RECAP advantage conditioning.
* `--policy.device`: Run on GPU (`cuda`) or CPU (`cpu`).

---

## 2. Automating CFG Sweeps

We use a bash script to sweep over different CFG weights and compare their success rates.

### The Sweep Script (`scripts/eval_cfg.sh`)

This script sweeps across different weights on `libero_spatial` (10 episodes per task, 100 episodes per weight) and automatically runs the comparison tool at the end:

```bash
#!/bin/bash
set -e

CFG_WEIGHTS=(0.0 0.25 0.5 0.75 1.0 1.5 2.0 3.0)

for cfg in "${CFG_WEIGHTS[@]}"; do
  echo "Running evaluation with CFG weight = ${cfg}..."
  PYTHONPATH=src uv run lerobot-eval \
    --policy.path=outputs/recap_libero_exported_migrated \
    --env.type=libero \
    --env.task=libero_spatial \
    --eval.batch_size=1 \
    --eval.n_episodes=10 \
    --env.max_parallel_tasks=1 \
    --policy.device=cuda \
    --policy.cfg_weight=${cfg} \
    --job_name=recap_spatial_cfg_${cfg}
done

# Compile results
PYTHONPATH=src uv run python3 scripts/compare_eval_runs.py
```

### Aggregating Results (`scripts/compare_eval_runs.py`)

This script parses all `eval_info.json` outputs in `outputs/eval` and prints a sorted pandas table comparing overall and per-task success rates:

```bash
PYTHONPATH=src uv run python3 scripts/compare_eval_runs.py
```

---

## 3. Experimental Findings: CFG Sweep

We executed a comprehensive CFG sweep on the **LIBERO Spatial Suite** (10 tasks, 10 episodes per task = 100 episodes per CFG weight value) using the exported and migrated RECAP model:

| CFG Weight | Overall Success Rate (%) | Total Episodes | Behavior / Description |
| :---: | :---: | :---: | :--- |
| **`0.0`** | **64.0%** | 100 | Unconditioned baseline (ignores advantage conditioning). |
| **`0.25`** | **63.0%** | 100 | Very weak advantage guidance. |
| **`0.5`** | **61.0%** | 100 | Weak advantage guidance. |
| **`0.75`** | **62.0%** | 100 | Moderate advantage guidance. |
| **`1.0`** | **65.0%** | 100 | Standard RECAP conditional execution (baseline). |
| **`1.5`** | **69.0%** | 100 | **Optimal guidance** (best performance). |
| **`2.0`** | **69.0%** | 100 | **Optimal guidance** (best performance). |
| **`3.0`** | **63.0%** | 100 | Over-guidance (begins to distort actions). |

### Key Takeaways

1. **Optimal Guidance Benefit:** Setting the guidance weight to **`1.5` or `2.0`** yields the highest success rate at **`69.0%`**, providing a **+4% absolute improvement** over standard conditioning (\(w = 1.0\)) and a **+5% improvement** over unconditioned execution (\(w = 0.0\)).
2. **Degradation under Low Weights:** CFG weights below `1.0` (such as `0.25`, `0.5`, and `0.75`) reduce the efficacy of the advantage token, performing slightly worse than the unconditioned baseline.
3. **Over-guidance Distortion:** At \(w = 3.0\), success rate drops to **63.0%**. This is expected in diffusion models, as too high of a guidance weight extrapolates the action vectors beyond the support of the training data distribution, leading to jerky or out-of-bounds movements.

---

## 4. Geometric & Physical Steering Principles

To understand how CFG changes the policy's decisions, we analyzed the denoising trajectory at the vector level:

### 1. High-Dimensional Linear Extrapolation (PCA Space)
At any denoising step \(t\), the policy evaluates the unconditional velocity \(v_{\text{uncond}}\) and conditional velocity \(v_{\text{cond}}\). CFG computes:
\[v_{\text{cfg}} = v_{\text{uncond}} + w \cdot (v_{\text{cond}} - v_{\text{uncond}})\]
Since \(v_{\text{cfg}}\) is a linear combination of two vectors, projecting these high-dimensional velocity vectors (448 dimensions) to 2D using PCA reveals that they lie **exactly** on a straight 1D line. PC1 represents the guidance direction.

### 2. Physical Joint Steering
In physical space, the guidance vector \(\vec{g} = v_{\text{cond}} - v_{\text{uncond}}\) points directly along task-relevant coordinates. 
* In spatial manipulation (descending and picking up objects), the unconditioned policy \(v_{\text{uncond}}\) plans a flat forward sweep. 
* The guidance vector \(\vec{g}\) points downwards (negative Z).
* By setting \(w = 2.0\), the resulting velocity \(v_{\text{cfg}}\) is rotated downwards, producing a steep and decisive descent towards the target. Conversely, negative guidance (\(w = -1.0\)) reverses this, commanding the arm to ascend.

### 3. Zoomed-Out Attractor Fields
Evaluating the vector field over a wide coordinate grid (\([-20.0, 20.0]\)) centered on the active trajectory shows the global flow dynamics:
* **Unconditioned Field:** Has no vertical correction flow; if the state is perturbed off-track, the arrows do not bend back to the center.
* **Conditioned/Guided Field:** A clear **attractor corridor** emerges. The vector field curves inwards, acting as a strong corrective force that funnels the arm's trajectory back onto the correct path even if large perturbations occur.

---

## 5. SnapFlow Performance & Complexity Profile

SnapFlow is a distilled version of the policy that executes the denoising trajectory in a single step (1 step) instead of 10 steps. We evaluated SnapFlow on the **LIBERO Spatial Suite** (100 episodes per sweep) and profiled its latency, throughput, and analytical GFLOPs complexity.

### 1. Success Rate Comparison
We compare the task success rates of SnapFlow vs. the original Flow Matching policy:

| Policy | CFG Weight | Success Rate (%) | Total Episodes | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Regular Flow Matching** | `0.0` | **64.0%** | 100 | Baseline Flow Matching (10 steps) |
| **SnapFlow** | `0.0` | **64.0%** | 100 | **100% success rate retention** |
| **Regular Flow Matching** | `1.5` | **69.0%** | 100 | Baseline with optimal CFG guidance |
| **SnapFlow** | `1.5` | **63.0%** | 100 | Distilled with CFG (small 6.0% drop) |

### 2. GPU Latency & Throughput Profile
Measured on GPU (CUDA) using single-frame inference benchmarks:

| Inference Mode | Latency (ms) | Throughput (Hz) | Speedup |
| :--- | :---: | :---: | :---: |
| **Regular Flow Matching (No CFG, 10 steps)** | 217.51 | 4.60 | Baseline |
| **SnapFlow (No CFG, 1 step)** | 82.54 | 12.12 | **2.64x Speedup** |
| **Regular Flow Matching (CFG=1.5, 10 steps)** | 262.03 | 3.82 | Baseline |
| **SnapFlow (CFG=1.5, 1 step)** | 181.48 | 5.51 | **1.44x Speedup** |

### 3. Analytical FLOP Complexity (GFLOPs)
Based on VLM (350M parameters) and Expert (98.2M parameters) model dimensions:

| Inference Mode | VLM FLOPs | Expert FLOPs | Total FLOPs | Speedup / Reduction |
| :--- | :---: | :---: | :---: | :---: |
| **Regular Flow Matching (No CFG, 10 steps)** | 488.04 | 102.37 | 590.41 | Baseline |
| **SnapFlow (No CFG, 1 step)** | 488.04 | 10.24 | 498.28 | **1.18x reduction** |
| **Regular Flow Matching (CFG=1.5, 10 steps)** | 976.09 | 204.74 | 1180.83 | Baseline |
| **SnapFlow (CFG=1.5, 1 step)** | 976.09 | 20.47 | 996.56 | **1.18x reduction** |

### 4. Key Performance Insights

1. **Wall-Clock vs. FLOP Mismatch**: While the GFLOPs reduction is a modest **1.18x** (due to the large VLM prefix pass running once per batch), the actual wall-clock speedup is **2.64x** (without CFG). This is because executing 10 sequential expert steps is heavily bound by **memory bandwidth** (loading 98.2M parameters 10 times vs. once) and **sequential CUDA kernel launch overhead**.
2. **CFG Overhead (2-pass)**: When using CFG, the batch size is doubled (from 1 to 2) to evaluate both conditioned and unconditioned fields. This increases the VLM prefix pass latency from ~67 ms to ~172 ms, making the VLM prefix pass the dominant bottleneck and reducing SnapFlow's relative speedup to **1.44x**.

