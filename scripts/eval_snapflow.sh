#!/bin/bash
set -e

echo "-------------------------------------------------------"
echo "Running SnapFlow evaluation with CFG weight = 0.0 (No CFG)..."
echo "-------------------------------------------------------"
PYTHONPATH=src uv run lerobot-eval \
  --policy.path=outputs/snapflow_migrated \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --env.max_parallel_tasks=1 \
  --policy.device=cuda \
  --policy.cfg_weight=0.0 \
  --job_name=snapflow_spatial_cfg_0.0

echo "-------------------------------------------------------"
echo "Running SnapFlow evaluation with CFG weight = 1.5 (With CFG)..."
echo "-------------------------------------------------------"
PYTHONPATH=src uv run lerobot-eval \
  --policy.path=outputs/snapflow_migrated \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --env.max_parallel_tasks=1 \
  --policy.device=cuda \
  --policy.cfg_weight=1.5 \
  --job_name=snapflow_spatial_cfg_1.5

echo "-------------------------------------------------------"
echo "All evaluations finished. Comparing results:"
echo "-------------------------------------------------------"
PYTHONPATH=src uv run python3 scripts/compare_eval_runs.py
