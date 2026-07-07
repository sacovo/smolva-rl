#!/bin/bash
set -e

# CFG weights to evaluate: 0.0 (no CFG) vs 1.5 (with CFG)
CFG_WEIGHTS=(0.0 1.5)

for cfg in "${CFG_WEIGHTS[@]}"; do
  echo "-------------------------------------------------------"
  echo "Running SnapFlow evaluation with CFG weight = ${cfg}..."
  echo "-------------------------------------------------------"
  PYTHONPATH=src uv run lerobot-eval \
    --policy.path=outputs/snapflow_migrated \
    --env.type=libero \
    --env.task=libero_spatial \
    --eval.batch_size=1 \
    --eval.n_episodes=10 \
    --env.max_parallel_tasks=1 \
    --policy.device=cuda \
    --policy.cfg_weight=${cfg} \
    --job_name=snapflow_spatial_cfg_${cfg}
done

echo "-------------------------------------------------------"
echo "All evaluations finished. Comparing results:"
echo "-------------------------------------------------------"
PYTHONPATH=src uv run python3 scripts/compare_eval_runs.py
