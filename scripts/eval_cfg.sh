#!/bin/bash
set -e

# Target PID of the running lerobot-eval process
# Weights to evaluate
CFG_WEIGHTS=(0.0 0.25 0.5 0.75 1.0 1.5 2.0 3.0)

for cfg in "${CFG_WEIGHTS[@]}"; do
  echo "-------------------------------------------------------"
  echo "Running evaluation with CFG weight = ${cfg}..."
  echo "-------------------------------------------------------"
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

echo "-------------------------------------------------------"
echo "All evaluations finished. Comparing results:"
echo "-------------------------------------------------------"
PYTHONPATH=src uv run python3 scripts/compare_eval_runs.py
