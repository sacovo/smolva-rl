#!/bin/bash
# Helper script to export and migrate 150k and 250k checkpoints on the HPC login node.
set -e

# Make sure we are in the repo root
cd "$(dirname "$0")/.."

echo "Starting export for 150k checkpoint..."
PYTHONPATH=src uv run scripts/export_recap_checkpoint.py \
  --checkpoint_path outputs/libero_recap_150000.pt \
  --output_dir outputs/libero_recap_150000 \
  --dataset_repo_id HuggingFaceVLA/libero \
  --action_chunk_size 20 \
  --num_vlm_layers 16

echo "Starting export for 250k checkpoint..."
PYTHONPATH=src uv run scripts/export_recap_checkpoint.py \
  --checkpoint_path outputs/libero_recap_250000.pt \
  --output_dir outputs/libero_recap_250000 \
  --dataset_repo_id HuggingFaceVLA/libero \
  --action_chunk_size 20 \
  --num_vlm_layers 16

echo "All checkpoints exported and migrated successfully!"
