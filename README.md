# SmolVLA with RL and FAST Tokenizer

Modifies SmolVLA to work with RL (RECAP), also uses the FAST Tokenizer to Co-Train the VLM Backbone on large scale robotics data.


## Implementation

Dataset collection:

- Use LeRobot dataset from here: https://huggingface.co/collections/IPEC-COMMUNITY/openx-lerobot
- There are differing inputs/outputs (joint position/velocity, ee position in cartesian space, ...). According to this [discussion](https://github.com/Physical-Intelligence/openpi/discussions/302) it does not really matter
- Maybe add names of the input and output values to the text prompt to condition it to generate the correct actions?
- 8 dimensions for out/input seems to be fine

Used datasets:

https://huggingface.co/datasets/IPEC-COMMUNITY/droid_lerobot
https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot
https://huggingface.co/datasets/IPEC-COMMUNITY/bc_z_lerobot
https://huggingface.co/datasets/IPEC-COMMUNITY/dobbe_lerobot
https://huggingface.co/datasets/IPEC-COMMUNITY/stanford_hydra_dataset_lerobot
https://huggingface.co/datasets/IPEC-COMMUNITY/berkeley_autolab_ur5_lerobot

Consider:
https://huggingface.co/datasets/IPEC-COMMUNITY/utaustin_mutex_lerobot



Critic network:
- Bins normalized returns between \([-1.0, 0.0]\) to classify time-to-completion.

## RECAP Training Stages

The RECAP training pipeline consists of three phases:

1. **Phase 1: Pretraining** (Pretrained VLA using large datasets and critic advantage values).
2. **Phase 2: Supervised Finetuning (Imitation Finetuning)**:
   - Finetunes the VLA on task-specific demonstration data.
   - Run the script with the `--expert_mode` flag to bypass the critic and treat all demonstration frames as having a positive advantage (`advantage_bool = True`).
   - Example command:
     ```bash
     python src/lerobot_policy_smolvla_rl/train_recap.py \
         --dataset_repo_id <dataset_id> \
         --expert_mode \
         --steps 100000
     ```
3. **Phase 3: Rollout and Policy Enhancement**:
   - Collects rollout data (e.g. using `scripts/record_eval.py`) with expert/human interventions. Success outcomes are saved in the episode-level metadata (`success` column in `meta/episodes.parquet`).
   - **Critic Finetuning**: Run `train_critic.py`. Failed episodes are penalized with a terminal penalty of `-C_FAIL` (normalized to push them to the lowest value bin `0`).
   - **Advantage Pre-computation**: Run `compute_thresholds.py` with the trained critic to pre-compute the N-step TD advantages and per-task thresholds (paper Appendix A-F: `A_t = Σ r_{t..t+N-1} + V(o_{t+N}) − V(o_t)`, with `--advantage_horizon N` defaulting to 50). This writes `task_advantages_<repo>.npy` and `task_thresholds_<repo>.json` into `--save_dir`.
     ```bash
     python src/lerobot_policy_smolvla_rl/compute_thresholds.py \
         --dataset_repo_id <dataset_id> \
         --critic_checkpoint <critic.pt> \
         --save_dir outputs/recap_phase1
     ```
   - **Policy Enhancement**: Run `train_recap.py` (without `--expert_mode`) pointing `--save_dir` (or `--precomputed_advantages`/`--thresholds_path`) at those files. The trainer only consumes the pre-computed advantages — it never runs the critic itself, and exits with an error if the files are missing. Any frames with human/expert interventions (`batch["intervention"] == True`) override the threshold and receive a positive advantage signal (`advantage_bool = True`).

