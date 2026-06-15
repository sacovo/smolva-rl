#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import torch

# Add src to sys.path to enable importing lerobot_policy_smolvla_rl
sys.path.append(os.path.join(os.getcwd(), "src"))

from lerobot.policies.pretrained import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot_policy_smolvla_rl import SmolVLARECAPPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned SmolVLA-RECAP Policy in LIBERO Simulation")
    parser.add_argument(
        "--policy_path",
        type=str,
        required=True,
        help="Path to the exported LeRobot policy directory containing model.safetensors and config.json",
    )
    parser.add_argument(
        "--benchmark_name",
        type=str,
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO task suite benchmark name",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=0,
        help="Task ID to evaluate in the benchmark suite (0-indexed)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=600,
        help="Maximum simulation steps per episode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce stdout logging and suppress library warnings/progress bars",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for parallel evaluation rollouts",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.quiet:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        import warnings
        warnings.filterwarnings("ignore")
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("lerobot").setLevel(logging.ERROR)
        logging.getLogger("robosuite").setLevel(logging.ERROR)
        logging.getLogger("urdf_parser_py").setLevel(logging.ERROR)

    # 1. Load LIBERO Dynamically
    if not args.quiet:
        print("Loading LIBERO benchmark suite...")
    try:
        import libero
        from libero.libero import benchmark
    except ImportError as e:
        print(f"\nError: Could not import 'libero'. Please ensure that LIBERO is installed in your python environment.")
        print("To install LIBERO, visit: https://github.com/Lifelong-Robot-Learning/LIBERO")
        sys.exit(1)

    # 2. Initialize LIBERO Environment and Task
    benchmark_dict = benchmark.get_benchmark_dict()
    if args.benchmark_name not in benchmark_dict:
        raise ValueError(f"Unknown benchmark suite: {args.benchmark_name}. Available: {list(benchmark_dict.keys())}")

    if not args.quiet:
        print(f"Initializing task suite: {args.benchmark_name}...")
    task_suite = benchmark_dict[args.benchmark_name]()
    
    num_tasks = task_suite.get_num_tasks()
    if args.task_id < 0 or args.task_id >= num_tasks:
        raise IndexError(f"Task ID {args.task_id} out of bounds for task suite with {num_tasks} tasks.")

    task = task_suite.get_task(args.task_id)
    task_name = task.name
    instruction = task.language
    if not args.quiet:
        print(f"\n==========================================")
        print(f"Task ID: {args.task_id}")
        print(f"Task Name: {task_name}")
        print(f"Language Instruction: '{instruction}'")
        print(f"==========================================\n")

    from libero.libero.envs import OffScreenRenderEnv
    task_bddl_file = task_suite.get_task_bddl_file_path(args.task_id)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }
    init_states = task_suite.get_task_init_states(args.task_id)

    # 3. Load Exported LeRobot Policy
    if not args.quiet:
        print(f"Loading policy from: {args.policy_path} on device: {args.device}...")
    # Load config to determine policy class
    config = PreTrainedConfig.from_pretrained(args.policy_path)
    if config.type == "smolvla_recap":
        policy = SmolVLARECAPPolicy.from_pretrained(args.policy_path, device=args.device)
    elif config.type == "smolvla":
        policy = SmolVLAPolicy.from_pretrained(args.policy_path, device=args.device)
    else:
        raise ValueError(f"Unknown policy type in config: {config.type}")
    policy.eval()

    # Inspect policy config to see what cameras and features it expects
    expected_cams = [k for k in policy.config.input_features.keys() if k.startswith("observation.images.")]
    if not args.quiet:
        print(f"Policy expected camera features: {expected_cams}")

    # Set up tokenizer prompt with LIBERO's language instruction
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    
    # Standard prompt structure expected by SmolVLA
    prompt = f"User: {instruction}\nAssistant: "
    tokens = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    lang_tokens = tokens["input_ids"].to(args.device) # shape (1, seq_len)
    lang_masks = tokens["attention_mask"].bool().to(args.device) # shape (1, seq_len)

    # 4. Rollout Evaluation Loop
    batch_size = min(args.batch_size, args.episodes)
    num_batches = (args.episodes + batch_size - 1) // batch_size
    success_count = 0

    for b in range(num_batches):
        current_batch_size = min(batch_size, args.episodes - b * batch_size)
        if not args.quiet:
            print(f"\n--- Batch {b + 1}/{num_batches} (Size: {current_batch_size}) ---")

        # Create environments for this batch
        batch_envs = []
        for i in range(current_batch_size):
            env_i = OffScreenRenderEnv(**env_args)
            env_i.seed(b * batch_size + i)
            batch_envs.append(env_i)

        # Reset all environments and set initial states
        obs_list = []
        for i, env_i in enumerate(batch_envs):
            env_i.reset()
            ep_idx = b * batch_size + i
            env_i.set_init_state(init_states[ep_idx % len(init_states)])
            obs_i = env_i.reset()
            obs_list.append(obs_i)

        policy.reset()

        active = [True] * current_batch_size
        success_flags = [False] * current_batch_size
        steps = [0] * current_batch_size

        while any(active):
            batch = {}

            # 1. Format visual observations
            for cam_key in expected_cams:
                cam_tensors = []
                for i in range(current_batch_size):
                    obs_i = obs_list[i]
                    if "agentview" in cam_key or cam_key == "observation.images.image":
                        img = obs_i.get("agentview_image")
                    elif "wrist" in cam_key or "eye_in_hand" in cam_key or cam_key == "observation.images.image2":
                        img = obs_i.get("robot0_eye_in_hand_image")
                    else:
                        img = obs_i.get("agentview_image")

                    if img is None:
                        raise KeyError(f"Could not find observation image for policy camera feature: {cam_key}")

                    if img.dtype == np.uint8:
                        img = img.astype(np.float32) / 255.0

                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).unsqueeze(0).to(args.device)
                    cam_tensors.append(img_tensor)

                batch[cam_key] = torch.cat(cam_tensors, dim=0)

            # 2. Format state observations
            state_tensors = []
            for i in range(current_batch_size):
                obs_i = obs_list[i]
                joint_pos = obs_i.get("robot0_joint_pos")
                gripper_qpos = obs_i.get("robot0_gripper_qpos")

                if joint_pos is not None and gripper_qpos is not None:
                    state_dim = policy.config.input_features["observation.state"].shape[0]
                    if state_dim == 8 and len(gripper_qpos) == 2:
                        gripper_val = gripper_qpos[:1]
                    else:
                        gripper_val = gripper_qpos
                    state = np.concatenate([joint_pos, gripper_val])
                else:
                    state = obs_i.get("robot0_joint_pos", np.zeros(7))

                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(args.device)
                state_tensors.append(state_tensor)

            batch["observation.state"] = torch.cat(state_tensors, dim=0)

            # 3. Format language observations
            batch["observation.language.tokens"] = lang_tokens.repeat(current_batch_size, 1)
            batch["observation.language.attention_mask"] = lang_masks.repeat(current_batch_size, 1)

            # 4. Policy Inference: select action
            with torch.no_grad():
                actions = policy.select_action(batch)
                if actions.ndim == 1:
                    actions = actions.unsqueeze(0)
                actions_np = actions.cpu().numpy()

            # 5. Step simulation for active environments
            for i in range(current_batch_size):
                if not active[i]:
                    continue

                obs_i, reward_i, done_i, info_i = batch_envs[i].step(actions_np[i])
                obs_list[i] = obs_i
                steps[i] += 1

                if reward_i > 0.0 or info_i.get("success", False):
                    success_flags[i] = True
                    active[i] = False
                elif steps[i] >= args.max_steps or done_i:
                    active[i] = False

        # Cleanup environments and print results for this batch
        for i, env_i in enumerate(batch_envs):
            ep_idx = b * batch_size + i
            if success_flags[i]:
                success_count += 1
                if not args.quiet:
                    print(f"Result for Episode {ep_idx + 1}: SUCCESS (in {steps[i]} steps)")
            else:
                if not args.quiet:
                    print(f"Result for Episode {ep_idx + 1}: FAILED (timed out after {steps[i]} steps)")
            
            try:
                env_i.close()
            except Exception:
                pass

    success_rate = (success_count / args.episodes) * 100.0
    print(f"\n==========================================")
    print(f"Evaluation Complete on Task: {task_name}")
    print(f"Episodes Evaluated: {args.episodes}")
    print(f"Successful Episodes: {success_count}")
    print(f"Overall Success Rate: {success_rate:.2f}%")
    print(f"==========================================\n")


if __name__ == "__main__":
    main()
