#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import torch

# Add src to sys.path to enable importing lerobot_policy_smolvla_rl
sys.path.append(os.path.join(os.getcwd(), "src"))

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
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load LIBERO Dynamically
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

    print(f"Initializing task suite: {args.benchmark_name}...")
    task_suite = benchmark_dict[args.benchmark_name]()
    
    num_tasks = task_suite.get_num_tasks()
    if args.task_id < 0 or args.task_id >= num_tasks:
        raise IndexError(f"Task ID {args.task_id} out of bounds for task suite with {num_tasks} tasks.")

    task = task_suite.get_task(args.task_id)
    task_name = task.name
    instruction = task.language
    print(f"\n==========================================")
    print(f"Task ID: {args.task_id}")
    print(f"Task Name: {task_name}")
    print(f"Language Instruction: '{instruction}'")
    print(f"==========================================\n")

    env = task_suite.get_env(args.task_id)

    # 3. Load Exported LeRobot Policy
    print(f"Loading policy from: {args.policy_path} on device: {args.device}...")
    # Override device in the config before loading if necessary
    policy = SmolVLARECAPPolicy.from_pretrained(args.policy_path, device=args.device)
    policy.eval()

    # Inspect policy config to see what cameras and features it expects
    expected_cams = [k for k in policy.config.input_features.keys() if k.startswith("observation.images.")]
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
    success_count = 0
    
    for ep in range(args.episodes):
        print(f"\n--- Episode {ep + 1}/{args.episodes} ---")
        obs = env.reset()
        policy.reset()
        
        step = 0
        done = False
        success = False

        while step < args.max_steps and not done:
            # Map observations from LIBERO to expected policy batch structure
            batch = {}
            
            # Format visual observation(s)
            # LIBERO provides RGB images as float/uint8 [0-255] under specific keys
            # We normalize to [0, 1] before passing to the policy's prepare_images
            for cam_key in expected_cams:
                # Map expected camera key to LIBERO's obs keys
                if "agentview" in cam_key:
                    img = obs.get("agentview_image")
                elif "wrist" in cam_key or "eye_in_hand" in cam_key:
                    img = obs.get("robot0_eye_in_hand_image")
                else:
                    # fallback to any available image
                    img = obs.get("agentview_image")

                if img is None:
                    raise KeyError(f"Could not find observation image for policy camera feature: {cam_key}")

                # Ensure image is float32 in range [0, 1], shape HxWxC
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                
                # Permute to CxHxW, add batch and sequence dimensions (B=1, S=1)
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).unsqueeze(0).to(args.device)
                batch[cam_key] = img_tensor

            # Format state observation
            # Extract 7-dim robot joint positions + 1-dim gripper position (or similar depending on environment)
            joint_pos = obs.get("robot0_joint_pos")
            gripper_qpos = obs.get("robot0_gripper_qpos")
            
            if joint_pos is not None and gripper_qpos is not None:
                # Concatenate robot joint position and gripper qpos to create state vector
                state = np.concatenate([joint_pos, gripper_qpos])
            else:
                # fallback to raw state if available
                state = obs.get("robot0_joint_pos", np.zeros(7))
            
            # Add batch dimension (B=1)
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(args.device)
            batch["observation.state"] = state_tensor

            # Pass language tokens and masks
            batch["observation.language_tokens"] = lang_tokens
            batch["observation.language_attention_mask"] = lang_masks

            # 5. Policy Inference: select action
            with torch.no_grad():
                # select_action returns a torch.Tensor command of actions
                action = policy.select_action(batch)
                action_np = action.cpu().numpy()

            # 6. Step Simulation
            # LIBERO actions are typically delta position/orientation + gripper qpos
            obs, reward, done, info = env.step(action_np)
            
            # Check for success (LIBERO environments typically set success in info or if reward is > 0)
            if reward > 0.0 or info.get("success", False):
                success = True
                done = True

            step += 1

        if success:
            success_count += 1
            print(f"Result: SUCCESS (in {step} steps)")
        else:
            print(f"Result: FAILED (timed out after {step} steps)")

    success_rate = (success_count / args.episodes) * 100.0
    print(f"\n==========================================")
    print(f"Evaluation Complete on Task: {task_name}")
    print(f"Episodes Evaluated: {args.episodes}")
    print(f"Successful Episodes: {success_count}")
    print(f"Overall Success Rate: {success_rate:.2f}%")
    print(f"==========================================\n")


if __name__ == "__main__":
    main()
