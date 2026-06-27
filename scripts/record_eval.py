#!/usr/bin/env python3
import argparse
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import einops
import cv2

# Add src to python path to enable loading local modules
sys.path.append(os.path.join(os.getcwd(), "src"))

# Explicitly import the custom policy to ensure it is registered with Draccus/LeRobot choice registries
try:
    from lerobot_policy_smolvla_rl import SmolVLARECAPPolicy, SmolVLARECAPConfig
except ImportError:
    pass

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.envs.utils import preprocess_observation, add_envs_task
from lerobot.utils.constants import ACTION, REWARD, DONE
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.device_utils import get_safe_torch_device

def parse_args():
    parser = argparse.ArgumentParser(description="Record simulation evaluation runs to a new LeRobot dataset with gamepad/keyboard intervention")
    parser.add_argument(
        "--policy_path",
        type=str,
        required=False,
        default=None,
        help="Path to the policy checkpoint folder (containing config.json and model.safetensors)"
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        default=False,
        help="Run in sandbox mode for manual practice. Policy is not loaded and data is not recorded."
    )
    parser.add_argument(
        "--env_task",
        type=str,
        default="libero_spatial",
        help="Task suite to run (e.g., libero_spatial, libero_object, libero_goal, libero_10)"
    )
    parser.add_argument(
        "--output_repo_id",
        type=str,
        default="outputs/libero_eval_recorded",
        help="Output dataset repo ID or local path"
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=10,
        help="Number of episodes to record per task"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for running the policy (e.g., cuda or cpu)"
    )
    parser.add_argument(
        "--use_videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Encode and store visual frames as mp4 videos (default: True)"
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="Optional specific task ID to record (e.g. 0). If None, records all tasks in the suite."
    )
    parser.add_argument(
        "--cfg_weight",
        type=float,
        default=1.5,
        help="Classifier-Free Guidance weight for the RECAP policy"
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=20,
        help="Number of action steps to execute"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run environment in headless mode without window display"
    )
    return parser.parse_args()

def batch_obs(obs):
    if isinstance(obs, dict):
        return {k: batch_obs(v) for k, v in obs.items()}
    elif isinstance(obs, np.ndarray):
        return np.expand_dims(obs, axis=0)
    else:
        return obs

class GamepadController:
    def __init__(self):
        self.js = None
        try:
            import pygame
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self.js = pygame.joystick.Joystick(0)
                self.js.init()
                print(f"\n[GAMEPAD] Initialized gamepad: {self.js.get_name()}")
                print("[GAMEPAD] Controls:")
                print("  Left Stick  : translation X and Y (forward/backward, left/right)")
                print("  L1 / R1     : translation Z (down/up)")
                print("  Right Stick : rotation pitch and yaw")
                print("  D-pad Left/Right : rotation roll (+/-)")
                print("  Cross (Btn 0)    : toggle gripper state")
                print("  Options (Btn 9) / Share (Btn 8) / PS (Btn 10) : toggle Policy vs Human control")
                self.manual_mode = False
                self.gripper_state = -1.0
            else:
                print("\n[GAMEPAD] No gamepad detected. Gamepad control disabled.")
        except Exception as e:
            print(f"\n[GAMEPAD] Failed to initialize pygame/joystick: {e}")

    def is_available(self):
        return self.js is not None

    def get_action(self):
        if not self.is_available():
            return None, False
            
        import pygame
        pygame.event.pump()
        
        # Read events to handle buttons (edge-triggered)
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button in [8, 9, 10]: # Options, Share, or PS
                    self.manual_mode = not self.manual_mode
                    print(f"\n[GAMEPAD] Toggled manual mode: {self.manual_mode}")
                elif event.button == 0: # Cross
                    self.gripper_state = -self.gripper_state
                    print(f"[GAMEPAD] Gripper toggled via gamepad: {'OPEN' if self.gripper_state > 0 else 'CLOSED'}")
                    
        if not self.manual_mode:
            return None, False
            
        # Left Stick: Y-axis (1) is forward/backward translation, X-axis (0) is left/right translation
        dx = -self.js.get_axis(1) # Up on stick is negative Y, so flip
        dy = -self.js.get_axis(0) # Left on stick is negative X
        
        # Deadzone filter
        if abs(dx) < 0.1: dx = 0.0
        if abs(dy) < 0.1: dy = 0.0
        
        # Scale speed
        dx *= 0.3
        dy *= 0.3
        
        # Vertical translation (Z): use L1 (Button 4) for down, R1 (Button 5) for up
        dz = 0.0
        if self.js.get_button(4): # L1
            dz = -0.3
        elif self.js.get_button(5): # R1
            dz = 0.3
            
        # Roll rotation: use D-pad (hat 0)
        droll = 0.0
        hat = self.js.get_hat(0) if self.js.get_numhats() > 0 else (0, 0)
        if hat[0] != 0:
            droll = hat[0] * 0.3
            
        # Pitch/Yaw rotation: use Right Stick. X-axis (3) is yaw, Y-axis (4) is pitch
        dyaw = self.js.get_axis(3)
        dpitch = -self.js.get_axis(4)
        
        if abs(dyaw) < 0.1: dyaw = 0.0
        if abs(dpitch) < 0.1: dpitch = 0.0
        
        dyaw *= 0.3
        dpitch *= 0.3
        
        action = np.zeros(7, dtype=np.float32)
        action[0] = dx
        action[1] = dy
        action[2] = dz
        action[3] = droll
        action[4] = dpitch
        action[5] = dyaw
        action[6] = self.gripper_state
        
        return action, True

def run_sandbox(args):
    # 1. Create the environment
    print(f"Making environment for task suite: {args.env_task}")
    from lerobot.envs.configs import LiberoEnv
    env_cfg = LiberoEnv(task=args.env_task)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    
    # envs is a dict: {suite_name: {task_id: SyncVectorEnv}}
    suite_name = list(envs.keys())[0]
    task_envs_dict = envs[suite_name]
    if args.task_id is not None:
        if args.task_id not in task_envs_dict:
            raise ValueError(f"Task ID {args.task_id} not found in suite {suite_name}. Available: {list(task_envs_dict.keys())}")
        task_ids = [args.task_id]
    else:
        task_ids = list(task_envs_dict.keys())
    print(f"Practice tasks in suite: {task_ids}")
    
    # 2. Initialize Gamepad and GUI
    gamepad = GamepadController()
    if gamepad.is_available():
        gamepad.manual_mode = True
        
    headless = False
    window_name = "LIBERO Practice Sandbox - Press R to reset, ESC to quit"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 512, 512)
        print("\n=======================================================")
        print("SANDBOX PRACTICE MODE ACTIVE")
        print("=======================================================")
        print("X11 Graphics display detected. Practice window is active.")
        print("Controls:")
        print("  W / S : forward / backward translation (+/- dx)")
        print("  A / D : left / right translation (+/- dy)")
        print("  Q / E : up / down translation (+/- dz)")
        print("  U / O : roll rotation (+/- droll)")
        print("  I / K : pitch rotation (+/- dpitch)")
        print("  J / L : yaw rotation (+/- dyaw)")
        print("  SPACE : toggle gripper open/closed")
        print("  R (Keyboard) / Triangle (Gamepad): Reset environment to practice again")
        print("  ESC : Quit sandbox mode")
    except Exception:
        print("X11 Graphics display not detected. Sandbox mode requires an X11 display to visualize the simulation and capture input.")
        return
        
    for task_id in task_ids:
        vec_env = task_envs_dict[task_id]
        task_env = vec_env.envs[0]
        print(f"\n=================== PRACTICE TASK: {task_id} ===================")
        
        # Get task instruction
        temp_obs, _ = task_env.reset()
        temp_obs_dict = preprocess_observation(batch_obs(temp_obs))
        temp_obs_dict["task_index"] = torch.tensor([task_id], dtype=torch.long)
        temp_obs_dict = add_envs_task(vec_env, temp_obs_dict)
        task_instruction = temp_obs_dict["task"][0]
        print(f"Instruction: '{task_instruction}'")
        
        quit_sandbox = False
        while not quit_sandbox:
            print("\nResetting environment...")
            obs, info = task_env.reset()
            done = False
            step = 0
            max_steps = task_env.spec.max_episode_steps if hasattr(task_env.spec, "max_episode_steps") else 280
            gripper_state = -1.0
            if gamepad.is_available():
                gamepad.manual_mode = True
                gamepad.gripper_state = -1.0
                
            reset_sandbox = False
            while not done and step < max_steps:
                obs_dict = preprocess_observation(batch_obs(obs))
                img = obs_dict["observation.images.image"][0].cpu().numpy().transpose(1, 2, 0)
                if img.dtype != np.uint8:
                    img = (img * 255.0).astype(np.uint8)
                
                # Show GUI
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                display_img = img_bgr.copy()
                display_img = cv2.resize(display_img, (512, 512))
                
                cv2.putText(display_img, "PRACTICE SANDBOX", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                cv2.putText(display_img, f"Step: {step}/{max_steps}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(display_img, f"Press R to Reset, ESC to Quit", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow(window_name, display_img)
                
                key = cv2.waitKey(100) & 0xFF
                if key == 27: # ESC
                    quit_sandbox = True
                    break
                if key == ord('r'):
                    reset_sandbox = True
                    break
                    
                # Check gamepad buttons (e.g. Triangle for reset)
                if gamepad.is_available():
                    if gamepad.js.get_button(3): # Triangle/Y button
                        reset_sandbox = True
                        print("[GAMEPAD] Reset triggered via Triangle button.")
                        break
                
                # Read manual action
                action_numpy = None
                if gamepad.is_available():
                    action_gamepad, is_gamepad_active = gamepad.get_action()
                    if is_gamepad_active:
                        action_numpy = action_gamepad
                        gripper_state = gamepad.gripper_state
                
                if action_numpy is None:
                    action_numpy = np.zeros(7, dtype=np.float32)
                    action_numpy[6] = gripper_state
                    
                    if key == ord('w'): action_numpy[0] = 0.3
                    elif key == ord('s'): action_numpy[0] = -0.3
                    if key == ord('a'): action_numpy[1] = 0.3
                    elif key == ord('d'): action_numpy[1] = -0.3
                    if key == ord('q'): action_numpy[2] = 0.3
                    elif key == ord('e'): action_numpy[2] = -0.3
                    
                    if key == ord('u'): action_numpy[3] = 0.3
                    elif key == ord('o'): action_numpy[3] = -0.3
                    if key == ord('i'): action_numpy[4] = 0.3
                    elif key == ord('k'): action_numpy[4] = -0.3
                    if key == ord('j'): action_numpy[5] = 0.3
                    elif key == ord('l'): action_numpy[5] = -0.3
                    
                    if key == ord(' '):
                        gripper_state = -gripper_state
                        action_numpy[6] = gripper_state
                        print(f"[KEYBOARD] Gripper toggled: {'OPEN' if gripper_state > 0 else 'CLOSED'}")
                      
                # Step environment
                next_obs, reward, terminated, truncated, info = task_env.step(action_numpy)
                done = terminated or truncated
                success = info.get("is_success", False)
                if success:
                    print("Task SUCCESS achieved!")
                    
                obs = next_obs
                step += 1
                
            if reset_sandbox:
                continue
            if done:
                print(f"Practice episode finished. Success: {success}")
                # Wait 1.5 seconds before resetting so they can see the final state
                cv2.waitKey(1500)
                
        if quit_sandbox:
            break
            
    cv2.destroyAllWindows()
    print("Sandbox practice mode finished.")

def main():
    register_third_party_plugins()
    args = parse_args()
    
    if args.sandbox:
        if args.policy_path is not None:
            print("[WARNING] Running in sandbox practice mode. --policy_path will be ignored.")
        run_sandbox(args)
        return
        
    if args.policy_path is None:
        raise ValueError("Policy path is required when not running in sandbox mode. Please specify --policy_path.")
        
    # 1. Load original dataset specs to copy features specification
    print("Loading metadata of original LIBERO dataset to clone features specifications...")
    orig_meta = LeRobotDatasetMetadata("HuggingFaceVLA/libero")
    
    # Map image features to video features if use_videos is enabled
    features = {}
    for key, val in orig_meta.features.items():
        val_copy = dict(val)
        if args.use_videos and val_copy.get("dtype") == "image":
            val_copy["dtype"] = "video"
        features[key] = val_copy
    
    # Ensure reward, success, and intervention features are defined in our dataset
    if "reward" not in features:
        features["reward"] = {"dtype": "float32", "shape": (1,), "names": None}
    if "success" not in features:
        features["success"] = {"dtype": "bool", "shape": (1,), "names": None}
    if "intervention" not in features:
        features["intervention"] = {"dtype": "bool", "shape": (1,), "names": None}
        
    # Determine the root directory and clean up if it already exists
    import shutil
    if args.output_repo_id.startswith("outputs/") or "/" in args.output_repo_id or Path(args.output_repo_id).is_absolute():
        root_dir = Path(args.output_repo_id).resolve()
        repo_id = root_dir.name
    else:
        repo_id = args.output_repo_id
        from lerobot.utils.constants import HF_LEROBOT_HOME
        root_dir = (Path(HF_LEROBOT_HOME) / repo_id).resolve()
        
    if root_dir.exists():
        print(f"Cleaning up existing dataset folder at {root_dir}...")
        shutil.rmtree(root_dir)
        
    # 2. Create the target LeRobot dataset writer
    print(f"Creating output dataset at: {root_dir}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=10, # LIBERO runs at 10 Hz
        features=features,
        use_videos=args.use_videos,
        root=root_dir,
    )
    
    # 3. Create the environment
    # Use batch_size=1 to simplify sequential recording of single episodes
    print(f"Making environment for task suite: {args.env_task}")
    from lerobot.envs.configs import LiberoEnv
    env_cfg = LiberoEnv(task=args.env_task)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    
    # envs is a dict: {suite_name: {task_id: SyncVectorEnv}}
    suite_name = list(envs.keys())[0]
    task_envs_dict = envs[suite_name]
    if args.task_id is not None:
        if args.task_id not in task_envs_dict:
            raise ValueError(f"Task ID {args.task_id} not found in suite {suite_name}. Available: {list(task_envs_dict.keys())}")
        task_ids = [args.task_id]
    else:
        task_ids = list(task_envs_dict.keys())
    print(f"Recording tasks in suite: {task_ids}")
    
    # 4. Make Policy
    print(f"Loading policy from: {args.policy_path}...")
    from lerobot.configs.policies import PreTrainedConfig
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = Path(args.policy_path)
    policy_cfg.device = args.device
    
    if getattr(args, "cfg_weight", None) is not None:
        policy_cfg.cfg_weight = args.cfg_weight
        print(f"Overriding policy cfg_weight to: {policy_cfg.cfg_weight}")
    if getattr(args, "n_action_steps", None) is not None:
        policy_cfg.n_action_steps = args.n_action_steps
        print(f"Overriding policy n_action_steps to: {policy_cfg.n_action_steps}")
        
    policy = make_policy(policy_cfg, env_cfg=env_cfg)
    policy.eval()
    
    # Create preprocessor and postprocessors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    
    # 5. Initialize Gamepad and GUI
    gamepad = GamepadController()
    
    headless = args.headless
    window_name = "LIBERO Interaction - Press ENTER to toggle Policy vs Human control"
    if not headless:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 512, 512)
            print("X11 Graphics display detected. Live window visualization and keyboard manual intervention are ENABLED.")
            print("Keyboard Controls (when Human Control is active):")
            print("  W / S : forward / backward translation (+/- dx)")
            print("  A / D : left / right translation (+/- dy)")
            print("  Q / E : up / down translation (+/- dz)")
            print("  U / O : roll rotation (+/- droll)")
            print("  I / K : pitch rotation (+/- dpitch)")
            print("  J / L : yaw rotation (+/- dyaw)")
            print("  SPACE : toggle gripper open/closed")
            print("  ENTER : toggle Policy Control vs Human Control")
        except Exception:
            print("X11 Graphics display not detected. Running in HEADLESS mode (Policy/Gamepad only, keyboard interface disabled).")
            headless = True
        
    # 6. Recording Loop
    episode_idx = 0
    episode_successes = []
    
    for task_id in task_ids:
        vec_env = task_envs_dict[task_id]
        # Get the underlying single environment to step step-by-step without auto-resetting
        task_env = vec_env.envs[0]
        print(f"\n=================== TASK: {task_id} ===================")
        
        # Get task instruction
        temp_obs, _ = task_env.reset()
        temp_obs_dict = preprocess_observation(batch_obs(temp_obs))
        temp_obs_dict["task_index"] = torch.tensor([task_id], dtype=torch.long)
        temp_obs_dict = add_envs_task(vec_env, temp_obs_dict)
        task_instruction = temp_obs_dict["task"][0]
        print(f"Instruction: '{task_instruction}'")
        
        for ep in range(args.n_episodes):
            print(f"\nRecording Episode {ep+1}/{args.n_episodes}...")
            policy.reset()
            obs, info = task_env.reset()
            
            done = False
            step = 0
            max_steps = task_env.spec.max_episode_steps if hasattr(task_env.spec, "max_episode_steps") else 280
            
            # Temporary buffers to collect this episode's data
            ep_frames = []
            success = False
            
            # Intervention variables
            manual_mode = False
            gripper_state = -1.0 # -1.0 means closed, 1.0 means open (LIBERO convention)
            if gamepad.is_available():
                gamepad.manual_mode = False
                gamepad.gripper_state = -1.0
            
            was_manual = False
            while not done and step < max_steps:
                # Convert raw observations to LeRobot format
                obs_dict = preprocess_observation(batch_obs(obs))
                obs_dict["task_index"] = torch.tensor([task_id], dtype=torch.long)
                obs_dict = add_envs_task(vec_env, obs_dict)
                
                # Apply preprocessors
                policy_obs = env_preprocessor(obs_dict)
                
                # Extract raw (unnormalized) observations for recording and visualization
                img = obs_dict["observation.images.image"][0].cpu().numpy().transpose(1, 2, 0)
                img2 = obs_dict["observation.images.image2"][0].cpu().numpy().transpose(1, 2, 0)
                # State is processed by env_preprocessor (LiberoProcessorStep) and is stored in policy_obs
                state = policy_obs["observation.state"][0].cpu().numpy()
                
                policy_obs = preprocessor(policy_obs)
                
                # Convert images from [0, 1] float back to [0, 255] uint8
                if img.dtype != np.uint8:
                    img = (img * 255.0).astype(np.uint8)
                if img2.dtype != np.uint8:
                    img2 = (img2 * 255.0).astype(np.uint8)
                    
                # Handle GUI display and manual control if not headless
                key = -1
                if not headless:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    display_img = img_bgr.copy()
                    display_img = cv2.resize(display_img, (512, 512))
                    
                    is_manual = manual_mode or (gamepad.is_available() and gamepad.manual_mode)
                    mode_text = "HUMAN CONTROL" if is_manual else "POLICY CONTROL"
                    color = (0, 0, 255) if is_manual else (0, 255, 0)
                    cv2.putText(display_img, mode_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    cv2.putText(display_img, f"Step: {step}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imshow(window_name, display_img)
                    
                    # Wait for key press (100ms when human control is active to give time for input, 1ms otherwise)
                    key = cv2.waitKey(100 if is_manual else 1) & 0xFF
                    
                    if key == 13: # Enter key to toggle control mode
                        manual_mode = not manual_mode
                        print(f"\n[KEYBOARD] Toggled manual mode: {manual_mode}")
                        key = -1
                
                # Check transition from manual to policy control
                is_currently_manual = manual_mode or (gamepad.is_available() and gamepad.manual_mode)
                if was_manual and not is_currently_manual:
                    print("\n[TRANSITION] Switching from Human to Policy control. Resetting policy history.")
                    policy.reset()
                was_manual = is_currently_manual

                # Determine action based on mode
                action_numpy = None
                
                # 1. Try to read action from gamepad
                if gamepad.is_available():
                    action_gamepad, is_gamepad_active = gamepad.get_action()
                    if is_gamepad_active:
                        action_numpy = action_gamepad
                        manual_mode = True # sync keyboard mode state
                        gripper_state = gamepad.gripper_state
                
                # 2. Try to read action from keyboard if manual mode is active and gamepad is not taking control
                if action_numpy is None:
                    if manual_mode:
                        action_numpy = np.zeros(7, dtype=np.float32)
                        action_numpy[6] = gripper_state
                        
                        # Handle translation key mappings
                        if key == ord('w'): action_numpy[0] = 0.3
                        elif key == ord('s'): action_numpy[0] = -0.3
                        if key == ord('a'): action_numpy[1] = 0.3
                        elif key == ord('d'): action_numpy[1] = -0.3
                        if key == ord('q'): action_numpy[2] = 0.3
                        elif key == ord('e'): action_numpy[2] = -0.3
                        
                        # Handle rotation key mappings
                        if key == ord('u'): action_numpy[3] = 0.3
                        elif key == ord('o'): action_numpy[3] = -0.3
                        if key == ord('i'): action_numpy[4] = 0.3
                        elif key == ord('k'): action_numpy[4] = -0.3
                        if key == ord('j'): action_numpy[5] = 0.3
                        elif key == ord('l'): action_numpy[5] = -0.3
                        
                        # Handle gripper toggle (Space)
                        if key == ord(' '):
                            gripper_state = -gripper_state
                            action_numpy[6] = gripper_state
                            print(f"[KEYBOARD] Gripper toggled: {'OPEN' if gripper_state > 0 else 'CLOSED'}")
                    else:
                        # 3. Policy inference
                        with torch.inference_mode():
                            action = policy.select_action(policy_obs)
                        action = postprocessor(action)
                        
                        # Convert action back for environment execution
                        action_transition = {ACTION: action}
                        action_transition = env_postprocessor(action_transition)
                        action_numpy = action_transition[ACTION][0].cpu().numpy()
                        
                        # Sync gripper state
                        gripper_state = float(action_numpy[6])
                        if gamepad.is_available():
                            gamepad.gripper_state = gripper_state
                
                # Step environment
                next_obs, reward, terminated, truncated, info = task_env.step(action_numpy)
                done = terminated or truncated
                
                success = info.get("is_success", False)
                is_currently_manual = manual_mode or (gamepad.is_available() and gamepad.manual_mode)
                
                frame = {
                    "observation.images.image": img,
                    "observation.images.image2": img2,
                    "observation.state": state,
                    "action": action_numpy,
                    "reward": np.array([reward], dtype=np.float32),
                    "success": np.array([success], dtype=np.bool_),
                    "intervention": np.array([is_currently_manual], dtype=np.bool_),
                    "task": task_instruction,
                }
                
                ep_frames.append(frame)
                
                obs = next_obs
                step += 1
            
            # Print outcome
            outcome = "SUCCESS" if success else "FAILURE"
            print(f"Episode {ep+1} finished. Length: {step} steps. Outcome: {outcome}")
            
            # Append all frames of the episode to the dataset
            for frame in ep_frames:
                frame["success"] = np.array([success], dtype=np.bool_)
                dataset.add_frame(frame)
                
            dataset.save_episode()
            episode_idx += 1
            episode_successes.append(success)
            
    # Finalize the dataset and cleanup GUI
    dataset.finalize()
    if not headless:
        cv2.destroyAllWindows()
        
    # Append success column to episodes parquet file(s)
    episodes_dir = Path(root_dir) / "meta" / "episodes"
    episodes_files = []
    if episodes_dir.exists():
        episodes_files = list(episodes_dir.glob("**/*.parquet"))
    else:
        # Fallback to single file if using older LeRobot version
        single_path = Path(root_dir) / "meta" / "episodes.parquet"
        if single_path.exists():
            episodes_files = [single_path]
            
    if episodes_files:
        import pyarrow.parquet as pq
        import pyarrow as pa
        print(f"Injecting success metadata into episodes files: {episodes_files}...")
        for ep_file in episodes_files:
            table = pq.read_table(ep_file)
            if "episode_index" in table.column_names:
                indices = table["episode_index"].to_pylist()
                file_successes = [episode_successes[idx] for idx in indices]
            else:
                file_successes = episode_successes[:len(table)]
                
            if "success" in table.column_names:
                success_idx = table.column_names.index("success")
                table = table.set_column(success_idx, "success", pa.array(file_successes, type=pa.bool_()))
            else:
                table = table.append_column("success", pa.array(file_successes, type=pa.bool_()))
            pq.write_table(table, ep_file)
        print("Successfully injected success metadata!")
        
    print(f"\n=======================================================")
    print(f"Successfully recorded and finalized dataset!")
    print(f"Total episodes saved: {episode_idx}")
    print(f"Recorded dataset saved to: {root_dir}")
    print(f"=======================================================")
    
if __name__ == "__main__":
    main()
