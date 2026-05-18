import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt
import os

def main():
    repo_id = "lerobot/berkeley_fanuc_manipulation"
    dataset = LeRobotDataset(repo_id, episodes=[0, 1, 2, 3, 4, 5])
    
    output_dir = "research/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    for ep_idx in range(5):
        # Filter for episode
        frames = [f for f in dataset if f["episode_index"] == ep_idx]
        if not frames:
            continue
            
        states = torch.stack([f["observation.state"] for f in frames])
        print(f"Episode {ep_idx} length: {len(frames)}")
        print(f"Start state: {states[0]}")
        print(f"Mid state:   {states[len(frames)//2]}")
        print(f"End state (last 10 frames, dim 7): {states[-10:, 7]}")
        # Also check if it was closed before
        print(f"Mid-to-end state (dim 7): {states[len(frames)//2:len(frames)//2+10, 7]}")
        # Plot all 8 dimensions of state
        plt.figure(figsize=(12, 6))
        for i in range(states.shape[1]):
            plt.plot(states[:, i], label=f"dim_{i}")
        
        plt.title(f"State Dimensions - Episode {ep_idx}")
        plt.xlabel("Frame Index")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"ep_{ep_idx}_state.png"))
        plt.close()
        
        print(f"Saved state plot for episode {ep_idx}")

if __name__ == "__main__":
    main()
