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
- 
