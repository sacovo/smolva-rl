import os

def resolve_checkpoints(resume_from, output_dir):
    """
    Scans the directory for checkpoints and resolves the target checkpoint to load
    and its fallback (the next older one).
    
    Returns (checkpoint_to_try, fallback_checkpoint).
    """
    checkpoints = []
    if os.path.exists(output_dir):
        checkpoints = [
            d
            for d in os.listdir(output_dir)
            if d.startswith("state_") and os.path.isdir(os.path.join(output_dir, d))
        ]
        # Sort checkpoints descending by step number
        try:
            checkpoints = sorted(
                checkpoints,
                key=lambda x: int(x.split("_")[1].split(".")[0]),
                reverse=True
            )
        except Exception as e:
            print(f"Warning: failed to sort checkpoints: {e}")
            checkpoints = []

    checkpoint_to_try = None
    fallback_checkpoint = None

    if resume_from == "auto":
        if checkpoints:
            checkpoint_to_try = os.path.join(output_dir, checkpoints[0])
            if len(checkpoints) > 1:
                fallback_checkpoint = os.path.join(output_dir, checkpoints[1])
    elif resume_from:
        checkpoint_to_try = resume_from
        try:
            specified_dir = os.path.dirname(resume_from)
            specified_base = os.path.basename(resume_from)
            specified_step = int(specified_base.split("_")[1].split(".")[0])
            
            if os.path.exists(specified_dir):
                other_checkpoints = [
                    d
                    for d in os.listdir(specified_dir)
                    if d.startswith("state_") and os.path.isdir(os.path.join(specified_dir, d))
                ]
                other_checkpoints = sorted(
                    other_checkpoints,
                    key=lambda x: int(x.split("_")[1].split(".")[0]),
                    reverse=True
                )
                older_checkpoints = [
                    os.path.join(specified_dir, cp) for cp in other_checkpoints
                    if int(cp.split("_")[1].split(".")[0]) < specified_step
                ]
                if older_checkpoints:
                    fallback_checkpoint = older_checkpoints[0]
        except Exception:
            pass

    return checkpoint_to_try, fallback_checkpoint


def load_checkpoint(accelerator, checkpoint_to_try, fallback_checkpoint=None):
    """
    Attempts to load the checkpoint using accelerator.load_state.
    If it fails and fallback_checkpoint is provided, attempts to load the fallback.
    
    Returns (loaded_checkpoint_path, step) or raises the exception if both fail.
    """
    if not checkpoint_to_try:
        return None, 0

    loaded_path = None
    try:
        print(f"Attempting to load checkpoint from {checkpoint_to_try}")
        accelerator.load_state(checkpoint_to_try)
        loaded_path = checkpoint_to_try
    except Exception as e:
        if fallback_checkpoint:
            print(f"Warning: Failed to load checkpoint {checkpoint_to_try} due to: {e}")
            print(f"Attempting to load next older checkpoint from {fallback_checkpoint}")
            try:
                accelerator.load_state(fallback_checkpoint)
                loaded_path = fallback_checkpoint
            except Exception as e_fallback:
                print(f"Error: Failed to load fallback checkpoint {fallback_checkpoint} due to: {e_fallback}")
                raise e_fallback
        else:
            print(f"Error: Failed to load checkpoint {checkpoint_to_try} and no fallback checkpoint was found.")
            raise e

    step = 0
    if loaded_path:
        try:
            step = int(os.path.basename(loaded_path).split("_")[1].split(".")[0])
            print(f"Resumed training state from {loaded_path} at step {step}")
        except (ValueError, IndexError):
            print(
                f"Resumed training state from {loaded_path}, but could not determine step. Starting from 0."
            )

    return loaded_path, step
