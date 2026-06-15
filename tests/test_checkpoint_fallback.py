import os
import pytest
from unittest.mock import MagicMock

from lerobot_policy_smolvla_rl.checkpoint_utils import resolve_checkpoints, load_checkpoint

@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    # Create some mock checkpoints
    # state_1000.pt, state_2000.pt, state_3000.pt
    for step in [1000, 2000, 3000]:
        cp_dir = tmp_path / f"state_{step}.pt"
        cp_dir.mkdir()
    return tmp_path

def test_resolve_checkpoints_auto(temp_checkpoint_dir):
    checkpoint_to_try, fallback_checkpoint = resolve_checkpoints("auto", str(temp_checkpoint_dir))
    
    # newest should be state_3000.pt
    assert checkpoint_to_try == os.path.join(str(temp_checkpoint_dir), "state_3000.pt")
    # fallback should be state_2000.pt
    assert fallback_checkpoint == os.path.join(str(temp_checkpoint_dir), "state_2000.pt")

def test_resolve_checkpoints_specific(temp_checkpoint_dir):
    specific_path = os.path.join(str(temp_checkpoint_dir), "state_2000.pt")
    checkpoint_to_try, fallback_checkpoint = resolve_checkpoints(specific_path, str(temp_checkpoint_dir))
    
    assert checkpoint_to_try == specific_path
    # fallback should be state_1000.pt (next older one)
    assert fallback_checkpoint == os.path.join(str(temp_checkpoint_dir), "state_1000.pt")

def test_resolve_checkpoints_no_older(temp_checkpoint_dir):
    specific_path = os.path.join(str(temp_checkpoint_dir), "state_1000.pt")
    checkpoint_to_try, fallback_checkpoint = resolve_checkpoints(specific_path, str(temp_checkpoint_dir))
    
    assert checkpoint_to_try == specific_path
    assert fallback_checkpoint is None

def test_load_checkpoint_success():
    accelerator = MagicMock()
    # Simple success scenario
    loaded_path, step = load_checkpoint(accelerator, "/some/path/state_1000.pt", None)
    
    accelerator.load_state.assert_called_once_with("/some/path/state_1000.pt")
    assert loaded_path == "/some/path/state_1000.pt"
    assert step == 1000

def test_load_checkpoint_fallback():
    accelerator = MagicMock()
    # Mock load_state to fail on the first path but succeed on the fallback path
    def load_state_side_effect(path):
        if path == "/some/path/state_3000.pt":
            raise ValueError("Corrupted checkpoint")
        return None
    
    accelerator.load_state.side_effect = load_state_side_effect
    
    loaded_path, step = load_checkpoint(
        accelerator, 
        "/some/path/state_3000.pt", 
        "/some/path/state_2000.pt"
    )
    
    assert accelerator.load_state.call_count == 2
    assert loaded_path == "/some/path/state_2000.pt"
    assert step == 2000

def test_load_checkpoint_crash():
    accelerator = MagicMock()
    # Mock load_state to fail on both
    accelerator.load_state.side_effect = ValueError("Corrupted checkpoint")
    
    with pytest.raises(ValueError, match="Corrupted checkpoint"):
        load_checkpoint(
            accelerator, 
            "/some/path/state_3000.pt", 
            "/some/path/state_2000.pt"
        )
