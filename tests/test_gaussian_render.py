import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import torch
from GaussianAvatars.render import render_set

def test_render_set_subprocess_calls():
    # Mocking arguments
    dataset = MagicMock()
    dataset.model_path = "/tmp/model"
    dataset.select_camera_id = -1
    
    views = [] # Empty list for testing subprocess calls after loop
    gaussians = MagicMock()
    pipeline = MagicMock()
    background = torch.tensor([0, 0, 0])
    
    with patch("GaussianAvatars.render.subprocess.run") as mock_run, \
         patch("GaussianAvatars.render.makedirs"):
        
        # Call render_set with empty views to skip the loop and go to ffmpeg calls
        render_set(dataset, "test_name", 30000, views, gaussians, pipeline, background, render_mesh=True)
        
        # Verify ffmpeg calls
        assert mock_run.call_count >= 2
        for call in mock_run.call_args_list:
            args_list = call[0][0]
            assert isinstance(args_list, list)
            assert args_list[0] == "ffmpeg"
            assert call[1].get("check") is True
