import pytest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from GaussianAvatars.flame_model.flame import FlameHead

def test_flame_head_safetensors_loading():
    # Mocking safetensors load_file
    mock_data = {
        "v_template": torch.randn(5023, 3),
        "shapedirs": torch.randn(5023, 3, 400),
        "posedirs": torch.randn(36, 15069),
        "J_regressor": torch.randn(5, 5023),
        "kintree_table": torch.zeros(2, 5),
        "weights": torch.randn(5023, 5),
        "f": np.zeros((9976, 3), dtype=np.int32)
    }
    
    with patch("GaussianAvatars.flame_model.flame.safetensors_load", return_value=mock_data) as mock_load, \
         patch("GaussianAvatars.flame_model.flame.np.load", return_value=MagicMock()), \
         patch("GaussianAvatars.flame_model.flame.load_obj", return_value=(None, MagicMock(), MagicMock())), \
         patch("GaussianAvatars.flame_model.flame.FlameMask"):
        
        # Initialize FlameHead with a .safetensors path
        # Note: We need to mock more things because __init__ does a lot
        try:
            FlameHead(shape_params=100, expr_params=50, flame_model_path="model.safetensors", add_teeth=False)
        except Exception as e:
            # We expect some failure downstream because we didn't mock everything perfectly,
            # but we want to verify the load call.
            pass
        
        mock_load.assert_called_with("model.safetensors")

def test_flame_head_pickle_fallback():
    with patch("GaussianAvatars.flame_model.flame.pickle.load") as mock_pickle_load, \
         patch("builtins.open", MagicMock()), \
         patch("GaussianAvatars.flame_model.flame.np.load", return_value=MagicMock()), \
         patch("GaussianAvatars.flame_model.flame.load_obj", return_value=(None, MagicMock(), MagicMock())), \
         patch("GaussianAvatars.flame_model.flame.FlameMask"):
        
        try:
            FlameHead(shape_params=100, expr_params=50, flame_model_path="model.pkl", add_teeth=False)
        except:
            pass
            
        assert mock_pickle_load.called
