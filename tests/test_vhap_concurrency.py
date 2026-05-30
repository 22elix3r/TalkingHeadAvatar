import pytest
import torch
from unittest.mock import patch, MagicMock
from VHAP.vhap.export_as_nerf_dataset import NeRFDatasetWriter

def test_vhap_exporter_concurrency_instantiation():
    # Mocking the dataloader to return a few items
    mock_dataloader = [
        {'timestep_index': 0, 'camera_index': 0, 'extrinsic': torch.eye(3, 4), 'intrinsic': torch.eye(3), 'rgb': torch.zeros(10, 10, 3), 'timestep_id': '0', 'timestep_index_original': 0, 'camera_id': '0'},
        {'timestep_index': 1, 'camera_index': 0, 'extrinsic': torch.eye(3, 4), 'intrinsic': torch.eye(3), 'rgb': torch.zeros(10, 10, 3), 'timestep_id': '1', 'timestep_index_original': 1, 'camera_id': '1'}
    ]
    
    with patch("VHAP.vhap.export_as_nerf_dataset.DataLoader", return_value=mock_dataloader), \
         patch("VHAP.vhap.export_as_nerf_dataset.concurrent.futures.ThreadPoolExecutor") as mock_executor_cls, \
         patch("VHAP.vhap.export_as_nerf_dataset.Path"), \
         patch("VHAP.vhap.export_as_nerf_dataset.import_module"), \
         patch("VHAP.vhap.export_as_nerf_dataset.tyro.to_yaml"):
        
        # We need a mock cfg and src_folder
        cfg = MagicMock()
        src_folder = MagicMock()
        tgt_folder = MagicMock()
        
        exporter = NeRFDatasetWriter(cfg, src_folder, tgt_folder)
        
        # Mocking write_data and wait to avoid file IO and hanging
        with patch("VHAP.vhap.export_as_nerf_dataset.write_json"), \
             patch("VHAP.vhap.export_as_nerf_dataset.write_data"), \
             patch("VHAP.vhap.export_as_nerf_dataset.concurrent.futures.wait"):
             try:
                 exporter.write()
             except Exception:
                 pass # We might hit errors because we didn't mock everything, but we care about the executor calls
        
        # Verify ThreadPoolExecutor was instantiated
        assert mock_executor_cls.call_count == 1
