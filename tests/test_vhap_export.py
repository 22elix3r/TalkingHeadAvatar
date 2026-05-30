import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
from VHAP.vhap.export_as_nerf_dataset import load_config

def test_load_config_safe_yaml(tmp_path):
    # Setup a mock config.yml
    config_dir = tmp_path / "config_dir"
    config_dir.mkdir()
    config_file = config_dir / "config.yml"
    
    config_content = """
    model_path: "some/path"
    sh_degree: 3
    """
    config_file.write_text(config_content)
    
    # Run load_config
    folder, cfg = load_config(config_dir)
    
    # Assertions
    assert folder == config_dir
    assert cfg["model_path"] == "some/path"
    assert cfg["sh_degree"] == 3

def test_load_config_rejects_unsafe_yaml(tmp_path):
    # Setup a mock config.yml with unsafe content
    config_dir = tmp_path / "unsafe_config_dir"
    config_dir.mkdir()
    config_file = config_dir / "config.yml"
    
    # Unsafe YAML that tries to execute code using !!python/object/apply
    unsafe_content = "!!python/object/apply:os.system ['echo pwned']"
    config_file.write_text(unsafe_content)
    
    # Run load_config and expect failure
    with pytest.raises(yaml.constructor.ConstructorError):
        load_config(config_dir)


def test_load_config_supports_vhap_tagged_yaml(tmp_path):
    config_dir = tmp_path / "tagged_config_dir"
    config_dir.mkdir()
    config_file = config_dir / "config.yml"
    config_file.write_text(
        """
!!python/object:vhap.config.base.BaseTrackingConfig
data: !!python/object:vhap.config.base.DataConfig
  root_folder: !!python/object/apply:pathlib.PosixPath
  - /tmp/my_subject
  sequence: video
  _target: vhap.data.video_dataset.VideoDataset
model: !!python/object:vhap.config.base.ModelConfig
  n_shape: 300
  n_expr: 100
"""
    )

    _, cfg = load_config(config_dir)
    assert isinstance(cfg, dict)
    assert cfg["data"]["sequence"] == "video"
    assert Path(cfg["data"]["root_folder"]) == Path("/tmp/my_subject")
    assert cfg["model"]["n_shape"] == 300
