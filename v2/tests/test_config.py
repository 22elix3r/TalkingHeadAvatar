from pathlib import Path

import pytest

from talkinghead_v2.config import ConfigError, LocalConfig, load_config


def test_paths_use_config_directory_not_working_directory(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = config_dir / "local.toml"
    config.write_text('artifact_cache_dir = "../cache"\noutput_dir = "../output"\n')
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    result = load_config(config)
    assert result.artifact_cache_dir == tmp_path / "cache"
    assert result.output_dir == tmp_path / "output"
    assert not result.output_dir.exists()
    assert not result.artifact_cache_dir.exists()


def test_default_paths_and_home_expansion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache-home"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = load_config()
    assert config.artifact_cache_dir == tmp_path / "cache-home/talkinghead-v2/artifacts"
    assert config.output_dir == tmp_path / ".tha-v2/output"
    source = tmp_path / "config.toml"
    source.write_text('output_dir = "~/output"\n')
    assert load_config(source).output_dir == tmp_path / "home/output"


@pytest.mark.parametrize("contents", [
    'profile = "server"', 'unknown = true', 'min_free_bytes = -1',
    'min_free_bytes = true', 'min_free_bytes = 1.5', 'min_free_bytes = "100"',
    'output_dir = ""', 'output_dir = "   "', 'output_dir = []',
    'artifact_cache_dir = 1', '[output_dir]\npath = "here"', 'invalid TOML',
    'output_dir = "\\u0000"',
])
def test_invalid_config_fails_clearly(tmp_path, contents):
    source = tmp_path / "bad.toml"
    source.write_text(contents)
    with pytest.raises(ConfigError):
        load_config(source)


def test_missing_config_does_not_fall_back_to_defaults(tmp_path):
    with pytest.raises(ConfigError, match="cannot read config"):
        load_config(tmp_path / "missing.toml")


def test_invalid_profile_and_direct_types():
    with pytest.raises(ConfigError, match="local"):
        load_config(profile="server")
    with pytest.raises(ConfigError, match="pathlib.Path"):
        LocalConfig(output_dir="output")
    assert LocalConfig(output_dir=Path("output"), min_free_bytes=0).min_free_bytes == 0
