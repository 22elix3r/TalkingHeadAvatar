from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from talkinghead_v2.config import LocalConfig
from talkinghead_v2 import doctor


@pytest.fixture
def config(tmp_path):
    return LocalConfig(artifact_cache_dir=tmp_path / "cache", output_dir=tmp_path / "new/output",
                       min_free_bytes=0)


def test_missing_optional_tools_and_artifacts_are_nonfatal_and_read_only(config, monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _: None)
    report = doctor.diagnose(config)
    checks = {check.name: check for check in report.checks}
    assert report.status == "warning"
    assert checks["python"].status == "ok"
    for name in ("gpu_driver", "cuda_toolkit", "nvenc", "artifact_cache"):
        assert checks[name].status == "warning"
    assert checks["output_space"].status == "ok"
    assert not config.output_dir.parent.exists()
    assert not config.artifact_cache_dir.exists()
    assert report.to_dict()["schema_version"] == 1


@pytest.mark.parametrize("failure", ["timeout", "oserror", "nonzero"])
def test_optional_tool_failures_are_warnings(config, monkeypatch, failure):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: f"/tools/{name}")

    def run(argv, **kwargs):
        assert kwargs["timeout"] == 3
        assert "shell" not in kwargs
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 3)
        if failure == "oserror":
            raise PermissionError("denied")
        return subprocess.CompletedProcess(argv, 1, "", "private tool output")

    monkeypatch.setattr(doctor.subprocess, "run", run)
    report = doctor.diagnose(config)
    assert report.status == "warning"
    for check in report.checks:
        assert "private tool output" not in check.detail
        if check.name in {"gpu_driver", "cuda_toolkit", "nvenc"}:
            assert check.status == "warning"


def test_tool_presence_is_not_claimed_as_runtime_qualification(config, monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: f"/tools/{name}")

    def run(argv, **kwargs):
        output = {
            "nvidia-smi": "Example GPU, 580.1, 16376\n",
            "nvcc": "Cuda compilation tools, release 12.8, V12.8.0\n",
            "ffmpeg": " V....D h264_nvenc NVIDIA NVENC H.264 encoder\n V....D hevc_nvenc encoder\n",
        }[Path(argv[0]).name]
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(doctor.subprocess, "run", run)
    checks = {check.name: check for check in doctor.diagnose(config).checks}
    for name in ("gpu_driver", "cuda_toolkit", "nvenc"):
        assert checks[name].status == "ok"
    assert "no CUDA workload tested" in checks["gpu_driver"].detail
    assert "runtime not qualified" in checks["cuda_toolkit"].detail
    assert "device access/encoding not qualified" in checks["nvenc"].detail


def test_empty_successful_tool_output_does_not_imply_availability(config, monkeypatch):
    monkeypatch.setattr(doctor, "_tool", lambda *args: ("", ""))
    checks = {check.name: check for check in doctor.diagnose(config).checks}
    for name in ("gpu_driver", "cuda_toolkit", "nvenc"):
        assert checks[name].status == "warning"


def test_cache_presence_does_not_open_or_validate_weights(config, monkeypatch):
    assert doctor._cache_check(config.artifact_cache_dir).status == "warning"
    config.artifact_cache_dir.mkdir()
    assert doctor._cache_check(config.artifact_cache_dir).status == "warning"
    (config.artifact_cache_dir / "untrusted.pkl").write_bytes(b"not a trusted checkpoint")

    def refuse_open(*args, **kwargs):
        raise AssertionError("doctor must not open model files")

    monkeypatch.setattr(Path, "open", refuse_open)
    check = doctor._cache_check(config.artifact_cache_dir)
    assert check.status == "ok"
    assert "content not loaded or verified" in check.detail
    assert "consent" in check.detail


def test_output_capacity_threshold_and_permissions(config, monkeypatch):
    monkeypatch.setattr(doctor.shutil, "disk_usage", lambda _: SimpleNamespace(free=100))
    assert doctor._output_check(config.output_dir, 100).status == "ok"
    assert doctor._output_check(config.output_dir, 101).status == "error"
    monkeypatch.setattr(doctor.os, "access", lambda *args: False)
    assert "not writable/searchable" in doctor._output_check(config.output_dir, 0).detail
    assert not config.output_dir.parent.exists()


def test_files_in_directory_positions_fail(config, tmp_path):
    file = tmp_path / "file"
    file.write_text("not a directory")
    assert doctor._cache_check(file).status == "error"
    assert doctor._output_check(file, 0).status == "error"
    assert doctor._output_check(file / "child", 0).status == "error"


def test_filesystem_errors_are_reported(config, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("denied")

    config.artifact_cache_dir.mkdir()
    monkeypatch.setattr(doctor.os, "scandir", denied)
    assert doctor._cache_check(config.artifact_cache_dir).status == "error"
    monkeypatch.setattr(doctor.shutil, "disk_usage", denied)
    assert doctor._output_check(config.output_dir, 0).status == "error"
