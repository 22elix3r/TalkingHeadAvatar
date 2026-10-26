import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from talkinghead_v2 import cli
from talkinghead_v2.doctor import Check, DoctorReport


def test_installed_entrypoint_help_and_offline_doctor_from_outside_repo(tmp_path):
    # Absolute entrypoint and an empty PATH ensure this smoke needs no GPU utilities.
    entrypoint = str(Path(sys.executable).parent / "tha")
    env = dict(os.environ, PATH="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               XDG_CACHE_HOME=str(tmp_path / "cache"))
    env.pop("PYTHONPATH", None)
    for args in (["--help"], ["doctor", "--help"], ["--version"]):
        result = subprocess.run([entrypoint, *args], cwd=tmp_path, env=env,
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert result.stdout
    result = subprocess.run([entrypoint, "doctor", "--profile", "local", "--json"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["profile"] == "local"
    assert report["status"] == "warning"
    assert not list(tmp_path.iterdir())  # Diagnostics created no cache/output directories.


def test_imports_do_not_load_inference_initialize_cuda_or_access_network(tmp_path):
    script = '''
import ctypes
import importlib.abc
import socket
import subprocess
import sys

def forbidden(*args, **kwargs):
    raise AssertionError("import attempted runtime initialization")

class NoInference(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "torch", "transformers", "huggingface_hub", "numpy", "tensorflow", "onnxruntime"
        }:
            forbidden()

sys.meta_path.insert(0, NoInference())
socket.socket.connect = forbidden
socket.create_connection = forbidden
subprocess.run = forbidden
subprocess.Popen = forbidden
ctypes.CDLL = forbidden
import talkinghead_v2
import talkinghead_v2.config
import talkinghead_v2.cli
import talkinghead_v2.audio.clock
import talkinghead_v2.audio.queue
import talkinghead_v2.audio.resample
import talkinghead_v2.audio.motion
import talkinghead_v2.session.epochs
import talkinghead_v2.replay
import talkinghead_v2.providers.tts.native
import talkinghead_v2.providers.tts.reference
import talkinghead_v2.audio.playback
assert talkinghead_v2.__version__ == "0.1.0"
'''
    result = subprocess.run([sys.executable, "-I", "-c", script], cwd=tmp_path,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("status, expected", [("ok", 0), ("warning", 0), ("error", 1)])
def test_cli_status_and_json(status, expected, monkeypatch, capsys):
    monkeypatch.setattr(cli, "diagnose", lambda _: DoctorReport("local", (Check("fixture", status, "synthetic"),)))
    assert cli.main(["doctor", "--json"]) == expected
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_cli_human_report(monkeypatch, capsys):
    monkeypatch.setattr(cli, "diagnose", lambda _: DoctorReport("local", (Check("fixture", "warning", "synthetic"),)))
    assert cli.main(["doctor"]) == 0
    text = capsys.readouterr().out
    assert "[WARNING] fixture" in text
    assert "readiness are unqualified" in text


@pytest.mark.parametrize("args", [["doctor", "--profile", "server"], ["doctor", "--unknown"]])
def test_cli_rejects_unsupported_arguments(args):
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 2


def test_cli_reports_config_errors_without_traceback(tmp_path, capsys):
    config = tmp_path / "config.toml"
    config.write_text("min_free_bytes = true")
    with pytest.raises(SystemExit) as exc:
        cli.main(["doctor", "--config", str(config)])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "min_free_bytes must be a nonnegative integer" in error
    assert "Traceback" not in error


def test_cli_diagnostic_error_exit_code(tmp_path, monkeypatch, capsys):
    # Real config -> diagnostics -> JSON -> exit status, without any host GPU tools.
    monkeypatch.setattr("talkinghead_v2.doctor.shutil.which", lambda _: None)
    config = tmp_path / "config.toml"
    config.write_text('min_free_bytes = 9223372036854775807\noutput_dir = "absent/output"')
    assert cli.main(["doctor", "--config", str(config), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "error"
    assert not (tmp_path / "absent").exists()
