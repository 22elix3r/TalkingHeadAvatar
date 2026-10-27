import json
from pathlib import Path
import subprocess
import sys

from talkinghead_v2.replay import run_replay


def test_replay_is_deterministic_and_checks_full_synthetic_path():
    first = run_replay()
    second = run_replay()
    assert first == second
    assert first["status"] == "passed"
    assert not first["actual_device_playback"] and not first["performance_or_quality_claim"]
    assert first["completed"]["output_samples"] == 2419
    assert first["completed"]["eof_tail_samples"] == 35
    assert first["completed"]["last_callback_samples"] == 17
    assert first["completed"]["reference_parity"]
    assert first["cancelled"]["ack"]["discarded_queue_samples"] > 0
    assert first["stale_callbacks_rejected"] == 6
    assert first["queue"]["peak_samples"] <= first["queue"]["max_samples"]


def test_installed_replay_cli_writes_only_explicit_synthetic_report(tmp_path):
    entrypoint = str(Path(sys.executable).parent / "tha")
    output = tmp_path / "reports/p01.json"
    result = subprocess.run([entrypoint, "replay", "--json", "--report", str(output)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == json.loads(output.read_text())
    assert json.loads(result.stdout)["boundary"] == "synthetic_in_process_scheduling_only"
