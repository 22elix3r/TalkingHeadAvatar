"""Bounded, read-only host diagnostics; no inference libraries or model reads."""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Literal

from talkinghead_v2 import __version__
from talkinghead_v2.config import LocalConfig

Status = Literal["ok", "warning", "error"]
TOOL_TIMEOUT_SECONDS = 3


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str


@dataclass(frozen=True)
class DoctorReport:
    profile: str
    checks: tuple[Check, ...]

    @property
    def status(self) -> Status:
        if any(check.status == "error" for check in self.checks):
            return "error"
        if any(check.status == "warning" for check in self.checks):
            return "warning"
        return "ok"

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": 1, "profile": self.profile, "status": self.status,
                "checks": [asdict(check) for check in self.checks]}


def _tool(name: str, args: tuple[str, ...]) -> tuple[str | None, str]:
    executable = shutil.which(name)
    if executable is None:
        return None, f"{name} is not on PATH (optional for P00 CPU work)"
    try:
        result = subprocess.run(
            [executable, *args], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=TOOL_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, f"{name} timed out after {TOOL_TIMEOUT_SECONDS}s"
    except OSError as exc:
        return None, f"{name} could not run: {exc}"
    if result.returncode:
        # Do not dump arbitrary tool output or local configuration into diagnostics.
        return None, f"{name} exited {result.returncode}; availability not established"
    return result.stdout, ""


def _gpu_check() -> Check:
    output, problem = _tool("nvidia-smi", (
        "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits",
    ))
    if output is None:
        return Check("gpu_driver", "warning", problem)
    devices = [line.strip() for line in output.splitlines() if line.strip()]
    if not devices:
        return Check("gpu_driver", "warning", "nvidia-smi reported no devices")
    return Check("gpu_driver", "ok", "; ".join(devices) +
                 " [name, driver, total MiB]; driver query only, no CUDA workload tested")


def _cuda_check() -> Check:
    output, problem = _tool("nvcc", ("--version",))
    if output is None:
        return Check("cuda_toolkit", "warning", problem +
                     "; a missing compiler does not prove CUDA inference is unavailable")
    version = next((line.strip() for line in output.splitlines() if "release" in line), "")
    if not version:
        return Check("cuda_toolkit", "warning", "nvcc ran but no toolkit version was recognized")
    return Check("cuda_toolkit", "ok", version + "; compiler presence only, runtime not qualified")


def _encoder_check() -> Check:
    output, problem = _tool("ffmpeg", ("-hide_banner", "-encoders"))
    if output is None:
        return Check("nvenc", "warning", problem)
    names = {parts[1] for line in output.splitlines() if len(parts := line.split()) >= 2}
    found = sorted(names & {"h264_nvenc", "hevc_nvenc", "av1_nvenc"})
    if not found:
        return Check("nvenc", "warning", "FFmpeg lists no NVENC encoders; no encode attempted")
    return Check("nvenc", "ok", ", ".join(found) +
                 "; compiled support only, device access/encoding not qualified")


def _cache_check(path: Path) -> Check:
    try:
        if not path.exists():
            return Check("artifact_cache", "warning", f"Absent: {path}; no artifacts provisioned here")
        if not path.is_dir():
            return Check("artifact_cache", "error", f"Cache path is not a directory: {path}")
        # Only inspect directory entries, never deserialize or open an artifact.
        with os.scandir(path) as entries:
            present = next(entries, None) is not None
        if not present:
            return Check("artifact_cache", "warning", f"Empty: {path}; no artifacts provisioned here")
        return Check("artifact_cache", "ok", f"Entries present: {path}; content not loaded or verified. "
                     "Presence does not establish Qwen readiness, hashes, rights or consent")
    except OSError as exc:
        return Check("artifact_cache", "error", f"Cannot inspect cache {path}: {exc}")


def _output_check(path: Path, min_free_bytes: int) -> Check:
    try:
        ancestor = path
        while not ancestor.exists():
            if ancestor == ancestor.parent:
                return Check("output_space", "error", f"No existing ancestor for {path}")
            ancestor = ancestor.parent
        if not ancestor.is_dir():
            return Check("output_space", "error", f"Output path/ancestor is not a directory: {ancestor}")
        if not os.access(ancestor, os.W_OK | os.X_OK):
            return Check("output_space", "error", f"Output ancestor is not writable/searchable: {ancestor}")
        free = shutil.disk_usage(ancestor).free
        status: Status = "ok" if free >= min_free_bytes else "error"
        return Check("output_space", status, f"{path}: {free} bytes free; require {min_free_bytes}; "
                     f"permissions checked at {ancestor}. No write performed or directories created")
    except OSError as exc:
        return Check("output_space", "error", f"Cannot inspect output {path}: {exc}")


def diagnose(config: LocalConfig) -> DoctorReport:
    """Inspect this process and optional tools. Warnings do not block the foundation."""
    python_ok = sys.version_info[:2] == (3, 12)
    isolated = sys.prefix != sys.base_prefix
    return DoctorReport(config.profile, (
        Check("python", "ok" if python_ok else "error",
              f"{platform.python_version()} at {sys.executable}; supported: Python 3.12"),
        Check("environment", "ok" if isolated else "warning",
              f"{'Virtual environment' if isolated else 'No virtual environment'}; "
              f"prefix={sys.prefix}; talkinghead-v2={__version__}; "
              f"{platform.system()} {platform.machine()}"),
        _gpu_check(), _cuda_check(), _encoder_check(),
        _cache_check(config.artifact_cache_dir),
        _output_check(config.output_dir, config.min_free_bytes),
    ))
