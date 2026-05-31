#!/usr/bin/env python3
"""Read-only VHAP tracking progress and ETA monitor."""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TIME_RE = re.compile(r"\[(?P<time>\d{2}/\d{2} \d{2}:\d{2}:\d{2})")
TRACK_RE = re.compile(
    r"\[train-(?P<stage>[a-z_]+)\]\s+timestep\s+(?P<timestep>\d+)\s+step\s+(?P<step>\d+):"
)
START_RE = re.compile(r"Start sequential tracking FLAME in (?P<total>\d+) frames")
EPOCH_RE = re.compile(r"EPOCH\s+(?P<epoch>\d+)\s*/\s*(?P<total>\d+)")
CHECKPOINT_RE = re.compile(r"Saved restart checkpoint after (?P<frames>\d+) frames")
ERROR_RE = re.compile(r"(Traceback|OutOfMemoryError|CUDA out of memory|RuntimeError|Error:)")


STAGE_LABELS = {
    "lmk_init_rigid": "landmark rigid init",
    "lmk_init_all": "landmark full init",
    "rgb_init_texture": "texture init",
    "rgb_init_all": "photometric init",
    "rgb_init_offset": "offset init",
    "lmk_sequential_tracking": "landmark sequential tracking",
    "rgb_sequential_tracking": "RGB sequential tracking",
    "lmk_global_tracking": "landmark global refinement",
    "rgb_global_tracking": "RGB global refinement",
}


@dataclass
class Config:
    total_frames: int
    batch_size: int
    begin_timestep: int = 0
    photometric: bool = True
    use_static_offset: bool = True
    stage_steps: dict[str, int] = field(
        default_factory=lambda: {
            "lmk_init_rigid": 500,
            "lmk_init_all": 500,
            "rgb_init_texture": 500,
            "rgb_init_all": 500,
            "rgb_init_offset": 500,
            "lmk_sequential_tracking": 50,
            "rgb_sequential_tracking": 50,
        }
    )
    global_epochs: int = 30


@dataclass
class Point:
    ts: datetime
    step_done: int


@dataclass
class State:
    config: Config
    phase: str
    stage: str
    latest_step_done: int
    latest_timestep: int
    total_steps: int
    init_steps: int
    seq_steps: int
    global_steps: int
    global_epoch: int | None
    global_total_epochs: int | None
    last_checkpoint_frames: int | None
    elapsed_seconds: float | None
    eta_seconds: float | None
    eta_basis: str
    log_age_seconds: float | None
    done: bool
    error: str | None
    gpu: str


def read_log(path: Path, max_bytes: int) -> list[str]:
    if not path.exists():
        return []
    with open(path, "r", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        return f.readlines()


def strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line.rstrip("\n"))


def parse_ts(line: str) -> datetime | None:
    match = TIME_RE.search(line)
    if not match:
        return None
    try:
        return datetime.strptime(
            f"{datetime.now().year}/{match.group('time')}",
            "%Y/%m/%d %H:%M:%S",
        )
    except ValueError:
        return None


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "on"}


def parse_config(lines: list[str], fallback_frames: int, fallback_batch_size: int) -> Config:
    config = Config(total_frames=fallback_frames, batch_size=fallback_batch_size)
    current_section: str | None = None

    for raw in lines:
        line = strip_ansi(raw)
        stripped = line.strip()

        start = START_RE.search(line)
        if start:
            config.total_frames = int(start.group("total"))

        if stripped.startswith(("lmk_", "rgb_")) and ": " in stripped:
            current_section = stripped.split(":", 1)[0]
            continue

        if stripped and not line.startswith(" ") and not line.startswith("\t"):
            current_section = None

        if stripped.startswith("batch_size:"):
            config.batch_size = int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("begin_timestep:"):
            config.begin_timestep = int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("photometric:"):
            config.photometric = parse_bool(stripped.split(":", 1)[1])
        elif stripped.startswith("use_static_offset:"):
            config.use_static_offset = parse_bool(stripped.split(":", 1)[1])
        elif current_section and stripped.startswith("num_steps:"):
            if current_section in config.stage_steps:
                config.stage_steps[current_section] = int(stripped.split(":", 1)[1].strip())
        elif current_section in {"rgb_global_tracking", "lmk_global_tracking"} and stripped.startswith("num_epochs:"):
            config.global_epochs = int(stripped.split(":", 1)[1].strip())

    config.total_frames = max(1, config.total_frames)
    config.batch_size = max(1, config.batch_size)
    config.begin_timestep = max(0, min(config.begin_timestep, config.total_frames))
    return config


def init_stage_names(config: Config) -> list[str]:
    names = ["lmk_init_rigid", "lmk_init_all"]
    if config.photometric:
        names.extend(["rgb_init_texture", "rgb_init_all"])
        if config.use_static_offset:
            names.append("rgb_init_offset")
    return names


def sequential_stage(config: Config) -> str:
    return "rgb_sequential_tracking" if config.photometric else "lmk_sequential_tracking"


def global_stage(config: Config) -> str:
    return "rgb_global_tracking" if config.photometric else "lmk_global_tracking"


def phase_budgets(config: Config) -> tuple[int, int, int]:
    init_steps = 0
    if config.begin_timestep == 0:
        init_steps = sum(config.stage_steps[name] for name in init_stage_names(config))
    remaining_frames = max(0, config.total_frames - config.begin_timestep)
    batches = max(1, math.ceil(remaining_frames / config.batch_size))
    seq_steps = batches * config.stage_steps[sequential_stage(config)]
    global_steps = batches * max(1, config.global_epochs)
    return init_steps, seq_steps, global_steps


def rate_from_points(points: list[Point], window: int) -> float | None:
    if len(points) < 2:
        return None
    deduped: list[Point] = []
    for point in points:
        if not deduped or point.step_done != deduped[-1].step_done:
            deduped.append(point)
    if len(deduped) < 2:
        return None
    window_points = deduped[-max(2, window) :]
    first = window_points[0]
    last = window_points[-1]
    elapsed = (last.ts - first.ts).total_seconds()
    steps = last.step_done - first.step_done
    if elapsed <= 0 or steps <= 0:
        return None
    return steps / elapsed


def gpu_status() -> str:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
        util, used, total, temp = [part.strip() for part in output.split(",")[:4]]
        return f"GPU {util}% | VRAM {used}/{total} MiB | {temp}C"
    except Exception:
        return "GPU unavailable"


def parse_state(
    log_path: Path,
    fallback_frames: int,
    fallback_batch_size: int,
    rate_window: int,
    max_bytes: int,
) -> State | None:
    raw_lines = read_log(log_path, max_bytes=max_bytes)
    if not raw_lines:
        return None

    lines = [strip_ansi(line) for line in raw_lines]
    config = parse_config(lines, fallback_frames, fallback_batch_size)
    init_steps, seq_steps, global_steps = phase_budgets(config)
    total_steps = init_steps + seq_steps + global_steps

    start_idx = 0
    for i, line in enumerate(lines):
        if START_RE.search(line):
            start_idx = i

    session_lines = lines[start_idx:]
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    points: list[Point] = []
    latest_stage = "starting"
    latest_step_done = 0
    latest_timestep = config.begin_timestep
    global_epoch: int | None = None
    global_total_epochs: int | None = None
    last_checkpoint_frames: int | None = None
    done = False
    error: str | None = None
    saw_global = False

    for line in session_lines:
        ts = parse_ts(line)
        if ts:
            first_ts = first_ts or ts
            last_ts = ts

        track = TRACK_RE.search(line)
        if track:
            latest_stage = track.group("stage")
            latest_step_done = int(track.group("step")) + 1
            latest_timestep = int(track.group("timestep"))
            if latest_stage.endswith("_global_tracking"):
                saw_global = True
            if ts:
                points.append(Point(ts=ts, step_done=latest_step_done))

        epoch = EPOCH_RE.search(line)
        if epoch:
            global_epoch = int(epoch.group("epoch"))
            global_total_epochs = int(epoch.group("total"))
            saw_global = True

        checkpoint = CHECKPOINT_RE.search(line)
        if checkpoint:
            last_checkpoint_frames = int(checkpoint.group("frames"))

        if "Start global optimization of all frames" in line:
            saw_global = True

        if "All done." in line:
            done = True

        if ERROR_RE.search(line):
            error = line.strip()

    if done:
        phase = "complete"
        stage = "complete"
        step_done = total_steps
    elif saw_global:
        phase = "global refinement"
        stage = latest_stage if latest_stage.endswith("_global_tracking") else global_stage(config)
        step_done = max(init_steps + seq_steps, latest_step_done)
    elif latest_step_done >= init_steps:
        phase = "sequential tracking"
        stage = latest_stage if latest_stage.endswith("_sequential_tracking") else sequential_stage(config)
        step_done = min(total_steps, latest_step_done)
    else:
        phase = "initialization"
        stage = latest_stage
        step_done = latest_step_done

    rate = rate_from_points(points, rate_window)
    eta_seconds: float | None = None
    eta_basis = "waiting for enough progress samples"
    if done:
        eta_seconds = 0
        eta_basis = "complete"
    elif rate:
        remaining_steps = max(0, total_steps - step_done)
        eta_seconds = remaining_steps / rate
        eta_basis = f"{rate:.2f} optimizer steps/sec over recent log window"

    elapsed_seconds = None
    if first_ts and last_ts:
        elapsed_seconds = (last_ts - first_ts).total_seconds()

    log_age_seconds = time.time() - log_path.stat().st_mtime if log_path.exists() else None

    return State(
        config=config,
        phase=phase,
        stage=stage,
        latest_step_done=step_done,
        latest_timestep=latest_timestep,
        total_steps=total_steps,
        init_steps=init_steps,
        seq_steps=seq_steps,
        global_steps=global_steps,
        global_epoch=global_epoch,
        global_total_epochs=global_total_epochs,
        last_checkpoint_frames=last_checkpoint_frames,
        elapsed_seconds=elapsed_seconds,
        eta_seconds=eta_seconds,
        eta_basis=eta_basis,
        log_age_seconds=log_age_seconds,
        done=done,
        error=error,
        gpu=gpu_status(),
    )


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    return str(timedelta(seconds=max(0, int(seconds))))


def progress_bar(frac: float, width: int) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(width * frac)
    return "#" * filled + "-" * (width - filled)


def find_default_log(subject: str | None, mode: str | None) -> Path:
    if subject and mode:
        return Path("output") / subject / "logs" / f"vhap_track_{mode}.log"
    logs = sorted(Path("output").glob("*/logs/vhap_track_*.log"), key=lambda p: p.stat().st_mtime)
    if logs:
        return logs[-1]
    raise FileNotFoundError("No VHAP tracking logs found under output/*/logs")


def render(log_path: Path, state: State) -> None:
    width = shutil.get_terminal_size((120, 25)).columns
    bar_width = max(20, min(72, width - 54))
    overall_frac = state.latest_step_done / max(1, state.total_steps)

    if state.phase == "initialization":
        phase_done = min(state.latest_step_done, state.init_steps)
        phase_total = state.init_steps
    elif state.phase == "sequential tracking":
        phase_done = min(max(0, state.latest_step_done - state.init_steps), state.seq_steps)
        phase_total = state.seq_steps
    elif state.phase == "global refinement":
        phase_done = min(max(0, state.latest_step_done - state.init_steps - state.seq_steps), state.global_steps)
        phase_total = state.global_steps
    else:
        phase_done = phase_total = 1

    phase_frac = phase_done / max(1, phase_total)
    frame_estimate = min(
        state.config.total_frames,
        max(state.latest_timestep, state.config.begin_timestep),
    )

    print("\033[H\033[J", end="")
    print("VHAP TRACKING ETA MONITOR")
    print(f"Log:      {log_path}")
    print(f"Status:   {state.phase} | {STAGE_LABELS.get(state.stage, state.stage)}")
    print(f"GPU:      {state.gpu}")
    print()
    print(
        f"Overall:  [{progress_bar(overall_frac, bar_width)}] "
        f"{overall_frac * 100:6.2f}%  {state.latest_step_done}/{state.total_steps} optimizer steps"
    )
    print(
        f"Phase:    [{progress_bar(phase_frac, bar_width)}] "
        f"{phase_frac * 100:6.2f}%  {phase_done}/{phase_total} phase steps"
    )
    print(
        f"Frames:   current~{frame_estimate}/{state.config.total_frames}"
        f" | batch={state.config.batch_size}"
        f" | checkpoints={state.last_checkpoint_frames if state.last_checkpoint_frames is not None else 'none yet'}"
    )
    if state.global_epoch is not None:
        print(f"Epoch:    {state.global_epoch}/{state.global_total_epochs}")
    print(
        f"Time:     elapsed={fmt_duration(state.elapsed_seconds)}"
        f" | ETA={fmt_duration(state.eta_seconds)}"
        f" | log_age={fmt_duration(state.log_age_seconds)}"
    )
    print(f"ETA:      {state.eta_basis}")
    if state.error:
        print(f"Warning:  last error marker: {state.error}")
    print()
    print("Read-only monitor. Ctrl+C exits this monitor only.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live read-only VHAP tracking progress + ETA monitor.")
    parser.add_argument("--log", type=Path, default=None, help="Explicit VHAP tracking log path")
    parser.add_argument("--subject", default=None, help="Subject name, e.g. new_speaker")
    parser.add_argument("--mode", choices=["pilot", "full"], default=None, help="Pipeline mode")
    parser.add_argument("--frames", type=int, default=8556, help="Fallback frame count")
    parser.add_argument("--batch-size", type=int, default=8, help="Fallback VHAP batch size")
    parser.add_argument("--rate-window", type=int, default=8, help="Recent log points used for ETA")
    parser.add_argument("--max-bytes", type=int, default=32_000_000, help="Max log bytes to scan")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval in seconds")
    parser.add_argument("--watch", "-w", action="store_true", help="Refresh until interrupted")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    args = parser.parse_args()

    if args.frames <= 0:
        raise SystemExit("--frames must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    log_path = args.log if args.log else find_default_log(args.subject, args.mode)

    while True:
        state = parse_state(
            log_path=log_path,
            fallback_frames=args.frames,
            fallback_batch_size=args.batch_size,
            rate_window=args.rate_window,
            max_bytes=args.max_bytes,
        )
        if state is None:
            print("\033[H\033[J", end="")
            print(f"Waiting for VHAP log data: {log_path}")
            print("Read-only monitor. Ctrl+C exits this monitor only.")
        else:
            render(log_path, state)

        if args.once or not args.watch:
            return
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
