#!/usr/bin/env python3
"""Read-only MotionTranslator audio-driver training progress and ETA monitor."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
DEVICE_RE = re.compile(r"\[train_audio_driver\]\s+Device:\s+(?P<device>\S+)")
HUBERT_FRAMES_RE = re.compile(r"(?:Cached )?HuBERT frames:\s+(?P<frames>\d+)")
TRAIN_RE = re.compile(
    r"Train frames:\s+(?P<frames>\d+),\s+windows:\s+(?P<windows>\d+),\s+batches/epoch:\s+(?P<batches>\d+)"
)
VAL_RE = re.compile(r"Val frames:\s+(?P<frames>\d+),\s+windows:\s+(?P<windows>\d+)")
PARAMS_RE = re.compile(r"MotionTranslator:\s+(?P<params>[\d.]+)\s+M parameters")
HEADER_RE = re.compile(r"Training MotionTranslator for (?P<epochs>\d+) epochs")
EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s*/\s*(?P<total>\d+)\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+"
    r"expr=(?P<expr>[-+0-9.eE]+)\s+"
    r"jaw=(?P<jaw>[-+0-9.eE]+)\s+"
    r"vel=(?P<vel>[-+0-9.eE]+)\s+"
    r"lr=(?P<lr>[-+0-9.eE]+)\s+"
    r"(?:val=(?P<val>[-+0-9.eE]+)\s+)?"
    r"t=(?P<seconds>[-+0-9.eE]+)s"
)
BEST_RE = re.compile(r"New best:\s+(?P<loss>[-+0-9.eE]+)")
COMPLETE_RE = re.compile(r"Training complete\.\s+Best loss:\s+(?P<loss>[-+0-9.eE]+)")
ERROR_RE = re.compile(r"(Traceback|OutOfMemoryError|CUDA out of memory|RuntimeError|Error:|Exception)")


@dataclass
class EpochPoint:
    epoch: int
    total: int
    loss: float
    expr: float
    jaw: float
    vel: float
    lr: str
    val: float | None
    seconds: float


@dataclass
class State:
    subject: str
    mode: str
    log_path: Path
    checkpoint_dir: Path
    stage: str
    status: str
    device: str | None
    hubert_frames: int | None
    train_frames: int | None
    train_windows: int | None
    batches_per_epoch: int | None
    val_frames: int | None
    val_windows: int | None
    params_m: float | None
    epoch: EpochPoint | None
    total_epochs: int | None
    best_loss: float | None
    elapsed_seconds: float | None
    eta_seconds: float | None
    avg_epoch_seconds: float | None
    log_age_seconds: float | None
    last_checkpoint: str | None
    best_checkpoint: str | None
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


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    return fmt_duration(seconds) + " ago"


def checkpoint_summary(path: Path) -> str | None:
    if not path.exists():
        return None
    size_mb = path.stat().st_size / (1024 * 1024)
    mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S")
    return f"{path} ({size_mb:.1f} MiB, {mtime})"


def process_status(session_name: str) -> str:
    try:
        subprocess.check_output(
            ["tmux", "has-session", "-t", session_name],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return f"running (tmux: {session_name})"
    except Exception:
        pass

    try:
        output = subprocess.check_output(
            ["pgrep", "-af", "scripts/train_audio_driver.py"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
        if output:
            first = output.splitlines()[0]
            return f"running (process: {first[:110]})"
    except Exception:
        pass

    return "not running"


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


def latest_session(lines: list[str]) -> list[str]:
    start = 0
    for i, line in enumerate(lines):
        if DEVICE_RE.search(line):
            start = i
    return lines[start:]


def parse_state(
    subject: str,
    mode: str,
    log_path: Path,
    checkpoint_dir: Path,
    session_name: str,
    max_bytes: int,
    rate_window: int,
) -> State | None:
    raw_lines = read_log(log_path, max_bytes)
    if not raw_lines:
        return None

    lines = latest_session([strip_ansi(line) for line in raw_lines])
    device: str | None = None
    hubert_frames: int | None = None
    train_frames: int | None = None
    train_windows: int | None = None
    batches_per_epoch: int | None = None
    val_frames: int | None = None
    val_windows: int | None = None
    params_m: float | None = None
    total_epochs: int | None = None
    best_loss: float | None = None
    epoch_points: list[EpochPoint] = []
    complete = False
    error: str | None = None

    for line in lines:
        if match := DEVICE_RE.search(line):
            device = match.group("device")
        if match := HUBERT_FRAMES_RE.search(line):
            hubert_frames = int(match.group("frames"))
        if match := TRAIN_RE.search(line):
            train_frames = int(match.group("frames"))
            train_windows = int(match.group("windows"))
            batches_per_epoch = int(match.group("batches"))
        if match := VAL_RE.search(line):
            val_frames = int(match.group("frames"))
            val_windows = int(match.group("windows"))
        if match := PARAMS_RE.search(line):
            params_m = float(match.group("params"))
        if match := HEADER_RE.search(line):
            total_epochs = int(match.group("epochs"))
        if match := EPOCH_RE.search(line):
            point = EpochPoint(
                epoch=int(match.group("epoch")),
                total=int(match.group("total")),
                loss=float(match.group("loss")),
                expr=float(match.group("expr")),
                jaw=float(match.group("jaw")),
                vel=float(match.group("vel")),
                lr=match.group("lr"),
                val=float(match.group("val")) if match.group("val") else None,
                seconds=float(match.group("seconds")),
            )
            epoch_points.append(point)
            total_epochs = point.total
        if match := BEST_RE.search(line):
            best_loss = float(match.group("loss"))
        if match := COMPLETE_RE.search(line):
            best_loss = float(match.group("loss"))
            complete = True
        if ERROR_RE.search(line):
            error = line

    latest_epoch = epoch_points[-1] if epoch_points else None
    if total_epochs is None and latest_epoch:
        total_epochs = latest_epoch.total

    avg_epoch_seconds: float | None = None
    eta_seconds: float | None = None
    elapsed_seconds: float | None = None
    if epoch_points:
        window = epoch_points[-max(1, rate_window) :]
        avg_epoch_seconds = sum(point.seconds for point in window) / len(window)
        elapsed_seconds = sum(point.seconds for point in epoch_points)
        if total_epochs and latest_epoch and not complete:
            eta_seconds = max(0, total_epochs - latest_epoch.epoch) * avg_epoch_seconds

    if complete:
        stage = "complete"
        eta_seconds = 0
    elif latest_epoch:
        stage = "training"
    elif hubert_frames:
        stage = "initializing training"
    elif any("Encoding with HuBERT" in line for line in lines):
        stage = "encoding HuBERT features"
    elif any("Loading audio" in line for line in lines):
        stage = "loading data"
    else:
        stage = "starting"

    log_age_seconds = None
    if log_path.exists():
        log_age_seconds = max(0.0, time.time() - log_path.stat().st_mtime)

    return State(
        subject=subject,
        mode=mode,
        log_path=log_path,
        checkpoint_dir=checkpoint_dir,
        stage=stage,
        status=process_status(session_name),
        device=device,
        hubert_frames=hubert_frames,
        train_frames=train_frames,
        train_windows=train_windows,
        batches_per_epoch=batches_per_epoch,
        val_frames=val_frames,
        val_windows=val_windows,
        params_m=params_m,
        epoch=latest_epoch,
        total_epochs=total_epochs,
        best_loss=best_loss,
        elapsed_seconds=elapsed_seconds,
        eta_seconds=eta_seconds,
        avg_epoch_seconds=avg_epoch_seconds,
        log_age_seconds=log_age_seconds,
        last_checkpoint=checkpoint_summary(checkpoint_dir / "last_model.pt"),
        best_checkpoint=checkpoint_summary(checkpoint_dir / "best_model.pt"),
        error=error,
        gpu=gpu_status(),
    )


def render(state: State) -> str:
    now = datetime.now()
    lines = [
        f"Audio driver progress - {state.subject} ({state.mode})",
        f"  status: {state.status}",
        f"  stage:  {state.stage}",
        f"  log:    {state.log_path} ({fmt_age(state.log_age_seconds)})",
    ]
    if state.device:
        lines.append(f"  device: {state.device}")
    lines.append(f"  gpu:    {state.gpu}")

    if state.hubert_frames is not None:
        lines.append(f"  hubert: {state.hubert_frames:,} feature frames")
    if state.train_windows is not None and state.batches_per_epoch is not None:
        lines.append(
            "  train:  "
            f"{state.train_frames:,} frames | {state.train_windows:,} windows | "
            f"{state.batches_per_epoch:,} batches/epoch"
        )
    if state.val_windows is not None:
        lines.append(f"  val:    {state.val_frames:,} frames | {state.val_windows:,} windows")
    if state.params_m is not None:
        lines.append(f"  model:  {state.params_m:.2f}M params")

    if state.epoch:
        total = state.total_epochs or state.epoch.total
        pct = 100.0 * state.epoch.epoch / max(1, total)
        val = f"{state.epoch.val:.5f}" if state.epoch.val is not None else "n/a"
        lines.extend(
            [
                f"  epoch:  {state.epoch.epoch}/{total} ({pct:.1f}%)",
                (
                    "  losses: "
                    f"train={state.epoch.loss:.5f} expr={state.epoch.expr:.5f} "
                    f"jaw={state.epoch.jaw:.5f} vel={state.epoch.vel:.5f} val={val}"
                ),
                f"  lr:     {state.epoch.lr}",
            ]
        )
    elif state.total_epochs:
        lines.append(f"  epoch:  0/{state.total_epochs} (0.0%)")

    if state.best_loss is not None:
        lines.append(f"  best:   {state.best_loss:.5f}")
    if state.avg_epoch_seconds is not None:
        lines.append(f"  speed:  {fmt_duration(state.avg_epoch_seconds)} / epoch")
    lines.append(f"  elapsed:{fmt_duration(state.elapsed_seconds)}")
    if state.eta_seconds is not None:
        finish = now + timedelta(seconds=state.eta_seconds)
        lines.append(f"  eta:    {fmt_duration(state.eta_seconds)} (finish ~ {finish:%H:%M:%S})")
    else:
        lines.append("  eta:    unknown")

    if state.last_checkpoint:
        lines.append(f"  last:   {state.last_checkpoint}")
    if state.best_checkpoint:
        lines.append(f"  bestck: {state.best_checkpoint}")
    if state.error:
        lines.append(f"  error:  {state.error}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only audio-driver progress and ETA monitor.")
    parser.add_argument("--subject", default="new_speaker")
    parser.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    parser.add_argument("--log", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--session", default=None)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--rate-window", type=int, default=8)
    parser.add_argument("--max-bytes", type=int, default=1_000_000)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    log_path = args.log or root / "output" / args.subject / "logs" / f"audio_driver_{args.mode}.log"
    checkpoint_dir = args.checkpoint_dir or root / "audio_driver" / "checkpoints" / args.subject
    session_name = args.session or f"{args.subject}_audio_{args.mode}"

    while True:
        state = parse_state(
            subject=args.subject,
            mode=args.mode,
            log_path=log_path,
            checkpoint_dir=checkpoint_dir,
            session_name=session_name,
            max_bytes=args.max_bytes,
            rate_window=args.rate_window,
        )
        if args.watch:
            print("\033[2J\033[H", end="")
        if state is None:
            print(f"No audio progress log found yet: {log_path}")
        else:
            print(render(state))
        if not args.watch:
            return 0
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
