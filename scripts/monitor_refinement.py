#!/usr/bin/env python3
import argparse
import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TIME_RE = re.compile(r"\[(?P<time>\d{2}/\d{2} \d{2}:\d{2}:\d{2})")
EPOCH_RE = re.compile(r"EPOCH\s+(?P<epoch>\d+)\s*/\s*(?P<total>\d+)")
GLOBAL_STEP_RE = re.compile(
    r"\[train-(?:rgb|lmk)_global_tracking\]\s+timestep\s+\d+\s+step\s+(?P<step>\d+):"
)
TOTAL_FRAMES_RE = re.compile(r"Start sequential tracking FLAME in (?P<total>\d+) frames")
BATCH_SIZE_RE = re.compile(r"^\s*batch_size:\s*(?P<batch>\d+)\s*$")
RESUME_RE = re.compile(r"Resuming global refinement from: .*tracked_flame_params_(?P<epoch>\d+)\.npz")
CHECKPOINT_INFO_RE = re.compile(
    r"checkpoint_epoch=(?P<completed>\d+),\s+remaining_global_epochs=(?P<remaining>\d+)"
)
DONE_RE = re.compile(r"All done\.")


def read_log_tail(path: Path, bytes_back: int = 2_000_000) -> list[str]:
    if not path.exists():
        return []
    with open(path, "r", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - bytes_back))
        return f.readlines()


def parse_ts(raw: str) -> datetime | None:
    m = TIME_RE.search(raw)
    if not m:
        return None
    try:
        return datetime.strptime(
            f"{datetime.now().year}/{m.group('time')}",
            "%Y/%m/%d %H:%M:%S",
        )
    except ValueError:
        return None


def fmt_td(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    seconds = max(0, int(seconds))
    return str(timedelta(seconds=seconds))


def gpu_status() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
        util, used, total, temp = [p.strip() for p in out.split(",")[:4]]
        return f"{Colors.OKGREEN}GPU {util}% | VRAM {used}/{total} MiB | {temp}C{Colors.ENDC}"
    except (subprocess.CalledProcessError, FileNotFoundError, TimeoutError, ValueError):
        return f"{Colors.FAIL}GPU status unavailable{Colors.ENDC}"


def progress_bar(frac: float, width: int) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(width * frac)
    return "█" * filled + "░" * (width - filled)


def parse_state(
    log_path: Path,
    default_frames: int,
    default_batch_size: int,
    default_total_epochs: int,
    rate_window_points: int,
):
    lines = read_log_tail(log_path)
    if not lines:
        return None

    start_idx = 0
    resume_completed_epochs = 0
    session_total_epochs = default_total_epochs

    for i, raw in enumerate(lines):
        clean = ANSI_RE.sub("", raw.rstrip("\n"))
        resume_m = RESUME_RE.search(clean)
        info_m = CHECKPOINT_INFO_RE.search(clean)
        epoch_m = EPOCH_RE.search(clean)
        if resume_m:
            start_idx = i
            resume_completed_epochs = int(resume_m.group("epoch"))
        elif info_m:
            start_idx = i if i > start_idx else start_idx
            resume_completed_epochs = int(info_m.group("completed"))
            session_total_epochs = int(info_m.group("remaining"))
        elif epoch_m and start_idx == 0:
            # Fresh runs may not emit resume metadata; keep the first epoch block.
            start_idx = i

    lines = lines[start_idx:]

    total_frames = default_frames
    batch_size = default_batch_size
    current_epoch = 1
    total_epochs = session_total_epochs
    latest_step = -1
    step_points: list[tuple[datetime, int]] = []
    first_ts = None
    last_ts = None
    done = False

    for raw in lines:
        clean = ANSI_RE.sub("", raw.rstrip("\n"))
        ts = parse_ts(clean)
        if ts:
            if first_ts is None:
                first_ts = ts
            last_ts = ts

        total_m = TOTAL_FRAMES_RE.search(clean)
        if total_m:
            total_frames = int(total_m.group("total"))

        batch_m = BATCH_SIZE_RE.search(clean)
        if batch_m:
            batch_size = int(batch_m.group("batch"))

        epoch_m = EPOCH_RE.search(clean)
        if epoch_m:
            current_epoch = int(epoch_m.group("epoch"))
            total_epochs = int(epoch_m.group("total"))

        step_m = GLOBAL_STEP_RE.search(clean)
        if step_m:
            step = int(step_m.group("step"))
            latest_step = max(latest_step, step)
            if ts:
                step_points.append((ts, step))

        if DONE_RE.search(clean):
            done = True

    steps_per_epoch = max(1, math.ceil(total_frames / max(1, batch_size)))
    total_steps = max(1, steps_per_epoch * max(1, total_epochs))
    overall_total_epochs = max(1, resume_completed_epochs + total_epochs)
    overall_total_steps = steps_per_epoch * overall_total_epochs

    if latest_step >= 0:
        # The tracker logs a global step counter for the resumed run, so derive
        # progress by wrapping that counter within the current epoch.
        step_in_epoch = (latest_step % steps_per_epoch) + 1
        inferred_epoch = (latest_step // steps_per_epoch) + 1
        if current_epoch <= 1:
            current_epoch = min(total_epochs, inferred_epoch)
        session_done_steps = latest_step + 1
    else:
        session_done_steps = 0
        step_in_epoch = 0

    overall_done_steps = min(overall_total_steps, resume_completed_epochs * steps_per_epoch + session_done_steps)
    session_frac = session_done_steps / total_steps
    overall_frac = overall_done_steps / overall_total_steps
    elapsed = None
    if first_ts and last_ts:
        elapsed = (last_ts - first_ts).total_seconds()

    # ETA from recent slope of (timestamp, global_step)
    eta_seconds = None
    if len(step_points) >= 2 and latest_step >= 0:
        dedup_points: list[tuple[datetime, int]] = []
        for p in step_points:
            if not dedup_points or p[1] != dedup_points[-1][1]:
                dedup_points.append(p)
        window = dedup_points[-max(2, rate_window_points) :]
        t0, s0 = window[0]
        t1, s1 = window[-1]
        dt = (t1 - t0).total_seconds()
        ds = s1 - s0
        if dt > 0 and ds > 0:
            rate = ds / dt
            remaining_steps = max(0, overall_total_steps - overall_done_steps)
            eta_seconds = remaining_steps / rate

    log_age = None
    if log_path.exists():
        log_age = time.time() - log_path.stat().st_mtime

    return {
        "done": done,
        "resume_completed_epochs": resume_completed_epochs,
        "total_frames": total_frames,
        "batch_size": batch_size,
        "current_epoch": current_epoch,
        "total_epochs": total_epochs,
        "step_in_epoch": step_in_epoch,
        "steps_per_epoch": steps_per_epoch,
        "session_done_steps": session_done_steps,
        "session_total_steps": total_steps,
        "overall_done_steps": overall_done_steps,
        "overall_total_steps": overall_total_steps,
        "session_frac": session_frac,
        "overall_frac": overall_frac,
        "eta_seconds": eta_seconds,
        "elapsed_seconds": elapsed,
        "log_age_seconds": log_age,
    }


def find_default_log() -> Path:
    direct = Path("output/sam_altman/logs/vhap_refine_resume.log")
    if direct.exists():
        return direct
    logs = sorted(Path("output/sam_altman/logs").glob("vhap_refine*.log"))
    if logs:
        return logs[-1]
    raise FileNotFoundError("No refinement log found under output/sam_altman/logs")


def render(log_path: Path, state: dict):
    width = shutil.get_terminal_size((120, 25)).columns
    bar_width = max(20, min(70, width - 52))
    bar = progress_bar(state["overall_frac"], bar_width)

    print("\033[H\033[J", end="")
    print(f"{Colors.BOLD}{Colors.HEADER}VHAP REFINEMENT LIVE MONITOR{Colors.ENDC} | {gpu_status()}\n")
    print(f"Log: {log_path}")
    print(
        f"Overall:  [{Colors.OKCYAN}{bar}{Colors.ENDC}] {state['overall_frac'] * 100:6.2f}%"
        f"  ({state['overall_done_steps']}/{state['overall_total_steps']} steps)"
    )
    print(
        f"Session:  {Colors.BOLD}{state['current_epoch']}/{state['total_epochs']}{Colors.ENDC}"
        f" | Batch: {state['step_in_epoch']}/{state['steps_per_epoch']}"
        f" | Resumed after epoch: {state['resume_completed_epochs']}"
    )
    print(
        f"Session %: {state['session_frac'] * 100:6.2f}%"
        f"  ({state['session_done_steps']}/{state['session_total_steps']} steps)"
    )
    print(
        f"Config:   frames={state['total_frames']} | batch_size={state['batch_size']}"
        f" | elapsed={fmt_td(state['elapsed_seconds'])}"
        f" | {Colors.WARNING}ETA={fmt_td(state['eta_seconds'])}{Colors.ENDC}"
    )
    print(
        f"Log age:  {fmt_td(state['log_age_seconds'])}"
        + (f"  {Colors.OKGREEN}[DONE]{Colors.ENDC}" if state["done"] else "")
    )
    print(f"\n{Colors.OKBLUE}Read-only monitor. Ctrl+C exits monitor only.{Colors.ENDC}")


def main():
    parser = argparse.ArgumentParser(description="Live VHAP refinement progress + ETA monitor (read-only).")
    parser.add_argument("--log", type=Path, default=None, help="Refinement log path")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds")
    parser.add_argument("--epochs", type=int, default=30, help="Fallback total epochs")
    parser.add_argument("--frames", type=int, default=8556, help="Fallback total frames")
    parser.add_argument("--batch-size", type=int, default=4, help="Fallback batch size")
    parser.add_argument("--rate-window", type=int, default=10, help="Number of recent step points for ETA slope")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    args = parser.parse_args()

    log_path = args.log if args.log else find_default_log()
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.frames <= 0:
        raise SystemExit("--frames must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    try:
        while True:
            state = parse_state(
                log_path=log_path,
                default_frames=args.frames,
                default_batch_size=args.batch_size,
                default_total_epochs=args.epochs,
                rate_window_points=args.rate_window,
            )
            if state is None:
                print("\033[H\033[J", end="")
                print(f"{Colors.FAIL}Waiting for log data in {log_path}{Colors.ENDC}")
                print(f"\n{Colors.OKBLUE}Read-only monitor. Ctrl+C exits monitor only.{Colors.ENDC}")
            else:
                render(log_path, state)
            if args.once:
                return
            time.sleep(max(0.5, args.interval))
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
