#!/usr/bin/env python3
import argparse
import datetime as dt
import re
import shutil
import subprocess
import time
from pathlib import Path


LOG_RE = re.compile(
    r"\[(?P<context>[^\]]+)\]\s+timestep\s+(?P<timestep>\d+)\s+step\s+(?P<step>\d+):"
)
EPOCH_RE = re.compile(r"EPOCH\s+(?P<epoch>\d+)\s*/\s*(?P<total>\d+)")
START_RE = re.compile(r"Start sequential tracking FLAME in (?P<total>\d+) frames")
DONE_RE = re.compile(r"All done\.")
ERROR_RE = re.compile(r"(Traceback|OutOfMemoryError|RuntimeError|Error:)")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_log(path: Path):
    total_frames = 8556
    entries = []
    current_epoch = None
    all_done = False
    last_error = None

    if not path.exists():
        return total_frames, entries, current_epoch, all_done, f"log not found: {path}"

    for line in path.read_text(errors="replace").splitlines():
        line = ANSI_RE.sub("", line)
        start = START_RE.search(line)
        if start:
            total_frames = int(start.group("total"))

        entry = LOG_RE.search(line)
        if entry:
            entries.append(
                {
                    "context": entry.group("context"),
                    "timestep": int(entry.group("timestep")),
                    "step": int(entry.group("step")),
                    "raw": line,
                }
            )

        epoch = EPOCH_RE.search(line)
        if epoch:
            current_epoch = (int(epoch.group("epoch")), int(epoch.group("total")))

        if DONE_RE.search(line):
            all_done = True

        if ERROR_RE.search(line):
            last_error = line.strip()

    return total_frames, entries, current_epoch, all_done, last_error


def gpu_summary():
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
        util, used, total, temp = [x.strip() for x in out.split(",")[:4]]
        return f"GPU {util}% | VRAM {used}/{total} MiB | {temp}C"
    except Exception:
        return "GPU unavailable"


def fmt_duration(seconds):
    if seconds is None:
        return "--:--:--"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def progress_bar(frac, width):
    frac = max(0.0, min(1.0, frac))
    filled = int(round(width * frac))
    return "#" * filled + "-" * (width - filled)


def compute_status(log_path: Path):
    total, entries, epoch, done, error = parse_log(log_path)
    now = time.time()
    stat = log_path.stat() if log_path.exists() else None
    mtime = stat.st_mtime if stat else None

    if done:
        return "done", total, total, 1.0, None, "complete", error, mtime

    if epoch:
        current, total_epochs = epoch
        frac = (current - 1) / max(total_epochs, 1)
        return (
            f"global epoch {current}/{total_epochs}",
            current - 1,
            total_epochs,
            frac,
            None,
            "global optimization ETA is not reliable from VHAP logs",
            error,
            mtime,
        )

    seq_entries = [e for e in entries if "rgb_sequential_tracking" in e["context"]]
    if seq_entries:
        latest = seq_entries[-1]
        current = min(latest["timestep"], total)
        frac = current / max(total, 1)
        eta = None
        if len(seq_entries) >= 2 and mtime:
            prev = seq_entries[-2]
            delta_frames = max(1, latest["timestep"] - prev["timestep"])
            # VHAP log writes each latest line near current mtime; estimate by log-line cadence.
            # The retry logs around 8 frames per ~20 seconds for this dataset/batch size.
            seconds_per_frame = 20.5 / delta_frames
            eta = (total - current) * seconds_per_frame
        return (
            latest["context"],
            current,
            total,
            frac,
            eta,
            f"step {latest['step']}",
            error,
            mtime,
        )

    if entries:
        latest = entries[-1]
        # Initial stage budget: 2500 logged optimizer steps before sequential tracking starts.
        current = min(latest["step"], 2500)
        frac = current / 2500
        return (
            latest["context"],
            current,
            2500,
            frac,
            None,
            f"initialization step {latest['step']}",
            error,
            mtime,
        )

    return "starting", 0, total, 0.0, None, "waiting for first progress line", error, mtime


def render(log_path: Path):
    stage, current, total, frac, eta, detail, error, mtime = compute_status(log_path)
    term_width = shutil.get_terminal_size((100, 20)).columns
    bar_width = max(20, min(50, term_width - 55))
    bar = progress_bar(frac, bar_width)
    percent = frac * 100
    age = time.time() - mtime if mtime else None

    print(f"VHAP progress: {stage}")
    print(f"[{bar}] {percent:6.2f}%  {current}/{total}")
    print(f"{detail} | ETA {fmt_duration(eta)} | log age {fmt_duration(age)}")
    print(gpu_summary())
    if error:
        print(f"last error marker: {error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log",
        nargs="?",
        default=None,
        help="VHAP tracking log path",
    )
    parser.add_argument("--watch", "-w", action="store_true", help="refresh every 5 seconds")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()

    if args.log is None:
        logs = sorted(Path("output/sam_altman/logs").glob("vhap_track*.log"))
        if not logs:
            raise SystemExit("No VHAP tracking logs found under output/sam_altman/logs")
        log_path = logs[-1]
    else:
        log_path = Path(args.log)
    if args.watch:
        while True:
            print("\033[2J\033[H", end="")
            render(log_path)
            time.sleep(args.interval)
    else:
        render(log_path)


if __name__ == "__main__":
    main()
