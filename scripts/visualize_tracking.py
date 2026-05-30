#!/usr/bin/env python3
import os
import re
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

LOG_RE = re.compile(r"\[(?P<context>[^\]]+)\]\s+timestep\s+(?P<timestep>\d+)\s+step\s+(?P<step>\d+):")
START_RE = re.compile(r"Start sequential tracking FLAME in (?P<total>\d+) frames")
TIME_RE = re.compile(r"\[(?P<time>\d{2}/\d{2} \d{2}:\d{2}:\d{2})")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

def get_gpu():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True).strip()
        u, used, total = out.split(",")
        return f"{Colors.OKGREEN}GPU: {u}% | VRAM: {used}/{total} MiB{Colors.ENDC}"
    except: return "GPU Check Failed"

def parse_log(path):
    if not path.exists(): return None
    try:
        content = path.read_text(errors="replace")
    except: return None
    lines = content.splitlines()
    if not lines: return None

    # Find the start of the latest run to ensure elapsed time is correct
    start_idx = 0
    for i, line in enumerate(lines):
        if "Start sequential tracking FLAME" in line:
            start_idx = i
    lines = lines[start_idx:]

    total, current, step, context = 8556, 0, 0, "Starting"
    first_time, last_time = None, None
    first_seq_time, first_seq_frame = None, 0

    for line in lines:
        clean_line = ANSI_RE.sub("", line)
        
        # Extract time
        t_match = TIME_RE.search(clean_line)
        t_obj = None
        if t_match:
            try:
                t_obj = datetime.strptime(f"{datetime.now().year}/{t_match.group('time')}", "%Y/%m/%d %H:%M:%S")
                if first_time is None: first_time = t_obj
                last_time = t_obj
            except: pass
            
        s_match = START_RE.search(clean_line)
        if s_match: 
            total = int(s_match.group('total'))
        
        l_match = LOG_RE.search(clean_line)
        if l_match:
            current = int(l_match.group('timestep'))
            step = int(l_match.group('step'))
            context = l_match.group('context')
            # Mark the start of sequential tracking for better ETA
            if "sequential_tracking" in context and first_seq_time is None and t_obj:
                first_seq_time = t_obj
                first_seq_frame = current

    if not last_time: return None
    
    # Elapsed is time since run started
    start_point = first_time or datetime.fromtimestamp(path.stat().st_ctime)
    elapsed_seconds = (last_time - start_point).total_seconds()
    
    # Progress calculation
    if current == 0:
        # Show initialization progress based on steps (est 2500 total)
        init_progress = min(0.95, step / 2500)
        percent = (init_progress / total) * 100
    else:
        percent = (current / total) * 100
        
    # ETA calculation: Use rate of sequential tracking if possible
    eta = "Calculating..."
    if current > 0:
        if first_seq_time and last_time > first_seq_time and current > first_seq_frame:
            # Rate of sequential frames (ignoring the slow initialization)
            seq_duration = (last_time - first_seq_time).total_seconds()
            seq_frames = current - first_seq_frame
            rate = seq_frames / seq_duration if seq_duration > 0 else 0
        else:
            # Fallback to total rate including init
            rate = current / elapsed_seconds if elapsed_seconds > 0 else 0
            
        if rate > 0:
            remaining_frames = total - current
            eta_sec = remaining_frames / rate
            eta = str(timedelta(seconds=int(eta_sec)))
        else:
            eta = "Calculating..."
    else:
        eta = "Initializing..."

    return {"cur": current, "tot": total, "step": step, "ctx": context, "pct": percent, "eta": eta, "elap": str(timedelta(seconds=int(elapsed_seconds)))}

def main():
    log = Path("output/sam_altman/logs/vhap_track_sam_altman.log")
    try:
        while True:
            data = parse_log(log)
            print("\033[H\033[J", end="")
            print(f"{Colors.BOLD}{Colors.HEADER}VHAP TRACKER MONITOR{Colors.ENDC} | {get_gpu()}\n")
            if data:
                w = max(20, shutil.get_terminal_size().columns - 50)
                bar = "█" * int(w * min(100, data['pct']) / 100) + "░" * (w - int(w * min(100, data['pct']) / 100))
                print(f"Progress: [{Colors.OKCYAN}{bar}{Colors.ENDC}] {data['pct']:6.2f}%")
                print(f"Frame:    {Colors.BOLD}{data['cur']} / {data['tot']}{Colors.ENDC} (Step: {data['step']})")
                print(f"Stage:    {Colors.OKBLUE}{data['ctx']}{Colors.ENDC}")
                print(f"Time:     Elapsed: {data['elap']} | {Colors.WARNING}ETA: {data['eta']}{Colors.ENDC}")
            else:
                print(f"{Colors.FAIL}Waiting for log content...{Colors.ENDC}")
            print(f"\n{Colors.OKBLUE}Press Ctrl+C to exit monitor (tracking continues in background){Colors.ENDC}")
            time.sleep(2)
    except KeyboardInterrupt: pass

if __name__ == "__main__": main()
