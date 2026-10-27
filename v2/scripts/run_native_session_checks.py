"""Run browser cohort with sampled NVML memory; no physical A/V inference."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from talkinghead_v2.local import digest
from talkinghead_v2.server.native_media import private_write

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    if args.directory.exists(): raise ValueError('refusing to overwrite cohort evidence')
    args.directory.mkdir(parents=True, mode=0o700)
    finished = threading.Event()
    samples, failures = [], []
    def sample():
        while not finished.is_set():
            try:
                device = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], text=True, timeout=3).strip().split(',')
                rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True, timeout=3)
                processes = []
                for row in rows.splitlines():
                    pid, memory = (int(value.strip()) for value in row.split(','))
                    try: command = Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0', ' ')
                    except (OSError, UnicodeError): command = ''
                    category = ('native_media' if 'tha_session_media' in command else
                        'qwen_worker' if 'qwen' in command.lower() and ('worker' in command or 'native' in command) else 'other')
                    processes.append({'pid': pid, 'memory_mib': memory, 'category': category})
                samples.append({'monotonic_ns': time.monotonic_ns(), 'device_used_mib': int(device[0]),
                    'device_total_mib': int(device[1]), 'processes': processes})
            except (ValueError, OSError, subprocess.SubprocessError): failures.append(time.monotonic_ns())
            finished.wait(.1)
    sampling = threading.Thread(target=sample); sampling.start()
    try:
        with open(args.directory / 'playwright.json', 'xb', opener=lambda p, flags: os.open(p, flags, 0o600)) as out:
            with open(args.directory / 'runner.stderr', 'xb', opener=lambda p, flags: os.open(p, flags, 0o600)) as err:
                run = subprocess.run(['npx', '--no-install', 'playwright', 'test', '--config', 'playwright.native-session.config.ts', '--reporter=json'],
                    cwd=ROOT / 'web', env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'), stdout=out, stderr=err, timeout=250)
    finally:
        finished.set(); sampling.join(timeout=10)
        private_write(args.directory / 'memory.json', (json.dumps({'samples': samples, 'query_failures': failures}, indent=2) + '\n').encode())
    sources = [ROOT / 'src/talkinghead_v2/server/native_media.py', ROOT / 'src/talkinghead_v2/server/actor.py',
               ROOT / 'native/media/session_worker.cpp', ROOT / 'web/tests/native-session.spec.ts', ROOT / 'web/src/client.ts']
    private_write(args.directory / 'run.json', (json.dumps({'returncode': run.returncode,
        'source_sha256': {path.relative_to(ROOT).as_posix(): digest(path) for path in sources},
        'executable_sha256': digest(ROOT / 'var/build/native-media/tha_session_media'),
        'memory_sampling_interval_target_ms': 100, 'memory_sampling_queries_sequential': True,
        'physical_av_qualified': False}, indent=2) + '\n').encode())
    print(json.dumps({'browser_exit': run.returncode, 'memory_samples': len(samples), 'sample_failures': len(failures)}))
    raise SystemExit(run.returncode)


if __name__ == '__main__': main()
