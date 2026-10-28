"""Run real synthetic/SFU queue probes and verify their complete process exits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main(args):
    if args.directory.exists() or args.report.exists():
        raise ValueError('preserve previous native audio evidence')
    args.directory.mkdir(parents=True, mode=0o700)
    child = ROOT / 'scripts/qualify_native_audio_queue.py'
    source_hash = digest(child)
    libraries = {mode: path.resolve(strict=True) for mode, path in [('patched', args.ffi), ('stock', args.stock_ffi)]}
    hashes = {mode: digest(path) for mode, path in libraries.items()}
    cases = []
    specs = [(f'{mode}-{index}', mode, 'patched') for index in range(5) for mode in ('normal', 'overflow')]
    specs.append(('eviction-control', 'eviction_control', 'stock'))
    for name, mode, kind in specs:
        output = args.directory / f'{name}.json'
        environment = dict(os.environ, LIVEKIT_LIB_PATH=str(libraries[kind]), RUST_BACKTRACE='full',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        for key in ('LD_PRELOAD', 'LD_AUDIT'):
            environment.pop(key, None)
        log_path = args.directory / f'{name}.log'
        with open(log_path, 'xb', opener=lambda path, flags: os.open(path, flags | os.O_NOFOLLOW, 0o600)) as log:
            try:
                process = subprocess.run([sys.executable, str(child), '--mode', mode, '--native-frames', '25',
                    '--ffi', str(libraries[kind]), '--ffi-sha256', hashes[kind], '--config', str(args.config.resolve()),
                    '--output', str(output.resolve())], env=environment, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, timeout=30)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                exit_code = 'timeout'
        log_data = log_path.read_bytes()
        panic = b'panicked at' in log_data or b'non-unwinding panic' in log_data
        row = {'name': name, 'process_exit': exit_code, 'panic_detected': panic}
        if output.exists():
            row['observation'] = json.loads(output.read_text())
        cases.append(row)
        print(json.dumps({'case': name, 'exit': exit_code, 'panic': panic}), flush=True)
        if exit_code != 0 or panic or not output.exists() or digest(child) != source_hash:
            break
    passed = len(cases) == len(specs) and all(row['process_exit'] == 0 and not row['panic_detected'] for row in cases)
    report = {'schema_version': 1, 'status': 'passed' if passed else 'failed', 'cases': cases,
              'probe_sha256': source_hash, 'runner_sha256': digest(Path(__file__)), 'ffi_sha256': hashes,
              'scope': 'Local native/Python audio queue and process-lifetime evidence with synthetic PCM, not browser or ASR quality',
              'model_downloads': False, 'physical_presentation_qualified': False}
    args.report.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(args.report, 'x', opener=lambda path, flags: os.open(path, flags | os.O_NOFOLLOW, 0o600)) as target:
        target.write(json.dumps(report, indent=2) + '\n')
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ffi', 'stock-ffi', 'config', 'directory', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
