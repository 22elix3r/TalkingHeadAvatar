"""Export sanitized evidence of the exercised offline local-profile lifecycle."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import xml.etree.ElementTree as ET

from talkinghead_v2.local import digest, load_profile, request_json, safe_read

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--browser-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite evidence')
    directory, marker = load_profile(ROOT, args.name)
    private = ROOT / 'var/p06'
    junit = private / 'local-core-tests.xml'
    suites = ET.parse(junit).getroot().iter('testsuite')
    counts = {key: 0 for key in ['tests', 'failures', 'errors', 'skipped']}
    for suite in suites:
        for key in counts:
            counts[key] += int(suite.get(key, 0))
    if counts['tests'] < 227 or any(counts[key] for key in ['failures', 'errors', 'skipped']):
        raise ValueError('CPU profile qualification failed or is incomplete')
    browser = json.loads(args.browser_report.read_text())
    stats = browser['stats']
    if stats['expected'] != 2 or any(stats[key] for key in ['unexpected', 'skipped', 'flaky']):
        raise ValueError('browser qualification failed or is incomplete')
    before = json.loads((private / 'local-db-before-final.json').read_text())
    after = json.loads((private / 'local-db-after-final.json').read_text())
    if before != after or after['migration'] != 'p05_0003' or after['sessions'].get('CLOSED', 0) < 2:
        raise ValueError('database restart preservation was not established')
    fault = json.loads((private / 'local-launcher-fault.json').read_text())
    if not all(fault.values()) or len(fault) != 4:
        raise ValueError('launcher fault evidence is incomplete')
    stopped = json.loads((private / 'local-down-final.json').read_text())
    if stopped['status'] != 'stopped' or not stopped['database_volume_retained'] or (directory / 'api-process.json').exists():
        raise ValueError('owned profile stop was not established')
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(('127.0.0.1', marker['ports']['api'])) == 0:
            raise ValueError('qualification API is still listening')
    volume = subprocess.check_output(['docker', 'volume', 'inspect', '--format={{.Name}}', marker['project'] + '_pgdata'], text=True).strip()
    if volume != marker['project'] + '_pgdata':
        raise ValueError('owned database volume is missing')
    runtime = safe_read(directory / 'runtime.json')
    sources = [ROOT / path for path in ['src/talkinghead_v2/local.py', 'src/talkinghead_v2/local_server.py',
        'src/talkinghead_v2/cli.py', 'scripts/serve_p04.py', 'scripts/qualify_local_profile.py',
        'tests/test_local_profiles.py', 'web/playwright.local.config.ts', 'web/tests/local-profile.spec.ts', 'web/tests/ready.ts']]
    report = {'schema_version': 1, 'phase': 'P06', 'phase_status': 'in_progress',
        'scope': 'fresh loopback profile and offline software environment using existing model/image caches; not a bare-machine or containerized-API release',
        'source_sha256': {path.relative_to(ROOT).as_posix(): digest(path) for path in sources},
        'runtime_lock_sha256': digest(ROOT / 'runtimes/session-api/uv.lock'),
        'runtime_descriptor_sha256': digest(Path(runtime['tts_runtime'])),
        'profile_config_sha256': marker['files'], 'profile_ports': marker['ports'],
        'asr_files_sha256': runtime['asr_files'], 'browser_report_sha256': digest(args.browser_report),
        'cpu_tests': counts, 'cpu_junit_sha256': digest(junit),
        'browser': {key: stats[key] for key in ['expected', 'unexpected', 'skipped', 'flaky', 'duration']},
        'database_restart_before': before, 'database_restart_after': after,
        'launcher_fault': fault, 'stop': stopped, 'database_volume_verified_retained': True,
        'native_media_integrated': False, 'normal_preview_ready_at_export': request_json('http://127.0.0.1:18700/health/ready')['ready'],
        'provisioning': {'software': 'uv sync --locked --offline, separate per-profile environment',
            'images': 'digest-pinned, pull_policy never and --pull never; existing local PG/SFU images',
            'models': 'existing readonly ASR/Qwen inputs shared; no checkpoint fetch or model pull',
            'bundled_dependency_weight': 'offline runtime install reuses the previously cached, unused faster-whisper Silero ONNX file; not a new network download'},
        'remaining': ['native product adapter', 'API runtime image', 'bare-machine offline cache bootstrap',
            'virtual-device conference loop', 'cross-profile GPU admission', 'real avatar/physical A-V/quality gates']}
    with args.output.open('x') as target:
        target.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'profile_cycle': 'passed', 'phase': 'in_progress', 'release': False}))


if __name__ == '__main__':
    main()
