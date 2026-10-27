"""Export actual N=1 admission/lifetime evidence; no capacity or sharing claim."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from talkinghead_v2.local import digest, request_json
from talkinghead_v2.session.device import DeviceLease

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('refusing to overwrite evidence')
    directory = ROOT / 'var/p06'
    checks = {}
    for name, minimum in [('device-core-tests-final.xml', 241), ('device-api-tests-final.xml', 20)]:
        totals = {key: 0 for key in ['tests', 'failures', 'errors', 'skipped']}
        for suite in ET.parse(directory / name).getroot().iter('testsuite'):
            for key in totals: totals[key] += int(suite.get(key, 0))
        if totals['tests'] < minimum or any(totals[key] for key in ['failures', 'errors', 'skipped']):
            raise ValueError('incomplete/failing device tests')
        checks[name] = {**totals, 'sha256': digest(directory / name)}
    browser = {}
    for name, expected in [('playwright-device-admission-final.json', 1), ('playwright-device-chromium-regression.json', 8), ('playwright-device-firefox-regression.json', 1)]:
        stats = json.loads((directory / name).read_text())['stats']
        if stats['expected'] != expected or any(stats[key] for key in ['unexpected', 'skipped', 'flaky']):
            raise ValueError('incomplete/failing real-provider browser tests')
        browser[name] = {key: stats[key] for key in ['expected', 'unexpected', 'skipped', 'flaky', 'duration']}
    evidence = json.loads((directory / 'device-browser.json').read_text())
    competitor = evidence['competitor']
    if not competitor['standalone_provider_rejected'] or competitor['competing_spawn_attempted'] or competitor['real_qwen_child_lease_holders'] != 1:
        raise ValueError('competing worker was not stopped before spawn')
    for key in ['cross_profile_busy_visible', 'actual_qwen_after_rejection', 'actual_qwen_after_handoff']:
        if evidence[key] is not True: raise ValueError('actual session handoff is incomplete')
    source = ['src/talkinghead_v2/session/device.py', 'src/talkinghead_v2/providers/tts/native.py', 'src/talkinghead_v2/server/actor.py',
        'web/src/client.ts', 'web/tests/device-admission.spec.ts', 'web/playwright.device.config.ts',
        'tests/test_device_lease.py', 'tests/test_native_provider.py', 'runtimes/session-api/tests/test_control_api.py',
        'scripts/check_device_admission.py', 'scripts/qualify_device_lease.py']
    startup = json.loads((directory / 'preview-device-owner.json').read_text())['source_sha256']
    if any(digest(ROOT / path) != expected for path, expected in startup.items()):
        raise ValueError('preview source changed after startup; revalidate loaded code')
    lease = DeviceLease(ROOT / 'var/gpu-device-0.lock'); lease.acquire('qualification-complete'); lease.release()
    report = {'schema_version': 1, 'phase': 'P06', 'phase_status': 'in_progress', 'scope': competitor['scope'],
        'source_sha256': {path: digest(ROOT / path) for path in source}, 'loaded_preview_source_sha256': startup,
        'tests': checks, 'browser': browser, 'competitor': competitor,
        'cross_profile_busy_visible': True, 'real_primary_playback_after_rejection': True, 'real_secondary_playback_after_handoff': True,
        'device_lease_available_after_sessions': True, 'preview_ready': request_json('http://127.0.0.1:18700/health/ready')['ready'],
        'lifetime': 'parent and native child share one inherited flock description; no LOCK_UN or lock-inode replacement on release',
        'limits': {'admitted_sessions': 1, 'shared_model_weights_qualified': False, 'unmanaged_cuda_process_isolation': False,
            'multiple_runtime_directory_or_gpu_uuid_qualification': False, 'native_media_coload_qualified': False,
            'real_avatar_quality_qualified': False, 'capacity_sweep_or_soak_qualified': False, 'release': False},
        'impact': 'first-party admission/startup/cleanup only; no model/config/checkpoint/dependency changes or downloads'}
    with args.output.open('x') as target: target.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'device_admission': 'passed', 'admitted_sessions': 1, 'release': False}))


if __name__ == '__main__': main()
