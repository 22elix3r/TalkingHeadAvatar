"""Qualify the declared synthetic virtual-device loop, including received speech."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from talkinghead_v2.local import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite evidence')
    directory = ROOT / 'var/p06/virtual'
    browser = json.loads(args.browser_report.read_text())
    stats = browser['stats']
    if stats['expected'] != 2 or any(stats[key] for key in ['unexpected', 'skipped', 'flaky']):
        raise ValueError('virtual-device browser checks did not pass')
    fixture_path = ROOT / 'var/p06/virtual-input/microphone-fixture.json'
    fixture = json.loads(fixture_path.read_text())
    ownership = json.loads((directory / 'owner-contract.json').read_text())
    if ownership.get('shared_device_owner_rejected') is not True:
        raise ValueError('exclusive project camera ownership was not qualified')
    if fixture['kind'] != 'synthetic_microphone_fixture' or digest(fixture_path.parent / 'microphone.wav') != fixture['sha256']:
        raise ValueError('synthetic input provenance/hash does not match')
    cohorts = {}
    for name in ['eof', 'cancel']:
        case = json.loads((directory / (name + '.json')).read_text())
        if not case.get('source_sha256_at_run') or case['runtime_binary_sha256'] != digest(ROOT / 'var/build/virtual-reference/tha_virtual_reference'):
            raise ValueError('missing/stale runtime binary fingerprint')
        for path, expected in case['source_sha256_at_run'].items():
            if path not in {'native/virtual/reference.cpp', 'native/virtual/CMakeLists.txt', 'web/tests/virtual-device.spec.ts',
                'web/tests/virtual-fixtures.ts', 'web/playwright.virtual.config.ts', 'web/playwright.virtual.firefox.config.ts'} or digest(ROOT / path) != expected:
                raise ValueError('source changed after the browser cohort; rerun before export')
        if not case['defaults_unchanged'] or not case['camera_format_unchanged'] or not case['receiver']['exactDevices'] or not case['source']['camera_cleared']:
            raise ValueError('device isolation or clear evidence failed')
        if case['source']['cancelled'] != (name == 'cancel') or case['receiver']['audioPeak'] <= .001:
            raise ValueError('source/received audio lifecycle evidence failed')
        state = json.loads((directory / (name + '-browser-state.json')).read_text())
        video_stats = [item for item in state['stats'] if item['type'] == 'inbound-rtp' and item['kind'] == 'video']
        cohorts[name] = {'source': case['source'], 'source_pcm_sha256': case['pcm_sha256'],
            'browser': re.search(r'Firefox/([\d.]+)', case['browser_user_agent']).group(0),
            'capture_dimensions': [case['receiver']['width'], case['receiver']['height']],
            'inbound_dimensions': [state['video']['width'], state['video']['height']],
            'presented_frames': state['frames'], 'presented_nonblack_frames': state['nonblack'],
            'receiver_peak': state['audioPeak'], 'exact_virtual_devices': True,
            'final_receiver_black': bool(state['colors']) and all(value < 20 for value in state['colors'][-1]),
            'frames_decoded': sum(item.get('framesDecoded', 0) for item in video_stats),
            'case_sha256': digest(directory / (name + '.json'))}
        if not cohorts[name]['final_receiver_black']:
            raise ValueError('receiver never acknowledged camera clear')
    capture = directory / 'received-synthetic-speech.media'
    if capture.is_symlink() or not capture.is_file() or capture.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('bounded private receiver recording required')
    pcm = subprocess.check_output(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(capture), '-t', '15',
        '-ar', '16000', '-ac', '1', '-f', 's16le', 'pipe:1'], timeout=20)
    from talkinghead_v2.server.config import Settings
    from talkinghead_v2.server.providers import Asr
    settings = Settings.load(ROOT / 'var/p04/runtime.json')
    asr = Asr(settings.asr_path, settings.asr_files)
    try:
        transcript = asr.decode(pcm)
    finally:
        asr.close()
    normalized = re.sub(r'[^a-z ]', ' ', transcript.lower().replace('2', 'two').replace('+', ' plus '))
    if not re.search(r'\btwo\s+plus\s+two\b', normalized):
        raise ValueError('received synthetic speech content check failed')
    modules = subprocess.check_output(['pactl', 'list', 'short', 'modules'], text=True, timeout=5)
    if 'tha_v2_probe_' in modules:
        raise ValueError('temporary virtual-device modules remain loaded')
    sources = [*sorted((ROOT / 'native/virtual').glob('*')), Path(__file__), ROOT / 'scripts/create_p04_microphone_fixture.py',
        *[ROOT / path for path in ['web/playwright.virtual.config.ts', 'web/playwright.virtual.firefox.config.ts',
            'web/tests/virtual-device.spec.ts', 'web/tests/virtual-fixtures.ts']]]
    report = {'schema_version': 1, 'phase': 'P06', 'phase_status': 'in_progress',
        'scope': 'CPU-copy virtual camera/microphone reference, disposable Firefox → local LiveKit → receiver; not live Qwen/native GPU product integration',
        'browser_report_sha256': digest(args.browser_report), 'browser_counts': {key: stats[key] for key in ['expected', 'unexpected', 'skipped', 'flaky', 'duration']},
        'source_sha256': {path.relative_to(ROOT).as_posix(): digest(path) for path in sources if path.is_file()},
        'binary_sha256': digest(ROOT / 'var/build/virtual-reference/tha_virtual_reference'),
        'environment': {'v4l2loopback_module': subprocess.check_output(['modinfo', '-F', 'version', 'v4l2loopback'], text=True, timeout=5).strip(),
            'libpulse_simple': subprocess.check_output(['pkg-config', '--modversion', 'libpulse-simple'], text=True, timeout=5).strip()},
        'fixture': fixture, 'cohorts': cohorts,
        'camera_owner_contract': ownership,
        'received_speech': {'recording_sha256': digest(capture), 'decoded_16k_samples': len(pcm) // 2,
            'asr': 'existing pinned faster-whisper-base.en, CPU int8, local_files_only',
            'transcript': transcript, 'known_phrase_content_check': True, 'quality_or_wer_gate': False},
        'temporary_modules_retired': True, 'host_driver_or_default_device_changed': False,
        'browser_profile': 'new temporary profile only; localhost camera/microphone grants, legacy enumeration, real capture; loopback ICE remains disabled',
        'chromium_status': 'camera not enumerated with current combined capture/output device capabilities; microphone enumerates; no kernel reload performed',
        'acceptance': {'virtual_reference_delivery': True, 'third_party_conference_app': False,
            'live_qwen_to_virtual_devices': False, 'physical_av_skew': False, 'arbitrary_client_cancel_flush': False,
            'hard_kill_camera_privacy': False,
            'native_product_integration': False, 'release': False},
        'copies': 'CPU BGRA frame → V4L2 write; browser conversion/VP8 encoding; PCM → PulseAudio/PipeWire; independent from CUDA no-readback trace'}
    with args.output.open('x') as target:
        target.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'virtual_reference': 'passed', 'received_phrase': transcript, 'phase': 'in_progress'}))


if __name__ == '__main__':
    main()
