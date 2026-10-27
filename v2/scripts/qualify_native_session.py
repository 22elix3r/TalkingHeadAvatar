"""Export native product checks with scoped memory and presentation evidence."""
import argparse
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

from talkinghead_v2.local import digest
from talkinghead_v2.server.native_media import NativeMediaConfig
from talkinghead_v2.session.device import DeviceLease

ROOT = Path(__file__).resolve().parents[1]


def passed_browser(path, expected):
    data = json.loads(path.read_text())
    stats = data['stats']
    if stats['expected'] != expected or any(stats[name] for name in ('unexpected', 'skipped', 'flaky')) or data.get('errors'):
        raise ValueError('browser cohort is incomplete/failed/retried')
    return {'passed': expected, 'duration_ms': stats['duration'], 'report_sha256': digest(path)}


def passed_cpu(path, expected):
    suites = list(ET.parse(path).getroot().iter('testsuite'))
    if sum(int(suite.get('tests', 0)) for suite in suites) != expected or any(int(suite.get(name, 0)) for suite in suites for name in ('errors', 'failures', 'skipped')):
        raise ValueError('CPU/API cohort incomplete or failed')
    return {'passed': expected, 'junit_sha256': digest(path)}


def observe(path, interrupted):
    data = json.loads(path.read_text()); events = data['observations']
    bindings = [event for event in events if event['kind'] == 'playback_binding']
    if not bindings or any(binding.get('transport') != 'livekit_cpp_preencoded' for binding in bindings): raise ValueError('native transport not observed')
    if interrupted and (len(bindings) < 2 or any(bindings[0][name] == bindings[-1][name] for name in ('epoch', 'audio_sid', 'video_sid'))):
        raise ValueError('replacement-track recovery missing')
    if not any(event['kind'] == 'video_presented_callback' for event in events): raise ValueError('no presented video callback')
    if len([event for event in events if event['kind'] == 'decoded_audio_observed']) < (2 if interrupted else 1): raise ValueError('decoded audio missing')
    completed = [event for event in events if event['kind'] == 'sender_complete' and event['epoch'] == bindings[-1]['epoch']]
    if len(completed) != 1 or completed[0]['presentation_ack'] is not False: raise ValueError('final source EOF missing')
    eof = completed[0]; metrics = eof['native_source_metrics']
    if not eof['valid_audio_samples'] > 0 or not 960 <= eof['codec_padding_samples'] < 1440 or not 0 < metrics['queue_peak_samples'] <= 48000:
        raise ValueError('EOF/queue accounting failed')
    return {'evidence_sha256': digest(path), 'native_binding_count': len(bindings), 'replacement_tracks': interrupted,
        'source_valid_samples': eof['valid_audio_samples'], 'source_codec_padding_samples': eof['codec_padding_samples'],
        'source_metrics': metrics, 'decoded_audio_observed': True, 'video_presented_callback': True, 'presentation_ack': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists(): raise ValueError('refusing to overwrite evidence')
    run = json.loads((args.directory / 'run.json').read_text())
    if run['returncode'] or any(digest(ROOT / name) != expected for name, expected in run['source_sha256'].items()):
        raise ValueError('browser run failed or its source changed')
    runtime = NativeMediaConfig.load(ROOT / 'var/p06/native-media-runtime.json')
    if digest(runtime.executable) != run['executable_sha256']: raise ValueError('native executable changed after run')
    worker = json.loads((ROOT / 'benchmarks/reports/p06-native-worker.json').read_text())
    if len(worker['cases']) != 11 or any(digest(ROOT / name) != expected for name, expected in worker['source_sha256'].items()):
        raise ValueError('worker qualification stale/incomplete')
    memory = json.loads((args.directory / 'memory.json').read_text())
    if memory['query_failures'] or len(memory['samples']) < 20: raise ValueError('memory sampling incomplete')
    samples = memory['samples']
    coload = [sample for sample in samples if all(any(process['category'] == name for process in sample['processes']) for name in ('native_media', 'qwen_worker'))]
    if not coload: raise ValueError('no actual Qwen/native co-load observed')
    def peak(category): return max(sum(process['memory_mib'] for process in sample['processes'] if process['category'] == category) for sample in samples)
    owner = json.loads((ROOT / 'var/p06/preview-device-owner.json').read_text())
    if any(digest(ROOT / name) != expected for name, expected in owner['source_sha256'].items()): raise ValueError('preview guard source changed after startup')
    with urllib.request.urlopen('http://127.0.0.1:18700/health/ready', timeout=3) as response:
        if not json.load(response)['ready']: raise ValueError('normal preview is not ready')
    lease = DeviceLease(ROOT / 'var/gpu-device-0.lock'); lease.acquire('native-session-export-post-retirement'); lease.release()
    report = {'schema_version': 1, 'phase': 'P06', 'phase_status': 'in_progress',
        'scope': 'actual cached Qwen/CPU ASR+LLM/browser conversation with CUDA static mesh, NVENC and LiveKit C++ pre-encoded publications',
        'source_sha256': run['source_sha256'], 'native_executable_sha256': run['executable_sha256'], 'ptx_sha256': digest(runtime.ptx),
        'checks': {'core': passed_cpu(ROOT / 'var/p06/native-session-core-tests-frozen.xml', 260),
            'api': passed_cpu(ROOT / 'var/p06/native-session-api-tests-restored.xml', 24),
            'chromium': passed_browser(args.directory / 'playwright.json', 2),
            'firefox': passed_browser(ROOT / 'var/p06/playwright-native-session-firefox-run.json', 1),
            'actual_worker_protocol_cases': 11, 'post_retirement_device_reacquired': True, 'normal_preview_ready': True},
        'journeys': {'chromium_typed': observe(ROOT / 'var/p06/native-session-browser-chromium-final/typed.json', True),
            'chromium_microphone': observe(ROOT / 'var/p06/native-session-browser-chromium-final/microphone.json', False),
            'firefox_typed': observe(ROOT / 'var/p06/native-session-browser/typed.json', True)},
        'memory': {'samples': len(samples), 'coload_samples': len(coload), 'query_failures': 0,
            'target_interval_ms': 100, 'queries': 'sequential device/process NVML snapshots; sampled maxima are not guaranteed peak memory',
            'device_total_mib': samples[0]['device_total_mib'], 'sampled_total_device_used_max_mib': max(sample['device_used_mib'] for sample in samples),
            'qwen_process_sampled_max_mib': peak('qwen_worker'), 'native_process_sampled_max_mib': peak('native_media'),
            'other_compute_processes_sampled_max_mib': peak('other'),
            'model_plus_media_same_process_query_max_mib': max(sum(p['memory_mib'] for p in sample['processes'] if p['category'] in ('native_media', 'qwen_worker')) for sample in coload),
            'planned_encoder_budget_mib': 512, 'encoder_budget_passed': peak('native_media') <= 512,
            'motion_present': False, 'gpu_llm_present': False, 'capacity_qualified': False},
        'clock': {'source_rate': 48000, 'pcm_quantum_samples': 960, 'worker_audio_tick_samples': 480,
            'prebuffer_ms': runtime.prebuffer_ms, 'worker_queue_bound_samples': 48000, 'video_sample_tick': 1600,
            'preparation_silence': 'all source samples before first_pcm_stream_sample; excluded from valid PCM, included in real startup',
            'clock_domains': 'controller/browser/C++ source clocks remain distinct; no physical A/V subtraction or presentation acknowledgment'},
        'cancellation': 'priority SIGTERM bypasses PCM backpressure; owned process retired before replacement; exact new audio/video SIDs gate receiver resume',
        'copies': 'three initial geometry uploads and GPU render/NV12 surfaces; encoded CPU payload copies; prior isolated trace, no new product-process trace claim',
        'environment': 'host reboot stopped services during the gap; restored only this project services and retained databases; API tests rerun after PostgreSQL recovery',
        'model_impact': 'existing pinned Base 1.7B X-vector synthetic voice; no new checkpoint/model/software download, training, human media or V1 changes',
        'remaining': ['real Claire/authorized speaking-avatar and calibrated retarget', 'independent Q0/human voice lifecycle', 'physical A/V and latency',
            'encoder memory budget', 'API/inference image qualification', 'live virtual-device output/third-party conference/fail-stop'],
        'acceptance': {'native_product_transport': True, 'static_reference_only': True, 'real_speaking_avatar': False, 'physical_av': False,
                       'capacity': False, 'quality_gate': False, 'release': False}}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'native_product_transport': True, 'browser_checks': 3, 'sampled_device_peak_mib': report['memory']['sampled_total_device_used_max_mib'], 'release': False}))


if __name__ == '__main__': main()
