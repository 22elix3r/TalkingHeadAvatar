"""Export bounded synthetic P06 observations; never turn them into a release pass."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser-report', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True, help='Nsight SQLite export')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite evidence')
    directory = ROOT / 'var/p06'
    browser = read(args.browser_report)
    stats = browser['stats']
    browser_pass = stats['expected'] == 2 and stats['unexpected'] == stats['skipped'] == stats['flaky'] == 0
    cases = {}
    for name in ['baseline', 'recovery']:
        path = directory / f'native-{name}-evidence.json'
        data = read(path)
        frames = data['receiver']['mediaTimes']
        ids = {frame['frameId'] for frame in frames if 30 <= frame.get('frameId', -1) < 270}
        video = [item for item in data['receiver']['stats'] if item['type'] == 'inbound-rtp' and item['kind'] == 'video']
        memory = data['process_memory']
        cases[name] = {'evidence_sha256': sha(path), 'sender': data['native'],
            'decoded_audio_observed': data['receiver']['decodedAudio'],
            'presented_valid_frame_ids': len(ids), 'valid_frames_sent': 240,
            'presented_valid_fraction': len(ids) / 240,
            'inbound_video': [{key: value for key, value in item.items() if key in
                ['framesDecoded', 'framesDropped', 'packetsLost', 'keyFramesDecoded', 'freezeCount', 'totalDecodeTime']}
                for item in video],
            'process_memory': memory, 'encoder_512_mib_budget_pass': memory['sampled_peak_mib'] is not None and memory['sampled_peak_mib'] <= 512,
            'presentation_cadence_diagnostic_pass': len(ids) >= 237 if name == 'baseline' else None}
    with sqlite3.connect(f'file:{args.trace.resolve()}?mode=ro', uri=True) as trace:
        tables = {row[0] for row in trace.execute("select name from sqlite_master where type='table'")}
        runtime = 'CUPTI_ACTIVITY_KIND_RUNTIME'
        if runtime not in tables or 'CUPTI_ACTIVITY_KIND_KERNEL' not in tables:
            raise ValueError('missing CUDA activities; empty trace cannot establish copy behavior')
        calls = list(trace.execute(f'select s.value,count(*) from {runtime} r join StringIds s on r.nameId=s.id group by s.value'))
        kernels = dict(trace.execute('select s.value,count(*) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id group by s.value'))
        if kernels.get('render_pattern') != 300 or kernels.get('rgba_to_nv12') != 300:
            raise ValueError('trace is not the declared 300-frame cohort')
        copies = {name: count for name, count in calls if 'Memcpy' in name}
        allocations = {name: count for name, count in calls if 'MemAlloc' in name}
        first_kernel = trace.execute('select min(start) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
        late_allocations = trace.execute(f"select count(*) from {runtime} r join StringIds s on r.nameId=s.id where s.value like '%MemAlloc%' and r.start>?", (first_kernel,)).fetchone()[0]
        memcpy_tables = sorted(name for name in tables if name.startswith('CUPTI_ACTIVITY_KIND_MEMCPY'))
        copy_count = sum(trace.execute(f'select count(*) from {name}').fetchone()[0] for name in memcpy_tables)
    probe = json.loads(next(line for line in (directory / 'encoder-trace-metadata.log').read_text().splitlines() if line.startswith('{')))
    ffprobe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-show_entries',
        'stream=codec_name,profile,width,height,pix_fmt,level,color_range,color_space,color_transfer,color_primaries,nb_read_frames',
        '-of', 'json', str(directory / 'pattern-traced-metadata.h264')], text=True))
    sources = [*sorted((ROOT / 'native/media').glob('*')), Path(__file__),
        ROOT / 'scripts/build_livekit_cpp.py', ROOT / 'web/tests/native-media.spec.ts', ROOT / 'web/playwright.native.config.ts']
    webrtc = list((ROOT / 'var/sources/livekit-cpp-v1.11.0/client-sdk-rust/target/release/build').glob('scratch-*/out/livekit_webrtc/livekit/linux-x64-release-webrtc-89d790b/linux-x64-release/lib/libwebrtc.a'))
    if len(webrtc) != 1:
        raise ValueError('expected one qualified WebRTC binary')
    report = {'schema_version': 1, 'phase': 'P06', 'phase_status': 'in_progress',
        'scope': 'synthetic standalone native publisher; not the product Qwen/session/mesh adapter',
        'browser_report_sha256': sha(args.browser_report), 'browser_checks_pass': browser_pass,
        'source_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in sources if path.is_file()},
        'sdk_build': read(directory / 'livekit-build.json'), 'gpu_contract': read(directory / 'encoder-contract.json'),
        'webrtc': {'tag': 'webrtc-89d790b', 'static_library_sha256': sha(webrtc[0]),
            'license_notice_sha256': sha(webrtc[0].parent.parent / 'LICENSE.md')},
        'probe': probe, 'decoded_file': ffprobe['streams'], 'browser': cases,
        'cuda_trace': {'sha256': sha(args.trace), 'kernels': kernels, 'allocation_api_calls': allocations,
            'allocation_calls_after_first_kernel': late_allocations, 'memcpy_api_calls': copies,
            'memcpy_activity_count': copy_count,
            'no_raw_readback_observed': not copies and copy_count == 0,
            'boundary': 'CUDA-visible activity in this isolated 300-frame encoder process only; CPU encoded-payload copies remain'},
        'acceptance': {'product_native_integration': False, 'physical_audio_video_skew': False,
            'qwen_motion_render_coload': False, 'loading_memory_peak': False, 'cancel_epoch_churn': False,
            'fresh_local_provision': False, 'conference_virtual_devices': False, 'release': False},
        'weights': 'no model provisioning performed for P06; software SDK/dependencies fetched',
        'cold_start_finding': 'earlier unprepared baseline decoded 236/240 with zero packet loss; retained private separated-cohort report',
        'preparation': '30 black frames and 48000 zero samples precede 240 valid pattern frames/384000 valid tone samples; never response latency credit'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as target:
        target.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'phase': 'in_progress', 'browser_checks_pass': browser_pass,
        'encoder_budget_pass': all(case['encoder_512_mib_budget_pass'] for case in cases.values())}))


if __name__ == '__main__':
    main()
