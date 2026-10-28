"""Matched source-pinned receiver diagnostics; real cached Qwen/Claire, no fetch."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from talkinghead_v2.server.inbound_config import InboundAudioConfig, file_hash

ROOT = Path(__file__).resolve().parents[1]


def live_group_members(group):
    members=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            fields=(path/'stat').read_text().rsplit(')',1)[1].split()
            if fields[0] not in {'Z','X'} and int(fields[2])==group:members.append(int(path.name))
        except (OSError,ValueError,IndexError):pass
    return members


def fence_group(process,grace):
    if not live_group_members(process.pid):return False
    # The unreaped leader reserves this group ID across exit and PID reuse.
    for sig,delay in ((signal.SIGTERM,grace),(signal.SIGKILL,0)):
        try:os.killpg(process.pid,sig)
        except ProcessLookupError:pass
        if delay:time.sleep(delay)
    deadline=time.monotonic()+10
    while live_group_members(process.pid):
        if time.monotonic()>=deadline:raise RuntimeError('owned qualifier group did not terminate; inspect its private log')
        time.sleep(.01)
    return True


def run_owned_process(command, environment, log, *, timeout=180, grace=2):
    process=subprocess.Popen(command,env=environment,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        deadline=time.monotonic()+timeout
        while True:
            observed=os.waitid(os.P_PID,process.pid,os.WEXITED|os.WNOHANG|os.WNOWAIT)
            if observed is not None:break
            if time.monotonic()>=deadline:
                fence_group(process,grace)
                process.wait(timeout=10)
                return 'timeout'
            time.sleep(.02)
        # Keep the leader unreaped until any descendants left by a panic,
        # signal or premature success are stopped and actually release FDs.
        fenced=fence_group(process,grace)
        code=process.wait(timeout=10)
        return 'descendants_after_success' if code==0 and fenced else code
    except BaseException:
        if process.returncode is None:
            fence_group(process,grace)
            process.wait(timeout=10)
        raise


def main(args):
    if args.directory.exists() or args.report.exists():
        raise ValueError('preserve previous receiver matrix evidence')
    runtime = InboundAudioConfig.load(args.receiver_runtime)
    child = ROOT / 'scripts/qualify_claire_session_worker.py'
    source_hash = file_hash(child)
    runner_hash = file_hash(Path(__file__))
    args.directory.mkdir(parents=True, mode=0o700)
    specs = [(f'{decoder}-{queue}-{index}', decoder, queue, 0)
             for index in range(1,args.repeats+1) for queue in (1, 8) for decoder in ('auto', 'disable-nvdec')]
    specs.append(('overflow-control', 'auto', 1, 500))
    cases = []
    for name, decoder, queue, stall in specs:
        report_path = args.directory / (name + '.json')
        log_path = args.directory / (name + '.log')
        command = [sys.executable, str(child), '--live-qwen', '--trace-source-captures',
            '--trace-receiver-ffi', '--receiver-runtime', str(args.receiver_runtime.resolve()),
            '--receiver-decoder', decoder, '--native-video-queue-frames', str(queue),
            '--stall-video-ffi-ms', str(stall), '--directory', str((args.directory / name).resolve()),
            '--report', str(report_path.resolve())]
        for key in ('config', 'engine', 'scene'):
            command.extend(['--' + key, str(getattr(args, key).resolve())])
        environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        for key in ('LIVEKIT_LIB_PATH', 'LK_DISABLE_NVDEC', 'LD_PRELOAD', 'LD_AUDIT'):
            environment.pop(key, None)
        with open(log_path, 'xb', opener=lambda path, flags: os.open(path, flags | os.O_NOFOLLOW, 0o600)) as log:
            code = run_owned_process(command, environment, log)
        log_data = log_path.read_bytes()
        panic = any(word in log_data for word in (b'panicked at', b'non-unwinding panic', b'FFI Panic'))
        row = {'name': name, 'requested_decoder': decoder, 'native_video_queue_frames': queue,
               'process_exit': code, 'panic_detected': panic, 'log_sha256': file_hash(log_path)}
        valid = code == 0 and not panic and report_path.is_file() and report_path.stat().st_size <= 4 * 1024 * 1024
        if valid:
            observed = json.loads(report_path.read_text())
            case = observed['cases'][0]
            receiver, sender = case['receiver'], case['sender']
            boundary = receiver['ffi_video_boundary']
            selected = observed['receiver_runtime']
            row.update({'source_report_sha256': file_hash(report_path), 'sender_source_frames': sender['source_video_frames'],
                'valid_audio_samples': sender['valid_samples'], 'consumer_source_frames': len(receiver['video']),
                'ffi_source_frames': len(boundary['source_frames']),
                'missing_before_ffi_callback': boundary['missing_before_ffi_callback'],
                'ffi_callbacks_missing_at_consumer': boundary['ffi_callbacks_missing_at_consumer'],
                'native_video_warning_events': receiver['native_video_drop_warning_events'],
                'native_video_drop_reported_lower_bound': receiver['native_video_drop_reported_lower_bound'],
                'injected_callback_stall_ns': boundary['injected_callback_stall_ns'],
                'checked_queue_peak_frames': receiver['checked_queue_peak_frames'],
                'video_rtp_stats': receiver['video_rtp_stats'], 'receiver_runtime': selected,
                'sender_rtp_stats': case.get('sender_rtp_stats'), 'sender_rtp_after_300ms': case.get('sender_rtp_after_300ms'),
                'memory': observed['memory']})
            valid = (selected['source_matched'] is True and selected['library_sha256'] == runtime.library_sha256 and
                selected['descriptor_sha256'] == runtime.revision and selected['nvdec_disabled_by_environment'] == (decoder == 'disable-nvdec') and
                observed['source_sha256']['scripts/qualify_claire_session_worker.py'] == source_hash)
            if stall:
                valid = valid and (boundary['injected_callback_stall_ns'] or 0) >= 450_000_000 and receiver['native_video_drop_warning_events'] > 0
        row['diagnostic_completed'] = valid
        cases.append(row)
        print(json.dumps({k: row.get(k) for k in ('name', 'process_exit', 'diagnostic_completed', 'consumer_source_frames', 'ffi_source_frames')}), flush=True)
        if not valid or file_hash(child) != source_hash:
            break
    stable = (file_hash(child) == source_hash and file_hash(Path(__file__)) == runner_hash and
        file_hash(runtime.library) == runtime.library_sha256 and file_hash(args.receiver_runtime) == runtime.revision)
    completed = len(cases) == len(specs) and all(row['diagnostic_completed'] for row in cases) and stable
    report = {'schema_version': 1, 'status': 'completed' if completed else 'failed', 'cases': cases,
        'repeats_per_setting':args.repeats,
        'qualifier_sha256': source_hash, 'runner_sha256': runner_hash,
        'receiver_library_sha256': runtime.library_sha256, 'receiver_descriptor_sha256': runtime.revision,
        'scope': 'Alternating same-host repeats per decoder/queue setting, plus deliberate overflow control. Completion describes diagnostics, not lossless delivery, independent samples, stable rate, latency, browser, physical A/V or capacity.',
        'model_downloads': False, 'quality_qualified': False, 'physical_presentation_qualified': False}
    args.report.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(args.report, 'x', opener=lambda path, flags: os.open(path, flags | os.O_NOFOLLOW, 0o600)) as target:
        target.write(json.dumps(report, indent=2) + '\n')
    return 0 if completed else 1


if __name__ == '__main__':
    def interrupted(signum,frame):raise KeyboardInterrupt('qualification interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,choices=(1,2),default=2)
    for name in ('receiver-runtime', 'config', 'engine', 'scene', 'directory', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
