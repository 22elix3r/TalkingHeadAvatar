"""Actual local SFU/GPU worker protocol/fault checks; synthetic PCM only."""
import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import signal
import struct
import time

from talkinghead_v2.assets.avatar import load_avatar
from talkinghead_v2.local import digest
from talkinghead_v2.providers.tts.native import load_native_config
from talkinghead_v2.rendering.projected import project_neutral
from talkinghead_v2.server.config import Settings
from talkinghead_v2.server.native_media import NativeMediaConfig, private_write
from talkinghead_v2.server.transport import grant
from talkinghead_v2.session.device import DeviceBusy, DeviceLease

ROOT = Path(__file__).resolve().parents[1]


async def main(args):
    if args.directory.exists() or args.report.exists(): raise ValueError('refusing to overwrite evidence')
    args.directory.mkdir(parents=True, mode=0o700)
    settings = Settings.load(args.config)
    config = NativeMediaConfig.load(settings.native_media_runtime)
    lease = DeviceLease(load_native_config(settings.tts_runtime).device_lease_path)
    lease.acquire('native-worker-protocol-qualification')
    cases = []
    try:
        for name, count, fault in [('empty', 0, None), ('one', 1, None), ('short', 479, None),
                ('aligned', 480, None), ('partial', 481, None), ('quantum', 960, None),
                ('tail', 961, None), ('wrong_epoch', 480, 'epoch'), ('gap', 480, 'gap'),
                ('sequence', 480, 'sequence'), ('backpressure_cancel', 4800, 'cancel')]:
            directory = args.directory / name; directory.mkdir(mode=0o700)
            asset = load_avatar(ROOT / 'src/talkinghead_v2/fixtures/p03-avatar/avatar.json', tenant_id='procedural')
            private_write(directory / 'mesh.bin', project_neutral(asset, 512).packed())
            sid = 'native-protocol-' + name
            private_write(directory / 'bootstrap.txt', ('\n'.join(('THA_NATIVE_MEDIA_1', settings.livekit_url,
                grant(settings, 'protocol-' + name, 'media-' + sid, 'media'), sid, 'test', '7', '300')) + '\n').encode())
            log = open(directory / 'diagnostics.log', 'xb', opener=lambda path, flags: os.open(path, flags, 0o600))
            process = await asyncio.create_subprocess_exec(str(config.executable), str(directory / 'bootstrap.txt'),
                str(config.ptx), str(directory / 'mesh.bin'), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=log, pass_fds=(lease.fileno,), limit=16384)
            log.close()
            producer = None
            try:
                async def event():
                    line = await asyncio.wait_for(process.stdout.readline(), 20)
                    if not line: raise ValueError('worker exited without protocol event')
                    return json.loads(line)
                ready = await event()
                if ready['kind'] != 'ready' or ready['epoch'] != 7: raise ValueError('worker not ready')
                samples = tuple(int(math.sin(index * 2 * math.pi * 440 / 48000) * 1600) for index in range(count))
                packet = struct.pack('<cQQQI', b'D' if fault == 'cancel' else b'F', 6 if fault == 'epoch' else 7,
                    1 if fault == 'sequence' else 0, 1 if fault == 'gap' else 0, count) + struct.pack(f'<{count}h', *samples)
                if fault == 'cancel':
                    async def flood():
                        for index in range(1200):
                            process.stdin.write(struct.pack('<cQQQI', b'D', 7, index, index * 4800, 4800) + packet[29:])
                            await process.stdin.drain()
                    process.stdin.transport.set_write_buffer_limits(high=8192, low=4096)
                    producer = asyncio.create_task(flood())
                    await asyncio.sleep(.15)
                    if producer.done(): raise ValueError('cancel case did not reach input backpressure')
                    began = time.monotonic(); process.send_signal(signal.SIGTERM)
                    await asyncio.wait_for(process.wait(), 5)
                    elapsed = (time.monotonic() - began) * 1000
                    if process.returncode != 0: raise ValueError('cooperative native cancel failed')
                    cases.append({'name': name, 'blocked_producer': True, 'owned_process_exit_ms': elapsed,
                                  'receiver_flush_claimed': False})
                else:
                    process.stdin.write(packet); await process.stdin.drain()
                    result = await event()
                    if fault:
                        if result != {'kind': 'error', 'code': 'native_media_failed'}: raise ValueError('malformed input accepted')
                        await asyncio.wait_for(process.wait(), 5)
                        if process.returncode != 1: raise ValueError('fault did not retire worker')
                        cases.append({'name': name, 'rejected': True, 'process_exit': 1})
                    else:
                        if (result.get('kind') != 'eof' or result.get('valid_samples') != count or
                            result.get('padding_samples') != (-count) % 480 + 960 or result.get('presentation_ack') is not False):
                            raise ValueError('worker lost samples or misreported EOF')
                        cases.append({'name': name, 'sender': {key: value for key, value in result.items()
                            if key not in {'session_id', 'turn_id'}}, 'process_exit': 0})
                        process.send_signal(signal.SIGTERM)
                        await asyncio.wait_for(process.wait(), 5)
                        if process.returncode != 0: raise ValueError('post-EOF retirement failed')
            finally:
                if process.returncode is None: process.kill(); await process.wait()
                if producer: producer.cancel(); await asyncio.gather(producer, return_exceptions=True)
                process.stdin.close()
                try: await process.stdin.wait_closed()
                except (BrokenPipeError, ConnectionResetError): pass
                # Remove tokens/geometry; retain only owned diagnostic stderr privately.
                (directory / 'bootstrap.txt').unlink(); (directory / 'mesh.bin').unlink()
    finally: lease.release()
    competitor = DeviceLease(lease.path); competitor.acquire('post-native-worker-retirement'); competitor.release()
    sources = [ROOT / 'native/media/session_worker.cpp', ROOT / 'native/media/projected_file.h',
        ROOT / 'src/talkinghead_v2/server/native_media.py', Path(__file__)]
    report = {'schema_version': 1, 'phase': 'P06', 'phase_status': 'in_progress',
        'scope': 'actual CUDA/NVENC/LiveKit worker sender protocol with synthetic sine PCM; no receiver/quality inference',
        'source_sha256': {path.relative_to(ROOT).as_posix(): digest(path) for path in sources},
        'executable_sha256': digest(config.executable), 'ptx_sha256': digest(config.ptx), 'cases': cases,
        'post_retirement_admission_reacquired': True, 'model_downloads': False,
        'acceptance': {'receiver_presentation': False, 'physical_av': False, 'real_avatar': False, 'release': False}}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'worker_cases_passed': len(cases), 'release': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    asyncio.run(main(parser.parse_args()))
