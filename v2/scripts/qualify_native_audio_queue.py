"""Exercise an explicit patched FFI with synthetic audio and a real local SFU."""
import argparse
import asyncio
import hashlib
import logging
import math
import os
from pathlib import Path
import re
import struct
import threading
import time
import uuid

from livekit import rtc
from livekit.rtc._ffi_client import FfiClient
from livekit.rtc._proto import audio_frame_pb2, ffi_pb2
from livekit.rtc.log import logger as livekit_logger

from talkinghead_v2.server.config import Settings
from talkinghead_v2.server.transport import CheckedReceiveQueue, grant


class BoundedNativeAudioStream(rtc.AudioStream):
    """Qualification only: the caller must verify the fail-closed FFI first."""
    def __init__(self, track, native_frames):
        if type(native_frames) is not int or not 1 <= native_frames <= 25:
            raise ValueError('bounded native audio queue required')
        self._native_frames = native_frames
        super().__init__(track, sample_rate=16000, num_channels=1, frame_size_ms=20)
        if type(self._queue).__name__ != 'RingQueue':
            raise ValueError('unqualified Python audio queue')
        self._queue = CheckedReceiveQueue(12)

    def _create_owned_stream(self):
        if self._track is None or self._audio_filter_module is not None or self._processor is not None:
            raise ValueError('unfiltered explicit audio track required')
        request = ffi_pb2.FfiRequest()
        stream = request.new_audio_stream
        stream.track_handle = self._track._ffi_handle.handle
        stream.sample_rate = self._sample_rate
        stream.num_channels = self._num_channels
        stream.frame_size_ms = self._frame_size_ms
        stream.queue_size_frames = self._native_frames
        stream.type = audio_frame_pb2.AudioStreamType.AUDIO_STREAM_NATIVE
        return FfiClient.instance.request(request).new_audio_stream.stream


class OverflowWarnings(logging.Handler):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.evictions = 0

    def emit(self, record):
        if re.search(r'native audio stream queue overflow; stream closed after \d+ overflow events', record.getMessage()):
            self.count += 1
        if re.search(r'native audio stream queue overflow; dropped \d+ queued frames', record.getMessage()):
            self.evictions += 1


async def run(args):
    library = args.ffi.resolve(strict=True)
    if (not library.is_file() or library.stat().st_uid != os.getuid() or library.stat().st_mode & 0o022 or
        hashlib.sha256(library.read_bytes()).hexdigest() != args.ffi_sha256 or
        Path(os.environ.get('LIVEKIT_LIB_PATH', '')).resolve() != library):
        raise ValueError('explicit verified project FFI and matching process environment required')
    if args.output.exists():
        raise ValueError('preserve previous qualification output')
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    settings = Settings.load(args.config)
    publisher, observer = rtc.Room(), rtc.Room()
    room_name = 'audio-queue-' + uuid.uuid4().hex
    streams, tasks = [], []
    ready, ended = asyncio.Event(), asyncio.Event()
    sample_count = frame_count = 0
    maximum_rms = 0.0
    target_handle = None
    blocked = threading.Event()
    blocked_ns = None
    native_overflow_eos = False
    source = None
    producer = None
    ffi_queue = FfiClient.instance.queue
    original_put = ffi_queue.put
    warnings = OverflowWarnings()
    livekit_logger.addHandler(warnings)

    def controlled_put(event):
        nonlocal blocked_ns, native_overflow_eos
        if (event.WhichOneof('message') == 'audio_stream_event' and
            event.audio_stream_event.stream_handle == target_handle and event.audio_stream_event.HasField('eos')):
            # The pinned patch adds optional bool field 1 to AudioStreamEOS.
            # The unchanged Python SDK retains this extension as protobuf unknown bytes.
            native_overflow_eos = event.audio_stream_event.eos.SerializeToString() == b'\x08\x01'
        if (args.mode != 'normal' and not blocked.is_set() and
            event.WhichOneof('message') == 'audio_stream_event' and
            event.audio_stream_event.stream_handle == target_handle and
            event.audio_stream_event.HasField('frame_received')):
            blocked.set()
            began = time.monotonic_ns()
            time.sleep(0.5)  # Deliberately stall this one FFI callback in the disposable probe.
            blocked_ns = time.monotonic_ns() - began
        original_put(event)

    ffi_queue.put = controlled_put

    @observer.on('track_subscribed')
    def subscribed(track, publication, participant):
        nonlocal target_handle
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        stream = BoundedNativeAudioStream(track, args.native_frames)
        streams.append(stream)
        target_handle = stream._ffi_handle.handle

        async def consume():
            nonlocal sample_count, frame_count, maximum_rms
            async for event in stream:
                values = event.frame.data
                frame_count += 1
                sample_count += len(values)
                maximum_rms = max(maximum_rms, math.sqrt(sum(float(x) ** 2 for x in values) / len(values)))
            ended.set()
        tasks.append(asyncio.create_task(consume()))
        ready.set()

    try:
        await observer.connect(settings.livekit_url, grant(settings, room_name, 'observer', 'controller'))
        await publisher.connect(settings.livekit_url, grant(settings, room_name, 'source', 'media'))
        source = rtc.AudioSource(48000, 1, queue_size_ms=1000)
        track = rtc.LocalAudioTrack.create_audio_track('synthetic-queue-test', source)
        await publisher.local_participant.publish_track(track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        await asyncio.wait_for(ready.wait(), 10)

        async def publish_audio():
            for index in range(200):
                values = [int(5000 * math.sin(2 * math.pi * 440 * (index * 480 + offset) / 48000)) for offset in range(480)]
                await source.capture_frame(rtc.AudioFrame(struct.pack('<480h', *values), 48000, 1, 480))
            await source.wait_for_playout()
        producer = asyncio.create_task(publish_audio())
        if args.mode == 'overflow':
            await asyncio.wait_for(ended.wait(), 8)
            before = frame_count
            await asyncio.sleep(0.2)
            if not blocked.is_set() or not native_overflow_eos or frame_count != before:
                raise ValueError('native overflow evidence incomplete: ' + str({
                    'callback_blocked': blocked.is_set(), 'overflow_warnings': warnings.count,
                    'native_overflow_eos': native_overflow_eos,
                    'frames_at_eos': before, 'frames_after_eos_wait': frame_count,
                    'producer_done': producer.done(), 'received_rms': round(maximum_rms, 3),
                    'receiver_connected': observer.connection_state == rtc.ConnectionState.CONN_CONNECTED}))
        else:
            await asyncio.wait_for(producer, 8)
            await asyncio.sleep(0.2)
            if ended.is_set() or frame_count < 75 or maximum_rms < 100 or warnings.count:
                raise ValueError('healthy bounded audio stream failed')
            if args.mode == 'eviction_control' and (not blocked.is_set() or warnings.evictions == 0):
                raise ValueError('unpatched bounded queue did not demonstrate eviction')
        for task in tasks:
            if task.done():
                task.result()
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != source_sha256:
            raise ValueError('qualification source changed during the run')
        report = {'schema_version': 1, 'mode': args.mode, 'real_local_sfu': True, 'synthetic_audio': True,
            'source_sha256': source_sha256,
            'ffi_sha256': args.ffi_sha256, 'native_queue_frames': args.native_frames, 'nominal_native_queue_ms': args.native_frames * 10,
            'python_queue_frames': 12, 'python_queue_peak_frames': streams[0]._queue.peak,
            'received_frames_20ms': frame_count, 'received_samples_16k': sample_count,
            'maximum_rms': maximum_rms, 'stream_ended_before_cleanup': ended.is_set(),
            'overflow_warnings_observed': warnings.count, 'injected_callback_stall_ns': blocked_ns,
            'eviction_warnings_observed': warnings.evictions,
            'native_queue_overflow_eos': native_overflow_eos,
            'room_connected_at_result': observer.connection_state == rtc.ConnectionState.CONN_CONNECTED,
            'physical_presentation_qualified': False, 'model_downloads': False,
            'scope': 'FFI/receiver queue behavior only; no ASR, conversation, browser or lossless PCM claim'}
        args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with open(args.output, 'x', opener=lambda path, flags: os.open(path, flags | os.O_NOFOLLOW, 0o600)) as target:
            import json
            target.write(json.dumps(report, indent=2) + '\n')
        print({'mode': args.mode, 'received_frames': frame_count, 'ended_before_cleanup': ended.is_set(), 'warnings': warnings.count})
    finally:
        ffi_queue.put = original_put
        if producer and not producer.done():
            producer.cancel()
        if producer:
            await asyncio.gather(producer, return_exceptions=True)
        if source:
            try:
                source.clear_queue()
            except Exception:
                pass
        try:
            await asyncio.wait_for(asyncio.gather(*(stream.aclose() for stream in streams), return_exceptions=True), 5)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.gather(observer.disconnect(), publisher.disconnect(), return_exceptions=True)
            livekit_logger.removeHandler(warnings)
            # Retire the process-owned native runtime while Python callbacks and
            # the event loop still exist, rather than deferring it to atexit.
            await asyncio.to_thread(FfiClient.instance._dispose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('normal', 'overflow', 'eviction_control'), required=True)
    parser.add_argument('--native-frames', type=int, choices=(6, 25), default=25)
    parser.add_argument('--ffi', type=Path, required=True)
    parser.add_argument('--ffi-sha256', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
