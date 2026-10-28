"""Keep receiver diagnostics scoped to counters, not track identities."""
import importlib.util
import logging
import json
from pathlib import Path
from types import SimpleNamespace

from livekit.rtc._proto import stats_pb2, ffi_pb2
import pytest


def test_video_rtp_stats_exclude_other_streams_and_identifiers():
    script = Path(__file__).resolve().parents[3] / 'scripts/qualify_claire_session_worker.py'
    spec = importlib.util.spec_from_file_location('claire_session_qualification', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    video = stats_pb2.RtcStats()
    video.inbound_rtp.stream.kind = 'video'
    video.inbound_rtp.stream.transport_id = 'private-transport'
    video.inbound_rtp.inbound.track_identifier = 'private-track'
    video.inbound_rtp.inbound.frames_received = 10
    video.inbound_rtp.inbound.frames_decoded = 9
    video.inbound_rtp.inbound.frames_dropped = 1
    video.inbound_rtp.received.packets_lost = -1
    video.inbound_rtp.inbound.decoder_implementation = 'NVIDIA H264 Decoder'
    audio = stats_pb2.RtcStats()
    audio.inbound_rtp.stream.kind = 'audio'
    other = stats_pb2.RtcStats()
    other.codec.codec.mime_type = 'video/H264'
    rows = module.receiver_video_stats([audio, other, video])
    assert len(rows) == 1
    assert rows[0]['frames_received'] == 10
    assert rows[0]['frames_decoded'] == 9
    assert rows[0]['frames_dropped'] == 1
    assert rows[0]['packets_lost'] == -1
    assert rows[0]['decoder_family'] == 'nvidia_h264'
    assert 'private' not in str(rows)
    assert module.receiver_video_stats([audio, other]) == []
    video.inbound_rtp.inbound.decoder_implementation = 'private-path-or-identifier'
    assert module.receiver_video_stats([video])[0]['decoder_family'] == 'other_or_unreported'


def test_source_capture_trace_validates_order_and_completeness(tmp_path):
    script = Path(__file__).resolve().parents[3] / 'scripts/qualify_claire_session_worker.py'
    spec = importlib.util.spec_from_file_location('claire_capture_qualification', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lines = ['vendor output', 'THA_CAPTURE_TRACE 8 0 100 105 110 500 1',
             'THA_CAPTURE_TRACE 9 1600 200 205 211 250 0']
    rows = module.source_capture_trace(lines, 2)
    assert [row['source_sample'] for row in rows] == [0, 1600]
    assert rows[0]['keyframe'] and not rows[1]['keyframe']
    assert rows[1]['capture_end_us'] - rows[1]['capture_start_us'] == 6
    with pytest.raises(ValueError, match='incomplete'):
        module.source_capture_trace(lines, 3)
    with pytest.raises(ValueError, match='unordered'):
        module.source_capture_trace([lines[2], lines[1]], 2)
    with pytest.raises(ValueError, match='bounds'):
        module.source_capture_trace(['THA_CAPTURE_TRACE 1 1 0 0 0 2 0'], 1)
    path = tmp_path / 'stderr.log'
    path.write_text('\n'.join(lines))
    assert module.read_source_capture_trace(path, 2) == rows
    path.write_bytes(b'x' * (4 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match='exceeded bound'):
        module.read_source_capture_trace(path, 2)
    assert module.sender_rtp_stats(['THA_SENDER_RTP_UNAVAILABLE']) is None
    counters = module.sender_rtp_stats(['THA_SENDER_RTP 100 98 99 500 1 2 3 4 5'])
    assert counters['video_frames_accepted_total'] == 100
    assert counters['frames_sent'] == 98 and counters['nack_count'] == 2
    settled = module.sender_rtp_stats(['THA_SENDER_RTP_AFTER_300MS 100 100 100 510 1 2 3 4 5'], 'THA_SENDER_RTP_AFTER_300MS')
    assert settled['frames_sent'] == 100
    assert module.sender_rtp_stats(['THA_SENDER_RTP_AFTER_300MS 100 100 100 510 1 2 3 4 5']) is None
    with pytest.raises(ValueError, match='invalid sender RTP'):
        module.sender_rtp_stats(['THA_SENDER_RTP 1 2'])


def test_native_video_queue_request_and_warning_redaction(monkeypatch):
    script = Path(__file__).resolve().parents[3] / 'scripts/qualify_claire_session_worker.py'
    spec = importlib.util.spec_from_file_location('claire_queue_qualification', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    requests = []
    def request(value):
        requests.append(value)
        return SimpleNamespace(new_video_stream=SimpleNamespace(stream='stream'))
    monkeypatch.setattr(module, 'FfiClient', SimpleNamespace(instance=SimpleNamespace(request=request)))
    stream = SimpleNamespace(_track=SimpleNamespace(_ffi_handle=SimpleNamespace(handle=42)),
                             _format=None, _native_queue_frames=8)
    assert module.QualifiedVideoStream._create_owned_stream(stream) == 'stream'
    config = requests[0].new_video_stream
    assert config.track_handle == 42 and config.queue_size_frames == 8 and config.normalize_stride
    assert config.HasField('queue_size_frames')
    warnings = module.NativeVideoDropWarnings()
    warnings.handle(logging.makeLogRecord({'msg': 'native video stream queue overflow; dropped 100 queued frames'}))
    warnings.handle(logging.makeLogRecord({'msg': 'private transcript must not be retained'}))
    assert warnings.snapshot() == (1, 100)


def qualification_module():
    script = Path(__file__).resolve().parents[3] / 'scripts/qualify_claire_session_worker.py'
    spec = importlib.util.spec_from_file_location('claire_ffi_qualification', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def video_event(sample=0, *, handle=42, session='fixture-session', **extra):
    event = ffi_pb2.FfiEvent()
    video = event.video_stream_event
    video.stream_handle = handle
    frame = video.frame_received
    frame.metadata.frame_id = sample // 1600 + 3
    frame.metadata.user_data = json.dumps(dict(session_id=session, turn_id='test', epoch=7,
                                               source_sample=sample, **extra)).encode()
    return event


def test_ffi_trace_preserves_forwarding_and_redacts_generation(monkeypatch):
    module = qualification_module()
    received, sleeps = [], []
    queue = SimpleNamespace(put=received.append)
    original = queue.put
    monkeypatch.setattr(module.time, 'sleep', sleeps.append)
    probe = module.FfiVideoBoundaryTrace(queue, 500)
    try:
        probe.register(42, 'fixture-session')
        foreign = video_event(handle=99, session='private-foreign')
        queue.put(foreign)
        queue.put(video_event(idle_before_audio=True))
        queue.put(video_event(0))
        queue.put(video_event(1600))
        queue.put(video_event(1600, idle_after_eof=True))
        observed = probe.snapshot(42)
        assert observed['source_frames'] == [{'source_sample': 0, 'frame_id': 3}, {'source_sample': 1600, 'frame_id': 4}]
        assert observed['preroll_frames'] == observed['idle_frames'] == 1
        assert sleeps == [.5] and len(received) == 5
        assert 'private' not in str(observed) and 'fixture-session' not in str(observed)
    finally:
        probe.close()
    assert queue.put == original


@pytest.mark.parametrize('fault', ['foreign_generation', 'duplicate', 'oversized_metadata', 'row_bound'])
def test_invalid_ffi_observation_is_forwarded_but_cannot_pass(fault):
    module = qualification_module()
    received = []
    queue = SimpleNamespace(put=received.append)
    probe = module.FfiVideoBoundaryTrace(queue)
    probe.register(42, 'fixture-session')
    queue.put(video_event())
    event = video_event(1600)
    if fault == 'foreign_generation':
        event = video_event(1600, session='private-other-session')
    elif fault == 'duplicate':
        event = video_event()
    elif fault == 'oversized_metadata':
        event.video_stream_event.frame_received.metadata.user_data = b'x' * 2049
    else:
        for index in range(1, 3601):
            queue.put(video_event(index * 1600))
    queue.put(event)
    with pytest.raises(ValueError, match='private metadata excluded'):
        probe.snapshot(42)
    assert received[-1] is event
    assert len(probe.streams[42]['rows']) <= 3600
    probe.close()


def test_source_matched_audio_receiver_requests_finite_native_queue(monkeypatch):
    module = qualification_module()
    requests = []
    def request(value):
        requests.append(value)
        return SimpleNamespace(new_audio_stream=SimpleNamespace(stream='stream'))
    monkeypatch.setattr(module, 'FfiClient', SimpleNamespace(instance=SimpleNamespace(request=request)))
    stream = SimpleNamespace(_track=SimpleNamespace(_ffi_handle=SimpleNamespace(handle=42)),
        _sample_rate=48000, _num_channels=1, _frame_size_ms=10, _audio_filter_module=None, _processor=None)
    assert module.QualifiedAudioStream._create_owned_stream(stream) == 'stream'
    config = requests[0].new_audio_stream
    assert config.queue_size_frames == 25 and config.sample_rate == 48000 and config.frame_size_ms == 10
