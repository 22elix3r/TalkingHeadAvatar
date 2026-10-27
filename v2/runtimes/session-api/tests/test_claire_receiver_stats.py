"""Keep receiver diagnostics scoped to counters, not track identities."""
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

from livekit.rtc._proto import stats_pb2
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
    assert 'private' not in str(rows)
    assert module.receiver_video_stats([audio, other]) == []


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
