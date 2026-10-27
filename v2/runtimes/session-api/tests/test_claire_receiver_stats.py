"""Keep receiver diagnostics scoped to counters, not track identities."""
import importlib.util
from pathlib import Path

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
