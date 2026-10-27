"""Keep receiver diagnostics scoped to counters, not track identities."""
import importlib.util
from pathlib import Path

from livekit.rtc._proto import stats_pb2


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
