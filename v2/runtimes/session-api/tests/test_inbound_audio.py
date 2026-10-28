import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import talkinghead_v2.server.actor as actor_module
from talkinghead_v2.server.actor import Actor
from talkinghead_v2.server.inbound_audio import BoundedNativeAudioStream
from talkinghead_v2.contracts import ContractError


class EndedStream:
    def __init__(self): self.aclose = AsyncMock()
    def __aiter__(self): return self
    async def __anext__(self): raise StopAsyncIteration


async def test_unexpected_native_eof_fails_the_session(monkeypatch):
    stream = EndedStream()
    factory = Mock(return_value=stream)
    monkeypatch.setattr(actor_module, 'checked_audio_stream', factory)
    actor = Actor.__new__(Actor)
    actor.closed = actor.closing = False
    actor.inbound_audio_config = object()
    actor.fail = AsyncMock()
    track = object()
    await actor.receive(track)
    actor.fail.assert_awaited_once_with('microphone_discontinuity')
    stream.aclose.assert_awaited_once()
    factory.assert_called_once_with(track, actor.inbound_audio_config)


async def test_receiver_setup_failure_is_fenced(monkeypatch):
    monkeypatch.setattr(actor_module, 'checked_audio_stream', Mock(side_effect=RuntimeError('fixture')))
    actor = Actor.__new__(Actor)
    actor.closed = actor.closing = False
    actor.fail = AsyncMock()
    await actor.receive(object())
    actor.fail.assert_awaited_once_with('microphone_discontinuity')


async def test_expected_receiver_cancellation_does_not_fail(monkeypatch):
    class CancelledStream(EndedStream):
        async def __anext__(self): raise asyncio.CancelledError
    stream = CancelledStream()
    monkeypatch.setattr(actor_module, 'checked_audio_stream', Mock(return_value=stream))
    actor = Actor.__new__(Actor)
    actor.closed = actor.closing = False
    actor.fail = AsyncMock()
    await actor.receive(object())
    actor.fail.assert_not_awaited()
    stream.aclose.assert_awaited_once()


def test_native_audio_request_has_a_finite_queue(monkeypatch):
    import talkinghead_v2.server.inbound_audio as module
    requests = []
    def request(value):
        requests.append(value)
        return SimpleNamespace(new_audio_stream=SimpleNamespace(stream='stream'))
    monkeypatch.setattr(module, 'FfiClient', SimpleNamespace(instance=SimpleNamespace(request=request)))
    stream = SimpleNamespace(_track=SimpleNamespace(_ffi_handle=SimpleNamespace(handle=42)),
        _sample_rate=16000, _num_channels=1, _frame_size_ms=20, _native_frames=25,
        _audio_filter_module=None, _processor=None)
    assert BoundedNativeAudioStream._create_owned_stream(stream) == 'stream'
    config = requests[0].new_audio_stream
    assert config.queue_size_frames == 25 and config.sample_rate == 16000 and config.frame_size_ms == 20


async def test_only_active_users_unpublished_microphone_fails_the_session():
    actor = Actor.__new__(Actor)
    actor.closed = actor.closing = False
    actor.row = SimpleNamespace(id='session')
    actor.input_loss_task = None
    actor.fail = AsyncMock()
    microphone = SimpleNamespace(source=actor_module.rtc.TrackSource.SOURCE_MICROPHONE)
    camera = SimpleNamespace(source=actor_module.rtc.TrackSource.SOURCE_CAMERA)
    actor.microphone_unpublished(microphone, SimpleNamespace(identity='another-user'))
    actor.microphone_unpublished(camera, SimpleNamespace(identity='user-session'))
    actor.fail.assert_not_awaited()
    actor.microphone_unpublished(microphone, SimpleNamespace(identity='user-session'))
    await actor.input_loss_task
    actor.fail.assert_awaited_once_with('microphone_discontinuity')
    actor.closing = True
    actor.microphone_unpublished(microphone, SimpleNamespace(identity='user-session'))
    actor.fail.assert_awaited_once()


def test_unselected_or_already_loaded_other_ffi_is_rejected(monkeypatch, tmp_path):
    import talkinghead_v2.server.inbound_audio as module
    config = SimpleNamespace(library=tmp_path / 'qualified.so')
    monkeypatch.setattr(module, 'version', lambda name: '1.1.21')
    monkeypatch.delenv('LIVEKIT_LIB_PATH', raising=False)
    client = SimpleNamespace(instance=SimpleNamespace(_ffi_lib=SimpleNamespace(_name=str(tmp_path / 'other.so'))))
    monkeypatch.setattr(module, 'FfiClient', client)
    with pytest.raises(ContractError, match='selected'):
        module.validate_loaded_runtime(config)
    monkeypatch.setenv('LIVEKIT_LIB_PATH', str(config.library))
    with pytest.raises(ContractError, match='already loaded'):
        module.validate_loaded_runtime(config)


def test_owned_disposal_rejects_another_runtime(monkeypatch, tmp_path):
    import talkinghead_v2.server.inbound_audio as module
    dispose = Mock()
    instance = SimpleNamespace(_ffi_lib=SimpleNamespace(_name=str(tmp_path / 'other.so')), _dispose=dispose)
    monkeypatch.setattr(module, 'FfiClient', SimpleNamespace(_instance=instance))
    with pytest.raises(ContractError, match='refusing'):
        module.dispose_owned_runtime(SimpleNamespace(library=tmp_path / 'qualified.so'))
    dispose.assert_not_called()
