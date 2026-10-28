"""Bounded audio input for an explicitly pinned fail-closed FFI runtime."""
from importlib.metadata import version
import os
from pathlib import Path

from livekit import rtc
from livekit.rtc._ffi_client import FfiClient
from livekit.rtc._proto import audio_frame_pb2, ffi_pb2

from talkinghead_v2.contracts import ContractError
from talkinghead_v2.server.inbound_config import PYTHON_LIVEKIT_VERSION


def validate_loaded_runtime(config):
    if (version('livekit') != PYTHON_LIVEKIT_VERSION or
        Path(os.environ.get('LIVEKIT_LIB_PATH', '')).resolve() != config.library):
        raise ContractError('qualified inbound FFI must be selected before SDK initialization')
    loaded = Path(FfiClient.instance._ffi_lib._name).resolve()
    if loaded != config.library:
        raise ContractError('another inbound FFI is already loaded')


def dispose_owned_runtime(config):
    instance = FfiClient._instance
    if instance is not None:
        if Path(instance._ffi_lib._name).resolve() != config.library:
            raise ContractError('refusing to dispose another inbound FFI')
        instance._dispose()


class BoundedNativeAudioStream(rtc.AudioStream):
    def __init__(self, track, config):
        validate_loaded_runtime(config)
        self._native_frames = config.native_frames
        super().__init__(track, sample_rate=16000, num_channels=1, frame_size_ms=20)

    def _create_owned_stream(self):
        if self._track is None or self._audio_filter_module is not None or self._processor is not None:
            raise ContractError('qualified unfiltered microphone track required')
        request = ffi_pb2.FfiRequest()
        stream = request.new_audio_stream
        stream.track_handle = self._track._ffi_handle.handle
        stream.sample_rate = self._sample_rate
        stream.num_channels = self._num_channels
        stream.frame_size_ms = self._frame_size_ms
        stream.queue_size_frames = self._native_frames
        stream.type = audio_frame_pb2.AudioStreamType.AUDIO_STREAM_NATIVE
        return FfiClient.instance.request(request).new_audio_stream.stream
