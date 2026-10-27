"""Boundary between actor-owned media and an independently paced presentation sink."""

from typing import Literal, Protocol

from talkinghead_v2.contracts import AudioEvent, GenerationKey, RigIdentity


class AudioSink(Protocol):
    sample_rate: int
    finished: bool
    cancel_ack_scope: Literal["synthetic_in_process", "local_device_queue_only"]

    def start(self, key: GenerationKey, rig: RigIdentity | None) -> None: ...
    def submit(self, event: AudioEvent) -> bool: ...
    def cancel(self) -> int: ...
