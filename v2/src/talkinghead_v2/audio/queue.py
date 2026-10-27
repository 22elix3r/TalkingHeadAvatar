"""Single-event-loop queue: bounded items AND exact media duration, no eviction."""

import asyncio
from collections import deque
from fractions import Fraction

from talkinghead_v2.contracts import AudioChunk, AudioEvent, ContractError, GenerationKey, integer


class EpochCancelled(Exception):
    """A blocked producer/consumer belongs to an invalidated generation."""


class MediaQueue:
    def __init__(self, sample_rate: int, max_duration: Fraction, max_items: int):
        integer(sample_rate, "sample_rate", 1)
        integer(max_items, "max_items", 1)
        if not isinstance(max_duration, Fraction) or max_duration <= 0:
            raise ContractError("max_duration must be a positive Fraction")
        self.sample_rate = sample_rate
        self.max_samples = (max_duration * sample_rate).__floor__()
        if self.max_samples < 1:
            raise ContractError("queue duration is shorter than one sample")
        self.max_items = max_items
        self.key: GenerationKey | None = None
        self._session_id: str | None = None
        self._highest_epoch = -1
        self._items: deque[AudioEvent] = deque()
        self.samples = 0
        self.peak_samples = 0
        self.peak_items = 0
        self.blocked_puts = 0
        self.put_blocked = asyncio.Event()
        self._changed = asyncio.Event()

    @property
    def item_count(self) -> int:
        return len(self._items)

    def _notify(self) -> None:
        self._changed.set()
        self._changed = asyncio.Event()

    def reset(self, key: GenerationKey | None) -> tuple[int, int]:
        if key is not None:
            if (self._session_id is not None and key.session_id != self._session_id) or key.epoch <= self._highest_epoch:
                raise ContractError("queue activation requires the same session and a strictly newer epoch")
            self._session_id = key.session_id
            self._highest_epoch = key.epoch
        removed = self.samples, len(self._items)
        self._items.clear()
        self.samples = 0
        self.key = key
        self._notify()
        return removed

    async def put(self, event: AudioEvent) -> bool:
        if event.envelope.key != self.key:
            return False
        if event.sample_rate != self.sample_rate:
            raise ContractError("queue sample-rate mismatch")
        count = event.sample_count if isinstance(event, AudioChunk) else 0
        if count > self.max_samples:
            raise ContractError("chunk exceeds queue duration; split before enqueueing")
        while event.envelope.key == self.key:
            changed = self._changed
            if len(self._items) < self.max_items and self.samples + count <= self.max_samples:
                self._items.append(event)
                self.samples += count
                self.peak_samples = max(self.peak_samples, self.samples)
                self.peak_items = max(self.peak_items, len(self._items))
                self._notify()
                return True
            self.blocked_puts += 1
            self.put_blocked.set()
            await changed.wait()
        return False

    def get_nowait(self, key: GenerationKey) -> AudioEvent:
        if key != self.key:
            raise EpochCancelled("consumer generation was cancelled")
        if not self._items:
            raise asyncio.QueueEmpty
        event = self._items.popleft()
        if isinstance(event, AudioChunk):
            self.samples -= event.sample_count
        self._notify()
        return event

    async def get(self, key: GenerationKey) -> AudioEvent:
        while True:
            changed = self._changed
            try:
                return self.get_nowait(key)
            except asyncio.QueueEmpty:
                await changed.wait()
