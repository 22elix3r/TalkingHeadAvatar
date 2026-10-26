"""One asyncio owner serializes native events; cancellation never waits on backpressure."""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction

from talkinghead_v2.audio.clock import SampleLedger
from talkinghead_v2.audio.queue import MediaQueue
from talkinghead_v2.audio.resample import RationalResampler
from talkinghead_v2.contracts import (
    AudioChunk, AudioEOF, CancelAck, ContractError, Discontinuity, Envelope,
    EpochStarted, GenerationKey, RigIdentity, integer, label,
)
from talkinghead_v2.session.synthetic import SyntheticSink


@dataclass
class _Generation:
    key: GenerationKey
    ledger: SampleLedger
    resampler: RationalResampler
    output_sequence: int = 0
    output_sample: int = 0
    segments: deque[tuple[int, int, str]] = field(default_factory=deque)


class GenerationSession:
    """CPU reference pipeline. Invoke methods on one event loop, not provider threads.

    Real callbacks must marshal into the owner loop. Only SyntheticSink is qualified
    here; hardware/remote flushing and provider-compute cancellation belong to P02+.
    """

    def __init__(self, session_id: str, sink: SyntheticSink, *,
                 max_queue_duration: Fraction = Fraction(1, 2), max_queue_items: int = 8,
                 output_chunk_samples: int = 480):
        label(session_id, "session_id")
        integer(output_chunk_samples, "output_chunk_samples", 1)
        self.session_id = session_id
        self.sink = sink
        self.queue = MediaQueue(sink.sample_rate, max_queue_duration, max_queue_items)
        if output_chunk_samples > self.queue.max_samples:
            raise ContractError("output chunk exceeds queue duration")
        self.output_chunk_samples = output_chunk_samples
        self.epoch = -1
        self._state: _Generation | None = None
        self._producer_lock = asyncio.Lock()
        self.rejected_stale = 0

    def start(self, turn_id: str, native_rate: int, *, rig: RigIdentity | None = None) -> EpochStarted:
        if self._state is not None and not self.sink.finished:
            raise ContractError("finish scheduling or cancel the active generation before restart")
        key = GenerationKey(self.session_id, turn_id, self.epoch + 1)
        resampler = RationalResampler(native_rate, self.sink.sample_rate)
        self.epoch = key.epoch
        self._state = _Generation(key, SampleLedger(key, native_rate), resampler)
        self.queue.reset(key)
        self.sink.start(key, rig)
        return EpochStarted(key, native_rate, self.sink.sample_rate)

    def _current(self, envelope: Envelope) -> _Generation | None:
        if self._state is None or envelope.key != self._state.key:
            self.rejected_stale += 1
            return None
        return self._state

    async def _enqueue(self, state: _Generation, samples: tuple[float, ...], trace_id: str) -> bool:
        offset = 0
        while offset < len(samples):
            if self._state is not state:
                return False
            # Labels follow each output sample's source position, not the callback
            # that happened to release the filter's right context.
            native_rate = state.ledger.sample_rate
            while state.segments:
                _, end, segment_id = state.segments[0]
                segment_end = (end*self.sink.sample_rate + native_rate-1) // native_rate
                if state.output_sample < segment_end:
                    break
                state.segments.popleft()
            else:
                raise ContractError("resampled output has no source segment span")
            count = min(len(samples)-offset, self.output_chunk_samples, segment_end-state.output_sample)
            chunk = AudioChunk(Envelope(state.key, state.output_sequence, trace_id), segment_id,
                               self.sink.sample_rate, state.output_sample, samples[offset:offset+count])
            if not await self.queue.put(chunk):
                return False
            state.output_sample = chunk.end_sample
            state.output_sequence += 1
            offset += count
        return self._state is state

    async def push(self, chunk: AudioChunk) -> bool:
        async with self._producer_lock:
            state = self._current(chunk.envelope)
            if state is None:
                return False
            # Bound transient resampling work too. Larger callbacks must be explicitly split.
            if chunk.sample_count > state.ledger.sample_rate:
                raise ContractError("native callback exceeds one-second reference work bound")
            state.ledger.accept(chunk)
            try:
                if chunk.sample_count:
                    if state.segments and state.segments[-1][2] == chunk.segment_id:
                        start, _, segment_id = state.segments.pop()
                        state.segments.append((start, chunk.end_sample, segment_id))
                    else:
                        state.segments.append((chunk.start_sample, chunk.end_sample, chunk.segment_id))
                output = state.resampler.push(chunk.samples)
                return await self._enqueue(state, output, chunk.envelope.trace_id)
            except asyncio.CancelledError:
                if self._state is state:
                    self.cancel("producer task cancelled during audio enqueue")
                raise
            except Exception:
                if self._state is state:
                    self.cancel("audio processing failed after native samples were accepted")
                raise

    async def finish(self, eof: AudioEOF) -> bool:
        async with self._producer_lock:
            state = self._current(eof.envelope)
            if state is None:
                return False
            # This reference takes complete native samples. Fractional native validity
            # needs an explicit provider adapter rather than silently playing padding.
            if eof.valid_end_sample != eof.end_sample:
                raise ContractError("fractional native padding is not supported by this adapter")
            state.ledger.accept(eof)
            try:
                tail = state.resampler.finish()
                if not await self._enqueue(state, tail, eof.envelope.trace_id):
                    return False
                output_eof = AudioEOF(Envelope(state.key, state.output_sequence, eof.envelope.trace_id),
                                      self.sink.sample_rate, state.output_sample,
                                      state.resampler.valid_end_sample)
                return await self.queue.put(output_eof)
            except asyncio.CancelledError:
                if self._state is state:
                    self.cancel("producer task cancelled during EOF enqueue")
                raise
            except Exception:
                if self._state is state:
                    self.cancel("EOF processing failed after native EOF was accepted")
                raise

    def cancel(self, reason: str) -> CancelAck:
        label(reason, "reason")
        state = self._state
        if state is None:
            raise ContractError("no active generation to cancel")
        integer(self.epoch + 1, "fence_epoch")
        self.epoch += 1
        state.resampler.cancel()
        discarded_samples, discarded_items = self.queue.reset(None)
        last = self.sink.cancel()
        self._state = None
        return CancelAck(state.key, self.epoch, reason, discarded_samples, discarded_items, last, self.sink.sample_rate)

    def discontinuity(self, event: Discontinuity) -> CancelAck | None:
        state = self._current(event.envelope)
        if state is None:
            return None
        state.ledger.accept(event)
        # No interpolation, filter flushing or zero-filled speech across an unknown gap.
        return self.cancel(f"discontinuity: {event.reason}")
