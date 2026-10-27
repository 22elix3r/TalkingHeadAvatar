"""Bounded Claire audio context frontend. This module does not run the neural model."""

from collections import deque
from dataclasses import dataclass
from fractions import Fraction

from talkinghead_v2.audio.clock import SampleLedger
from talkinghead_v2.contracts import AudioChunk, AudioEOF, ContractError, GenerationKey, integer


@dataclass(frozen=True)
class AudioWindow:
    key: GenerationKey
    frame_index: int
    audio_sample: int
    start_sample: int
    samples: tuple[float, ...]
    left_padding: int
    right_padding: int
    valid_audio_end: Fraction | None


class ClaireContext:
    """16 kHz, 8,320 samples, center offset 4,160. Caller drains before pushing.

    Resample with P01's stateful branch first. Context padding is model-only;
    every returned target is strictly before the real valid EOF boundary.
    """
    sample_rate = 16000
    window_samples = 8320
    offset_samples = 4160
    max_chunk_samples = 320

    def __init__(self, key: GenerationKey, *, fps: int = 60):
        integer(fps, "motion fps", 1)
        if fps > 120:
            raise ContractError("motion cadence exceeds qualified frontend bound")
        self.key, self.fps = key, fps
        self.ledger = SampleLedger(key, self.sample_rate)
        self.buffer = deque()
        self.origin = self.frame_index = self.peak_buffered_samples = 0
        self.cancelled = False
        self.eof = False
        self.max_buffered_samples = self.window_samples+self.max_chunk_samples

    @property
    def target_sample(self):
        return self.frame_index*self.sample_rate//self.fps

    @property
    def required_end_sample(self):
        return self.target_sample-self.offset_samples+self.window_samples

    def push(self, event: AudioChunk | AudioEOF) -> bool:
        if self.cancelled or event.envelope.key != self.key:
            return False
        if isinstance(event, AudioChunk):
            if event.sample_count > self.max_chunk_samples:
                raise ContractError("split motion input into at most 20 ms chunks")
            if len(self.buffer)+event.sample_count > self.max_buffered_samples:
                raise ContractError("motion context backpressure: drain ready windows first")
        elif not isinstance(event, AudioEOF):
            raise ContractError("motion context needs audio or explicit EOF")
        self.ledger.accept(event)
        if isinstance(event, AudioChunk):
            self.buffer.extend(event.samples)
            self.peak_buffered_samples = max(self.peak_buffered_samples,len(self.buffer))
        else:
            self.eof = True
        return True

    def pop(self) -> AudioWindow | None:
        if self.cancelled:
            return None
        center = self.target_sample
        if self.eof:
            if center >= self.ledger.valid_end_sample:
                self.buffer.clear()
                return None
        elif self.required_end_sample > self.ledger.next_sample:
            return None
        start = center-self.offset_samples
        end = start+self.window_samples
        left, right = max(0,-start), max(0,end-self.ledger.next_sample)
        real_start, real_end = max(start,0), min(end,self.ledger.next_sample)
        if real_start < self.origin:
            raise RuntimeError("motion context history was released too early")
        # A window is materialized only on demand, never accumulated per utterance.
        from itertools import islice
        real = tuple(islice(self.buffer,real_start-self.origin,real_end-self.origin))
        result = AudioWindow(self.key,self.frame_index,center,start,(0.0,)*left+real+(0.0,)*right,
                             left,right,self.ledger.valid_end_sample if self.eof else None)
        if len(result.samples) != self.window_samples:
            raise RuntimeError("motion context sample conservation failed")
        self.frame_index += 1
        keep = min(self.ledger.next_sample,max(0,self.target_sample-self.offset_samples))
        while self.origin < keep:
            self.buffer.popleft()
            self.origin += 1
        return result

    def cancel(self):
        self.cancelled = True
        self.buffer.clear()
