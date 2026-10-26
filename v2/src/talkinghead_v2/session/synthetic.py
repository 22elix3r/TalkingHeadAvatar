"""Deterministic scheduling recorder. This is NOT a device playback acknowledgment."""

from dataclasses import dataclass
from fractions import Fraction

from talkinghead_v2.audio.clock import SampleLedger
from talkinghead_v2.contracts import AudioChunk, AudioEOF, AudioEvent, ContractError, GenerationKey, MotionFrame, RigIdentity, integer


@dataclass(frozen=True)
class ScheduledPCM:
    chunk: AudioChunk
    session_start_sample: int
    start_ns: Fraction
    end_ns: Fraction


class SyntheticSink:
    """Unbounded recorder for small test fixtures only; no audio hardware or sleeps."""

    def __init__(self, sample_rate: int = 48000):
        integer(sample_rate, "sample_rate", 1)
        self.sample_rate = sample_rate
        self.key: GenerationKey | None = None
        self.rig: RigIdentity | None = None
        self.records: list[ScheduledPCM] = []
        self.motion: list[MotionFrame] = []
        self.eofs: list[AudioEOF] = []
        self.transport_cursor = 0
        self.base_sample = 0
        self.ledger: SampleLedger | None = None
        self.finished = False
        self.rejected_stale = 0

    def start(self, key: GenerationKey, rig: RigIdentity | None) -> None:
        self.key = key
        self.rig = rig
        self.base_sample = self.transport_cursor
        self.ledger = SampleLedger(key, self.sample_rate)
        self.finished = False

    def submit(self, event: AudioEvent) -> bool:
        # Check again here: a callback may have been dequeued before cancellation.
        if event.envelope.key != self.key:
            self.rejected_stale += 1
            return False
        if self.ledger is None:
            raise ContractError("synthetic sink has no active ledger")
        self.ledger.accept(event)
        if isinstance(event, AudioChunk):
            start = self.base_sample + event.start_sample
            end = self.base_sample + event.end_sample
            self.records.append(ScheduledPCM(event, start, Fraction(start*10**9, self.sample_rate),
                                             Fraction(end*10**9, self.sample_rate)))
            self.transport_cursor = end
        else:
            self.finished = True
            self.eofs.append(event)
        return True

    def submit_motion(self, frame: MotionFrame) -> bool:
        if frame.envelope.key != self.key:
            self.rejected_stale += 1
            return False
        if frame.rig != self.rig or frame.sample_rate != self.sample_rate:
            raise ContractError("motion rig/basis/timebase mismatch")
        if self.finished:
            raise ContractError("motion after EOF")
        previous = next((item for item in reversed(self.motion) if item.envelope.key == self.key), None)
        if previous is not None and (frame.audio_sample < previous.audio_sample or
                                     frame.envelope.sequence_no <= previous.envelope.sequence_no):
            raise ContractError("reordered motion")
        self.motion.append(frame)
        return True

    def cancel(self) -> int:
        last = self.ledger.next_sample if self.ledger is not None else 0
        self.key = None
        self.rig = None
        return last
