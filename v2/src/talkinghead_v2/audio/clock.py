"""Integer cursors and rational timebase conversion; no wall-clock scheduling."""

from fractions import Fraction

from talkinghead_v2.contracts import AudioChunk, AudioEOF, ContractError, Discontinuity, GenerationKey, integer


def map_samples(position: int, source_rate: int, target_rate: int) -> Fraction:
    integer(position, "position")
    integer(source_rate, "source_rate", 1)
    integer(target_rate, "target_rate", 1)
    return Fraction(position * target_rate, source_rate)


class SampleLedger:
    """One generation/rate. Reject holes/reordering before changing the cursor."""

    def __init__(self, key: GenerationKey, sample_rate: int):
        integer(sample_rate, "sample_rate", 1)
        self.key = key
        self.sample_rate = sample_rate
        self.next_sample = 0
        self.last_sequence = -1
        self.closed = False
        self.valid_end_sample: Fraction | None = None

    def accept(self, event: AudioChunk | AudioEOF | Discontinuity) -> None:
        if event.envelope.key != self.key or event.sample_rate != self.sample_rate:
            raise ContractError("ledger generation/rate mismatch")
        if self.closed:
            raise ContractError("ledger already closed; queue-empty is not EOF")
        if event.envelope.sequence_no <= self.last_sequence:
            raise ContractError("duplicate/reordered audio sequence")
        position = event.start_sample if isinstance(event, AudioChunk) else (
            event.end_sample if isinstance(event, AudioEOF) else event.at_sample
        )
        if position != self.next_sample:
            raise ContractError(f"gap/overlap: expected sample {self.next_sample}, got {position}")
        if isinstance(event, AudioChunk):
            self.next_sample = event.end_sample
        else:
            self.closed = True
            if isinstance(event, AudioEOF):
                self.valid_end_sample = event.valid_end_sample
        self.last_sequence = event.envelope.sequence_no
