"""Versioned in-process media contracts. All sample ranges are half open."""

from dataclasses import dataclass
from fractions import Fraction
import math
import re
from typing import Literal


class ContractError(ValueError):
    """Invalid media, identity, ordering or compatibility."""


def integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise ContractError(f"{name} must be an integer in [{minimum}, 2**63-1]")


def label(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a nonempty string")


def numeric_tuple(values: tuple[float, ...], name: str, size: int | None = None) -> None:
    if not isinstance(values, tuple) or (size is not None and len(values) != size):
        raise ContractError(f"{name} must be an immutable tuple of size {size}")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ContractError(f"{name} must contain finite numbers")


@dataclass(frozen=True)
class GenerationKey:
    session_id: str
    turn_id: str
    epoch: int

    def __post_init__(self) -> None:
        label(self.session_id, "session_id")
        label(self.turn_id, "turn_id")
        integer(self.epoch, "epoch")


@dataclass(frozen=True)
class Envelope:
    key: GenerationKey
    sequence_no: int
    trace_id: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.key, GenerationKey):
            raise ContractError("key must be a GenerationKey")
        integer(self.sequence_no, "sequence_no")
        label(self.trace_id, "trace_id")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ContractError("unsupported envelope schema version")


@dataclass(frozen=True)
class TextSegment:
    envelope: Envelope
    segment_id: str
    text: str
    language: str
    voice_id: str
    is_last: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, Envelope):
            raise ContractError("envelope must be an Envelope")
        for name in ("segment_id", "text", "language", "voice_id"):
            label(getattr(self, name), name)
        if type(self.is_last) is not bool:
            raise ContractError("is_last must be boolean")


@dataclass(frozen=True)
class AudioChunk:
    envelope: Envelope
    segment_id: str
    sample_rate: int
    start_sample: int
    samples: tuple[float, ...]
    channels: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, Envelope):
            raise ContractError("envelope must be an Envelope")
        label(self.segment_id, "segment_id")
        integer(self.sample_rate, "sample_rate", 1)
        integer(self.start_sample, "start_sample")
        numeric_tuple(self.samples, "samples")
        if type(self.channels) is not int or self.channels != 1:
            raise ContractError("P01 explicitly supports mono PCM only")
        integer(self.end_sample, "end_sample")

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def end_sample(self) -> int:
        return self.start_sample + self.sample_count

    @property
    def duration(self) -> Fraction:
        return Fraction(self.sample_count, self.sample_rate)


@dataclass(frozen=True)
class AudioEOF:
    envelope: Envelope
    sample_rate: int
    end_sample: int
    # Exact valid boundary in this rate, including a fractional final sample interval.
    valid_end_sample: Fraction | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, Envelope):
            raise ContractError("envelope must be an Envelope")
        integer(self.sample_rate, "sample_rate", 1)
        integer(self.end_sample, "end_sample")
        valid = Fraction(self.end_sample) if self.valid_end_sample is None else self.valid_end_sample
        if not isinstance(valid, Fraction):
            raise ContractError("valid_end_sample must be an exact Fraction")
        if (self.end_sample == 0 and valid != 0) or (
            self.end_sample > 0 and not self.end_sample - 1 < valid <= self.end_sample
        ):
            raise ContractError("EOF valid boundary must lie in its final sample interval")
        object.__setattr__(self, "valid_end_sample", valid)


@dataclass(frozen=True)
class Discontinuity:
    envelope: Envelope
    sample_rate: int
    at_sample: int
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, Envelope):
            raise ContractError("envelope must be an Envelope")
        integer(self.sample_rate, "sample_rate", 1)
        integer(self.at_sample, "at_sample")
        label(self.reason, "reason")


@dataclass(frozen=True)
class EpochStarted:
    key: GenerationKey
    native_rate: int
    output_rate: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, GenerationKey):
            raise ContractError("key must be a GenerationKey")
        integer(self.native_rate, "native_rate", 1)
        integer(self.output_rate, "output_rate", 1)


@dataclass(frozen=True)
class CancelAck:
    cancelled_key: GenerationKey
    fence_epoch: int
    reason: str
    discarded_queue_samples: int
    discarded_queue_items: int
    last_scheduled_sample: int
    sample_rate: int
    # Software queue fence only. Device/compute stop require separate acknowledgments.
    scope: Literal["synthetic_in_process", "local_device_queue_only"] = "synthetic_in_process"

    def __post_init__(self) -> None:
        if not isinstance(self.cancelled_key, GenerationKey):
            raise ContractError("cancelled_key must be a GenerationKey")
        integer(self.fence_epoch, "fence_epoch")
        if self.fence_epoch <= self.cancelled_key.epoch:
            raise ContractError("cancellation must advance the epoch fence")
        label(self.reason, "reason")
        integer(self.sample_rate, "sample_rate", 1)
        for name in ("discarded_queue_samples", "discarded_queue_items", "last_scheduled_sample"):
            integer(getattr(self, name), name)
        if self.scope not in ("synthetic_in_process", "local_device_queue_only"):
            raise ContractError("device/provider cancellation is not qualified by this acknowledgment")


@dataclass(frozen=True)
class ProviderCapabilities:
    model_id: str
    model_revision: str
    runtime_revision: str
    native_rates: tuple[int, ...]
    incremental_text_input: bool
    incremental_audio_output: bool
    cancellation: Literal["cooperative", "worker_restart", "none"]
    context_rate: int
    left_context_samples: int
    right_context_samples: int
    max_chunk_samples: int
    peak_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in ("model_id", "model_revision", "runtime_revision"):
            label(getattr(self, name), name)
        if not isinstance(self.native_rates, tuple) or not self.native_rates:
            raise ContractError("native_rates must be a nonempty tuple")
        for rate in self.native_rates:
            integer(rate, "native_rate", 1)
        if len(set(self.native_rates)) != len(self.native_rates):
            raise ContractError("duplicate native rates")
        for name in ("incremental_text_input", "incremental_audio_output"):
            if type(getattr(self, name)) is not bool:
                raise ContractError(f"{name} must be boolean")
        if self.cancellation not in ("cooperative", "worker_restart", "none"):
            raise ContractError("unknown cancellation capability")
        integer(self.context_rate, "context_rate", 1)
        integer(self.left_context_samples, "left_context_samples")
        integer(self.right_context_samples, "right_context_samples")
        integer(self.max_chunk_samples, "max_chunk_samples", 1)
        if self.peak_memory_bytes is not None:
            integer(self.peak_memory_bytes, "peak_memory_bytes")


@dataclass(frozen=True)
class RigIdentity:
    schema_id: str
    basis_hash: str
    topology_hash: str
    coefficient_names: tuple[str, ...]
    coordinates: Literal["right_handed_y_up"] = "right_handed_y_up"
    pose_semantics: Literal["absolute_from_canonical", "delta_from_neutral"] = "absolute_from_canonical"

    def __post_init__(self) -> None:
        label(self.schema_id, "schema_id")
        for name in ("basis_hash", "topology_hash"):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", getattr(self, name)):
                raise ContractError(f"{name} must be a SHA-256 identity")
        if not isinstance(self.coefficient_names, tuple) or not self.coefficient_names:
            raise ContractError("coefficient_names must be a nonempty tuple")
        for name in self.coefficient_names:
            label(name, "coefficient name")
        if len(set(self.coefficient_names)) != len(self.coefficient_names):
            raise ContractError("coefficient names must be unique")
        if self.coordinates != "right_handed_y_up" or self.pose_semantics not in (
            "absolute_from_canonical", "delta_from_neutral"
        ):
            raise ContractError("unsupported coordinate/pose conventions")


@dataclass(frozen=True)
class MotionFrame:
    envelope: Envelope
    rig: RigIdentity
    audio_sample: int
    sample_rate: int
    valid_until: int
    coefficients: tuple[float, ...]
    jaw_xyzw: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0)
    head_xyzw: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0)
    translation_m: tuple[float, ...] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, Envelope) or not isinstance(self.rig, RigIdentity):
            raise ContractError("motion needs a typed envelope and rig identity")
        integer(self.audio_sample, "audio_sample")
        integer(self.sample_rate, "sample_rate", 1)
        integer(self.valid_until, "valid_until")
        if self.valid_until < self.audio_sample:
            raise ContractError("motion validity ends before its audio position")
        numeric_tuple(self.coefficients, "coefficients", len(self.rig.coefficient_names))
        numeric_tuple(self.translation_m, "translation_m", 3)
        for name in ("jaw_xyzw", "head_xyzw"):
            values = getattr(self, name)
            numeric_tuple(values, name, 4)
            if not math.isclose(sum(x*x for x in values), 1.0, abs_tol=1e-6):
                raise ContractError(f"{name} must be a unit quaternion in xyzw order")


AudioEvent = AudioChunk | AudioEOF
