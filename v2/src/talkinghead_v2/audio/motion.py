"""Bounded interpolation in one generation, rig and audio timebase only."""

from dataclasses import replace
import math

from talkinghead_v2.contracts import ContractError, Envelope, MotionFrame, integer


def _slerp(a: tuple[float, ...], b: tuple[float, ...], fraction: float) -> tuple[float, ...]:
    dot = sum(x*y for x, y in zip(a, b, strict=True))
    if dot < 0:
        b = tuple(-x for x in b)
        dot = -dot
    if dot > 0.9995:
        values = tuple(x+(y-x)*fraction for x, y in zip(a, b, strict=True))
        norm = math.sqrt(sum(x*x for x in values))
        return tuple(x/norm for x in values)
    angle = math.acos(min(1.0, dot))
    return tuple((math.sin((1-fraction)*angle)*x + math.sin(fraction*angle)*y) / math.sin(angle)
                 for x, y in zip(a, b, strict=True))


def interpolate_motion(left: MotionFrame, right: MotionFrame, audio_sample: int) -> MotionFrame:
    integer(audio_sample, "audio_sample")
    if left.envelope.key != right.envelope.key or left.rig != right.rig or left.sample_rate != right.sample_rate:
        raise ContractError("cannot interpolate across epochs, turns, sessions, rigs or timebases")
    if not left.audio_sample <= audio_sample <= right.audio_sample or left.audio_sample >= right.audio_sample:
        raise ContractError("interpolation must stay within a forward motion interval")
    if left.valid_until < right.audio_sample:
        raise ContractError("motion interval crosses a validity gap")
    fraction = (audio_sample-left.audio_sample) / (right.audio_sample-left.audio_sample)

    def mix(a, b):
        return tuple(x+(y-x)*fraction for x, y in zip(a, b, strict=True))

    return replace(left, envelope=Envelope(left.envelope.key,
                   max(left.envelope.sequence_no, right.envelope.sequence_no)+1, left.envelope.trace_id),
                   audio_sample=audio_sample, valid_until=right.valid_until,
                   coefficients=mix(left.coefficients, right.coefficients),
                   translation_m=mix(left.translation_m, right.translation_m),
                   jaw_xyzw=_slerp(left.jaw_xyzw, right.jaw_xyzw, fraction),
                   head_xyzw=_slerp(left.head_xyzw, right.head_xyzw, fraction))
