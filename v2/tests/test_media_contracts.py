from dataclasses import replace
from fractions import Fraction

import pytest

from talkinghead_v2.audio.clock import SampleLedger, map_samples
from talkinghead_v2.audio.motion import interpolate_motion
from talkinghead_v2.contracts import (
    AudioChunk, AudioEOF, CancelAck, ContractError, Discontinuity, Envelope,
    GenerationKey, MotionFrame, ProviderCapabilities, RigIdentity, TextSegment,
)


KEY = GenerationKey("session", "turn", 0)
RIG = RigIdentity("fixture-rig", "sha256:"+"0"*64, "sha256:"+"1"*64, ("lip", "brow"))


def envelope(sequence=0, key=KEY):
    return Envelope(key, sequence, "trace")


def chunk(start=0, samples=(0.1, 0.2), sequence=0, key=KEY, rate=24000):
    return AudioChunk(envelope(sequence, key), "segment", rate, start, samples)


def test_ledger_rejects_holes_overlap_duplicates_wrong_generation_and_rate_without_mutation():
    ledger = SampleLedger(KEY, 24000)
    ledger.accept(chunk())
    invalid = [chunk(start=3, sequence=1), chunk(start=1, sequence=1), chunk(start=2, sequence=0),
               chunk(start=2, sequence=1, rate=48000)]
    for key in (replace(KEY, session_id="other"), replace(KEY, turn_id="other"), replace(KEY, epoch=1)):
        invalid.append(chunk(start=2, sequence=1, key=key))
    for event in invalid:
        with pytest.raises(ContractError):
            ledger.accept(event)
        assert ledger.next_sample == 2 and ledger.last_sequence == 0 and not ledger.closed
    ledger.accept(chunk(start=2, samples=(), sequence=1))
    ledger.accept(chunk(start=2, samples=(0.0,), sequence=3))
    ledger.accept(AudioEOF(envelope(4), 24000, 3))
    assert ledger.closed and ledger.valid_end_sample == 3
    with pytest.raises(ContractError, match="closed"):
        ledger.accept(AudioEOF(envelope(4), 24000, 3))


def test_eof_discontinuity_and_empty_utterance_have_explicit_different_meanings():
    empty = SampleLedger(KEY, 16000)
    assert not empty.closed  # An empty queue/buffer is not end-of-stream.
    empty.accept(AudioEOF(envelope(), 16000, 0))
    assert empty.valid_end_sample == 0
    discontinuous = SampleLedger(KEY, 24000)
    discontinuous.accept(chunk())
    with pytest.raises(ContractError, match="gap/overlap"):
        discontinuous.accept(Discontinuity(envelope(1), 24000, 3, "missing callback"))
    discontinuous.accept(Discontinuity(envelope(1), 24000, 2, "reset"))
    assert discontinuous.closed and discontinuous.valid_end_sample is None
    with pytest.raises(ContractError):
        discontinuous.accept(chunk(start=2, sequence=2))


def test_exact_clock_mapping_beyond_float_integer_precision_and_fractional_eof():
    position = 2**53+1
    mapped = map_samples(position, 44100, 48000)
    assert mapped == Fraction(position*160, 147)
    assert mapped * Fraction(44100, 48000) == position
    assert map_samples(1, 48000, 10**9) == Fraction(62500, 3)
    event = AudioEOF(envelope(), 48000, 2, Fraction(3, 2))
    assert event.valid_end_sample == Fraction(3, 2)
    for valid in (Fraction(1), Fraction(3), 1.5):
        with pytest.raises(ContractError):
            AudioEOF(envelope(), 48000, 2, valid)


@pytest.mark.parametrize("change", [
    {"samples": [0.1]}, {"samples": (float("nan"),)}, {"samples": (float("inf"),)},
    {"samples": (True,)}, {"sample_rate": 0}, {"sample_rate": True},
    {"channels": 2}, {"start_sample": -1}, {"start_sample": 2**63}, {"envelope": None},
])
def test_invalid_pcm_is_rejected_before_entering_pipeline(change):
    with pytest.raises(ContractError):
        replace(chunk(), **change)


def test_text_capabilities_and_epoch_contracts_do_not_claim_streaming_by_default():
    text = TextSegment(envelope(), "segment", "Committed text.", "en", "opaque-voice")
    assert not text.is_last
    caps = ProviderCapabilities("synthetic", "fixture-v1", "reference-v1", (24000,),
                                False, True, "cooperative", 24000, 32, 32, 2400)
    assert not caps.incremental_text_input and caps.incremental_audio_output
    for changes in ({"native_rates": ()}, {"native_rates": (24000, 24000)},
                    {"incremental_audio_output": "yes"}, {"max_chunk_samples": 0},
                    {"right_context_samples": -1}, {"cancellation": "instant"}):
        with pytest.raises(ContractError):
            replace(caps, **changes)
    with pytest.raises(ContractError):
        replace(text, text=" ")
    with pytest.raises(ContractError):
        Envelope(KEY, 0, "trace", schema_version=2)
    with pytest.raises(ContractError):
        CancelAck(KEY, 0, "cancel", 0, 0, 0, 48000)


def motion(sample=0, **changes):
    frame = MotionFrame(envelope(sample), RIG, sample, 48000, sample+960, (0.0, 0.2))
    return replace(frame, **changes)


def test_motion_interpolates_audio_positions_with_unit_quaternions():
    left = motion()
    right = motion(480, coefficients=(1.0, 0.8), head_xyzw=(0.0, 1.0, 0.0, 0.0))
    midpoint = interpolate_motion(left, right, 240)
    assert midpoint.audio_sample == 240
    assert midpoint.coefficients == pytest.approx((0.5, 0.5))
    assert midpoint.head_xyzw == pytest.approx((0, 2**-0.5, 0, 2**-0.5))
    opposite = motion(480, head_xyzw=(0, 0, 0, -1))
    assert interpolate_motion(left, opposite, 240).head_xyzw == (0, 0, 0, 1)


@pytest.mark.parametrize("changes", [
    {"envelope": envelope(1, replace(KEY, epoch=1))},
    {"envelope": envelope(1, replace(KEY, turn_id="other"))},
    {"envelope": envelope(1, replace(KEY, session_id="other"))},
    {"rig": replace(RIG, basis_hash="sha256:"+"2"*64)},
    {"rig": replace(RIG, topology_hash="sha256:"+"2"*64)},
    {"rig": replace(RIG, coefficient_names=("brow", "lip"))}, {"sample_rate": 16000},
])
def test_equal_dimensions_never_override_motion_identity_or_epoch(changes):
    with pytest.raises(ContractError, match="cannot interpolate"):
        interpolate_motion(motion(), motion(480, **changes), 240)


def test_motion_refuses_padding_invalid_quaternions_gaps_or_extrapolation():
    for changes in ({"coefficients": (0.1,)}, {"head_xyzw": (0, 0, 0, 2)}, {"valid_until": -1}):
        with pytest.raises(ContractError):
            motion(**changes)
    with pytest.raises(ContractError, match="validity gap"):
        interpolate_motion(motion(valid_until=100), motion(480), 240)
    for target in (-1, 481):
        with pytest.raises(ContractError):
            interpolate_motion(motion(), motion(480), target)
