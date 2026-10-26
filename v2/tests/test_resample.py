from fractions import Fraction
import itertools
import math
import random

import pytest

from talkinghead_v2.audio.resample import RationalResampler
from talkinghead_v2.contracts import ContractError


@pytest.mark.parametrize("rates", list(itertools.product((16000, 24000, 44100, 48000), repeat=2)))
def test_chunk_independence_exact_counts_phase_and_bounded_history(rates):
    source, target = rates
    pcm = tuple(((i*31) % 1000)/2000 for i in range(1009))
    whole = RationalResampler(source, target)
    expected = whole.push(pcm) + whole.finish()
    stream = RationalResampler(source, target)
    rng = random.Random(17)
    actual = []
    cursor = 0
    while cursor < len(pcm):
        count = rng.randint(1, 113)
        assert stream.push(()) == ()
        actual.extend(stream.push(pcm[cursor:cursor+count]))
        cursor += count
        assert stream.buffered_samples <= 2*stream.radius
    actual.extend(stream.finish())
    assert tuple(actual) == expected
    assert len(actual) == (len(pcm)*target+source-1)//source
    assert stream.valid_end_sample == Fraction(len(pcm)*target, source)
    assert stream.phase == Fraction((len(actual)*source) % target, target)
    assert stream.buffered_samples == 0 and stream.finish() == ()


@pytest.mark.parametrize("count", [0, 1, 2, 31, 32, 33, 97])
@pytest.mark.parametrize("rates", [(24000, 48000), (48000, 16000), (44100, 48000)])
def test_tiny_and_partial_final_tails_are_flushed_once(count, rates):
    source, target = rates
    stream = RationalResampler(source, target)
    before = stream.push(tuple(0.25 for _ in range(count)))
    tail = stream.finish()
    assert len(before)+len(tail) == (count*target+source-1)//source
    assert stream.finish() == ()
    if count:
        assert tail and any(sample != 0 for sample in tail)
        assert Fraction((len(before)+len(tail)-1)*source, target) < count
    with pytest.raises(ContractError, match="closed"):
        stream.push((0.25,))


def test_independent_direct_fir_oracle_for_impulse_and_fractional_positions():
    # Independently evaluate the mathematical window at every output sample.
    source, target, radius = 16000, 24000, 4
    pcm = (0.0, 0.0, 1.0, 0.0, -0.25, 0.0, 0.0)
    cutoff = 0.94
    expected = []
    for j in range(math.ceil(len(pcm)*target/source)):
        position = Fraction(j*source, target)
        center = position.__floor__()
        weights, indices = [], []
        for i in range(center-radius+1, center+radius+1):
            distance = float(Fraction(i)-position)
            z = cutoff*distance
            sinc = 1 if z == 0 else math.sin(math.pi*z)/(math.pi*z)
            window = (1+math.cos(math.pi*distance/radius))/2 if abs(distance) < radius else 0
            weights.append(cutoff*sinc*window)
            indices.append(i)
        expected.append(sum(weight*pcm[i] for weight, i in zip(weights, indices) if 0 <= i < len(pcm))/sum(weights))
    stream = RationalResampler(source, target, radius=radius)
    actual = stream.push(pcm) + stream.finish()
    assert actual == pytest.approx(expected, abs=1e-14)


def test_filter_preserves_dc_silence_and_rejects_downsampling_aliases():
    def convert(samples):
        stream = RationalResampler(48000, 16000)
        return (stream.push(samples)+stream.finish())[32:-32]

    dc = convert((0.25,)*4800)
    assert dc == pytest.approx((0.25,)*len(dc), abs=1e-12)
    assert all(x == 0 for x in convert((0.0,)*4800))
    stopband = convert(tuple(math.sin(2*math.pi*20000*i/48000) for i in range(4800)))
    passband = convert(tuple(math.sin(2*math.pi*1000*i/48000) for i in range(4800)))
    assert math.sqrt(sum(x*x for x in stopband)/len(stopband)) < 0.005
    assert math.sqrt(sum(x*x for x in passband)/len(passband)) > 0.6


def test_stream_emits_before_eof_and_cancellation_does_not_flush():
    stream = RationalResampler(24000, 48000)
    assert stream.right_context == Fraction(32, 24000)
    body = stream.push((0.5,)*100)
    assert body and stream.buffered_samples > 0
    stream.cancel()
    assert stream.finish() == () and stream.buffered_samples == 0
    same = RationalResampler(48000, 48000)
    assert same.right_context == 0
    assert same.push((0.1, 0.2)) == (0.1, 0.2)
    assert same.finish() == ()
