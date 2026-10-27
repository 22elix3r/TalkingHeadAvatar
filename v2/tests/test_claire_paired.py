"""Cross-rate stream/EOF/cancellation fixtures without models, devices or I/O."""
from fractions import Fraction
import math
import struct

import pytest

from talkinghead_v2.audio.resample import RationalResampler
from talkinghead_v2.contracts import AudioChunk, AudioEOF, ContractError, Envelope, GenerationKey
from talkinghead_v2.providers.motion.paired import ClaireInputMux

KEY = GenerationKey('session', 'turn', 3)


@pytest.mark.parametrize('size', [1, 2, 239, 240, 479, 480, 481, 800, 997, 1600, 3333])
@pytest.mark.parametrize('quanta', [(2400,), (1, 17, 319, 2, 997)])
def test_irregular_qwen_chunks_preserve_both_pcm_domains_and_one_eof(size, quanta):
    samples = tuple(math.sin(index * .071) * .6 for index in range(size))
    mux = ClaireInputMux(KEY)
    cursor = sequence = 0
    packets = []
    while cursor < size:
        count = min(quanta[sequence % len(quanta)], size - cursor)
        packets.extend(mux.accept(AudioChunk(Envelope(KEY, sequence, 'test'), 'segment', 24000,
                                            cursor, samples[cursor:cursor + count])))
        cursor += count
        sequence += 1
    packets.extend(mux.accept(AudioEOF(Envelope(KEY, sequence, 'test'), 24000, cursor)))
    audio, motion = [], []
    for index, packet in enumerate(packets):
        header = struct.unpack('<cQQQII', packet.packed()[:33])
        assert header == (b'F' if index == len(packets)-1 else b'D', 3, index, len(audio),
                          len(packet.audio), len(packet.motion))
        if not packet.eof:
            assert len(packet.audio) == 960 and len(packet.motion) == 320
        else:
            assert len(packet.audio) < 960 and len(packet.motion) == (len(packet.audio) + 2) // 3
        audio.extend(packet.audio)
        motion.extend(packet.motion)
    playback = RationalResampler(24000, 48000)
    expected = playback.push(samples) + playback.finish()
    assert audio == [int(max(-1., min(1., value)) * 32767) for value in expected]
    neural = RationalResampler(24000, 16000)
    assert motion == list(neural.push(samples) + neural.finish())
    assert mux.sent_samples == 2 * size and mux.sent_motion_samples == (2 * size + 2) // 3
    assert sum(packet.eof for packet in packets) == 1


def test_stale_input_cannot_advance_filter_history_and_cancel_never_flushes_tail():
    mux = ClaireInputMux(KEY)
    other = GenerationKey('session', 'other', 2)
    assert mux.accept(AudioChunk(Envelope(other, 0, 'test'), 'segment', 24000, 0, (1.,))) == ()
    assert mux.playback.input_count == mux.motion.input_count == mux.ledger.next_sample == 0
    assert mux.accept(AudioChunk(Envelope(KEY, 0, 'test'), 'segment', 24000, 0, (.5,) * 30)) == ()
    mux.cancel()
    assert mux.accept(AudioEOF(Envelope(KEY, 1, 'test'), 24000, 30)) == ()
    assert mux.sent_samples == mux.sent_motion_samples == 0
    assert mux.playback.buffered_samples == mux.motion.buffered_samples == 0


def test_gap_invalid_eof_and_oversized_input_cannot_mutate_ledger():
    mux = ClaireInputMux(KEY)
    with pytest.raises(ContractError, match='chunk'):
        mux.accept(AudioChunk(Envelope(KEY, 0, 'test'), 'segment', 24000, 0, (0.,) * 2401))
    with pytest.raises(ContractError, match='gap'):
        mux.accept(AudioChunk(Envelope(KEY, 0, 'test'), 'segment', 24000, 1, (0.,)))
    with pytest.raises(ContractError, match='nonempty'):
        mux.accept(AudioEOF(Envelope(KEY, 0, 'test'), 24000, 0))
    assert mux.ledger.next_sample == mux.playback.input_count == 0
    mux.accept(AudioChunk(Envelope(KEY, 0, 'test'), 'segment', 24000, 0, (0.,) * 2))
    with pytest.raises(ContractError, match='whole'):
        mux.accept(AudioEOF(Envelope(KEY, 1, 'test'), 24000, 2, Fraction(3, 2)))
    assert not mux.ledger.closed and mux.ledger.next_sample == 2


def test_failed_motion_amplitude_cannot_flush_a_partial_audio_tail():
    mux = ClaireInputMux(KEY)
    with pytest.raises(ContractError, match='amplitude'):
        mux.accept(AudioChunk(Envelope(KEY, 0, 'test'), 'segment', 24000, 0, (5.,) * 1000))
    assert mux.cancelled and mux.sent_samples == mux.sent_motion_samples == 0
    assert mux.accept(AudioEOF(Envelope(KEY, 1, 'test'), 24000, 1000)) == ()
    assert mux.playback.buffered_samples == mux.motion.buffered_samples == 0
