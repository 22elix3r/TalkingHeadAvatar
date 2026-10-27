from fractions import Fraction

import pytest

from talkinghead_v2.contracts import AudioChunk, AudioEOF, ContractError, Envelope, GenerationKey
from talkinghead_v2.providers.motion.claire import ClaireContext


KEY=GenerationKey('s','t',0)


def push(context,start,count,sequence):
    return context.push(AudioChunk(Envelope(KEY,sequence,'test'),'segment',16000,start,
                                    tuple(float(i) for i in range(start,start+count))))


def drain(context):
    values=[]
    while (window:=context.pop()) is not None: values.append(window)
    return values


def test_claire_requires_real_future_context_and_does_not_treat_empty_queue_as_eof():
    context=ClaireContext(KEY)
    for i in range(13): push(context,i*320,320,i)
    first=context.pop()
    assert first.audio_sample==0 and first.left_padding==4160 and first.right_padding==0
    assert first.samples[:4160]==(0.,)*4160
    assert first.samples[4160:]==tuple(float(i) for i in range(4160))
    assert first.valid_audio_end is None and context.pop() is None
    assert context.required_end_sample==4426


@pytest.mark.parametrize('count',[0,1,17,321,10001])
def test_eof_context_padding_never_becomes_playable_or_creates_past_eof_targets(count):
    context=ClaireContext(KEY)
    windows=[];sequence=0
    for start in range(0,count,320):
        push(context,start,min(320,count-start),sequence);sequence+=1
        windows.extend(drain(context))
    context.push(AudioEOF(Envelope(KEY,sequence,'test'),16000,count))
    windows.extend(drain(context))
    assert len(windows)==(count*60+15999)//16000
    for window in windows:
        assert len(window.samples)==8320 and window.audio_sample<count
        for j,value in enumerate(window.samples):
            position=window.start_sample+j
            assert value==(float(position) if 0<=position<count else 0.)
    assert not context.buffer and context.pop() is None


def test_fractional_valid_tail_disallows_model_target_on_padding_boundary():
    context=ClaireContext(KEY)
    push(context,0,267,0)
    context.push(AudioEOF(Envelope(KEY,1,'test'),16000,267,Fraction(1065,4)))
    assert [w.audio_sample for w in drain(context)]==[0,266]


def test_context_backpressure_is_explicit_and_cancel_rejects_late_audio():
    context=ClaireContext(KEY)
    for i in range(27): push(context,i*320,320,i)
    assert context.peak_buffered_samples==8640
    with pytest.raises(ContractError,match='backpressure'): push(context,8640,320,27)
    assert context.ledger.next_sample==8640
    held=context.pop()
    context.cancel()
    assert held.key==KEY and context.pop() is None and not context.buffer
    assert not push(context,8640,1,27)


def test_context_refuses_rate_gap_reorder_and_oversized_work():
    context=ClaireContext(KEY)
    with pytest.raises(ContractError,match='20 ms'): push(context,0,321,0)
    with pytest.raises(ContractError,match='gap'): push(context,1,10,0)
    with pytest.raises(ContractError,match='rate'):
        context.push(AudioChunk(Envelope(KEY,0,'test'),'segment',24000,0,(0.,)))
