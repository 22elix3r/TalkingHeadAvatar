import pytest
from talkinghead_v2.conversation import TurnDetector,StableSegmenter
from talkinghead_v2.contracts import ContractError


@pytest.mark.parametrize('pause',[10,25,50])
def test_endpoint_keeps_200_500_1000ms_internal_pauses(pause):
    detector=TurnDetector();pcm=bytes(640)
    for _ in range(20): assert detector.push(pcm,True)[0] is None
    for _ in range(pause): assert detector.push(pcm,False)[0] is None
    for _ in range(20): assert detector.push(pcm,True)[0] is None
    result=None
    for _ in range(60): result,_=detector.push(pcm,False)
    assert result is not None
    assert result.committed_sample-result.last_voiced_sample==19200
    assert len(result.pcm)==(40+pause+4)*640


def test_turn_limit_is_explicit_and_buffer_is_bounded():
    detector=TurnDetector(max_frames=70)
    result=None
    for _ in range(70):
        value,_=detector.push(bytes(640),True)
        if value: result=value
    assert result.reason=='maximum_utterance' and len(result.pcm)<=70*640
    assert not detector.frames
    with pytest.raises(ContractError): detector.push(bytes(639),True)


def test_segmenter_keeps_numbers_abbreviations_and_final_tail():
    segmenter=StableSegmenter();output=[]
    for chunk in ['Dr. Green paid $','12.','50 at 3:45 p.m. ','It was a fair price. ','Yes.']:
        output+=segmenter.push(chunk)
    output+=segmenter.push('',final=True)
    assert ' '.join(output)=='Dr. Green paid $12.50 at 3:45 p.m. It was a fair price. Yes.'
    assert not any(s=='Dr.' for s in output)
    assert not segmenter.buffer


def test_segmenter_releases_only_complete_words_and_bounds_pending_text():
    segmenter=StableSegmenter()
    segmenter.push('A clear daytime sky is blu')
    assert segmenter.release_words()==['A clear daytime sky is']
    assert segmenter.push('e.',final=True)==['blue.']
    with pytest.raises(ContractError): StableSegmenter().push('x'*65537)
