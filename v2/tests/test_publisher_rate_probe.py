import pytest
from talkinghead_v2.contracts import ContractError
from talkinghead_v2.publish_probe import publisher_rate_probe


def test_rate_feedback_keeps_requested_and_applied_rates_separate():
    result=publisher_rate_probe(['vendor log','THA_RATE_PROBE 600000 2',
        'THA_RATE_FEEDBACK 0 900000 600000 30','THA_RATE_FEEDBACK 48000 50000 100000 24'],600000)
    assert result['initial_bitrate_bps']==600000
    assert result['feedback'][0]['requested_bitrate_bps']==900000
    assert result['feedback'][1]['applied_bitrate_bps']==100000
    assert result['clock']=='publisher_stream_samples_48000hz'


@pytest.mark.parametrize('lines',[
    [],['THA_RATE_PROBE 600000 0'],['THA_RATE_PROBE 300000 1','THA_RATE_FEEDBACK 0 600000 600000 30'],
    ['THA_RATE_PROBE 600000 2','THA_RATE_FEEDBACK 0 600000 600000 30'],
    ['THA_RATE_PROBE 600000 1','THA_RATE_PROBE 600000 1','THA_RATE_FEEDBACK 0 600000 600000 30'],
    ['THA_RATE_PROBE 600000 1','THA_RATE_FEEDBACK 0 900000 900000 30'],
    ['THA_RATE_PROBE 600000 1','THA_RATE_FEEDBACK 0 0 100000 30'],
    ['THA_RATE_PROBE 600000 1','THA_RATE_FEEDBACK -1 600000 600000 30'],
    ['THA_RATE_PROBE 600000 1','THA_RATE_FEEDBACK 0 600000 600000 nan'],
    ['THA_RATE_PROBE 600000 1','THA_RATE_FEEDBACK 0 600000 600000 61'],
    ['THA_RATE_PROBE 600000 1','THA_RATE_FEEDBACK 0 bad 600000 30'],
    ['THA_RATE_PROBE 600000 2','THA_RATE_FEEDBACK 480 600000 600000 30','THA_RATE_FEEDBACK 0 600000 600000 30'],
    ['THA_RATE_PROBE 600000 257']+['THA_RATE_FEEDBACK 0 600000 600000 30']*257,
])
def test_rate_feedback_rejects_missing_malformed_truncated_or_inconsistent_observation(lines):
    with pytest.raises(ContractError):publisher_rate_probe(lines,600000)


SCHEDULED=['THA_RATE_PROBE_V2 1200000 down-up 5',
    'THA_RATE_FEEDBACK_V2 0 0 1200000 1200000 30 initial',
    'THA_RATE_FEEDBACK_V2 48000 46400 900000 900000 30 feedback',
    'THA_RATE_FEEDBACK_V2 145600 144000 900000 300000 30 ceiling',
    'THA_RATE_FEEDBACK_V2 193600 192000 800000 300000 30 feedback',
    'THA_RATE_FEEDBACK_V2 289600 288000 1100000 1100000 30 ceiling']


def test_in_turn_feedback_preserves_target_during_ceiling_changes():
    result=publisher_rate_probe(SCHEDULED,1200000,schedule='down-up')
    assert result['feedback'][2]['requested_bitrate_bps']==900000
    assert result['feedback'][2]['applied_bitrate_bps']==300000
    assert result['feedback'][-1]['applied_bitrate_bps']==1100000
    assert result['feedback'][-1]['source_sample']==288000


@pytest.mark.parametrize('rows',[
    SCHEDULED[:-1],
    [SCHEDULED[0],*SCHEDULED[1:3],SCHEDULED[3].replace('300000 30','900000 30'),*SCHEDULED[4:]],
    [SCHEDULED[0],*SCHEDULED[1:3],SCHEDULED[3].replace('144000','142400'),*SCHEDULED[4:]],
    [SCHEDULED[0],*SCHEDULED[1:-1],SCHEDULED[-1].replace('288000','286400')],
    [SCHEDULED[0],*SCHEDULED[1:-1],SCHEDULED[-1].replace('ceiling','feedback')],
    [SCHEDULED[0],SCHEDULED[1].replace('initial','feedback'),*SCHEDULED[2:]],
    [SCHEDULED[0],*SCHEDULED[1:3],SCHEDULED[3].replace('144000','144001'),*SCHEDULED[4:]],
])
def test_in_turn_probe_rejects_missing_or_inconsistent_transition(rows):
    with pytest.raises(ContractError):publisher_rate_probe(rows,1200000,schedule='down-up')
