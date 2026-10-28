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
