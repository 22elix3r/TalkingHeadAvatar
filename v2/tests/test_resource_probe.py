import os
import pytest
from talkinghead_v2.contracts import ContractError


def parse(value):
    from talkinghead_v2.resource_probe import parse_gpu_sample
    return parse_gpu_sample(value)


def test_gpu_resource_sample_keeps_unknown_metrics_distinct_from_zero():
    sample=parse('0, 4000, 16376, 50, [Not Supported], 45, 1200, 95.5\n')
    assert sample['memory_used_mib']==4000 and sample['memory_total_mib']==16376
    assert sample['encoder_percent'] is None and sample['power_watts']==95.5


@pytest.mark.parametrize('value',[
    '', '0, 4000, 16376, 50, 0, 45, 1200',
    '0, 4000, 16376, 50, 0, 45, 1200, 95\n0, 4000, 16376, 50, 0, 45, 1200, 95',
    '1, 4000, 16376, 50, 0, 45, 1200, 95', '0, -1, 16376, 50, 0, 45, 1200, 95',
    '0, 17000, 16376, 50, 0, 45, 1200, 95', '0, 4000, 0, 50, 0, 45, 1200, 95',
    '0, N/A, 16376, 50, 0, 45, 1200, 95', '0, 4000, 16376, 101, 0, 45, 1200, 95',
    '0, 4000, 16376, 50, nan, 45, 1200, 95', '0, 4000, 16376, 50, 0, 45, 1200, inf',
])
def test_corrupt_or_ambiguous_gpu_observation_is_rejected(value):
    with pytest.raises(ContractError):parse(value)


def test_process_sample_is_real_and_rejects_reused_identity():
    from talkinghead_v2.resource_probe import process_sample
    row=process_sample(os.getpid())
    assert row and row['rss_bytes']>0 and row['fd_count']>=3 and row['threads']>=1
    assert process_sample(os.getpid(),expected_start=row['start_ticks']+1) is None
    assert process_sample(2**31-1) is None


def test_missing_gpu_attribution_does_not_count_as_observed_coload():
    from talkinghead_v2.resource_probe import native_coload
    assert not native_coload([{'role':'qwen','gpu_mib':4000},{'role':'media','gpu_mib':None}])
    assert not native_coload([{'role':'qwen','gpu_mib':None},{'role':'media','gpu_mib':800}])
    assert native_coload([{'role':'qwen','gpu_mib':4000},{'role':'media','gpu_mib':800}])
