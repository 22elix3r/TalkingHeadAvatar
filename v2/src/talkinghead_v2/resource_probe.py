"""Read-only Linux/driver observations; sampled maxima are not guaranteed peaks."""
import csv
import math
import os
from pathlib import Path
from talkinghead_v2.contracts import ContractError


def parse_gpu_sample(value):
    rows=list(csv.reader(value.strip().splitlines(),skipinitialspace=True))
    if len(rows)!=1 or len(rows[0])!=8:raise ContractError('one explicit GPU resource row required')
    fields=rows[0]
    try:
        index,used,total=map(int,fields[:3])
        if index!=0 or not 0<=used<=total or total<=0:raise ValueError
        result={'index':index,'memory_used_mib':used,'memory_total_mib':total}
        for name,text,limit in zip(('gpu_percent','encoder_percent','temperature_celsius','graphics_mhz','power_watts'),fields[3:],(100,100,140,100000,3000)):
            if text.strip() in ('N/A','[N/A]','[Not Supported]'):result[name]=None;continue
            number=float(text)
            if not math.isfinite(number) or not 0<=number<=limit:raise ValueError
            result[name]=number
    except ValueError:raise ContractError('invalid GPU resource observation') from None
    return result


def process_sample(pid,expected_start=None):
    if type(pid) is not int or not 0<pid<2**31:return None
    base=Path('/proc')/str(pid)
    try:
        before=(base/'stat').read_text().rsplit(')',1)[1].split()
        if before[0] in ('Z','X') or base.stat().st_uid!=os.getuid():return None
        start=int(before[19])
        if expected_start is not None and start!=expected_start:return None
        with os.scandir(base/'fd') as files:count=sum(1 for _ in files)
        after=(base/'stat').read_text().rsplit(')',1)[1].split()
        if after[0] in ('Z','X') or int(after[19])!=start:return None
        return {'start_ticks':start,'rss_bytes':max(0,int(after[21]))*os.sysconf('SC_PAGE_SIZE'),
            'threads':int(after[17]),'fd_count':count,'cpu_ticks':int(after[11])+int(after[12])}
    except (OSError,ValueError,IndexError):return None


def native_coload(processes):
    return {'qwen','media'}<={p['role'] for p in processes if (p.get('gpu_mib') or 0)>0}
