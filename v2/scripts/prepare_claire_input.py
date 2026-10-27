"""Reproduce bounded 16 kHz input from an exact recorded synthetic Qwen request."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

import soundfile as sf

from talkinghead_v2.audio.resample import RationalResampler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wave',type=Path,required=True)
    parser.add_argument('--tts-report',type=Path,required=True)
    parser.add_argument('--case',required=True)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    if args.directory.exists():raise ValueError('preserve an existing input/evidence directory')
    if args.wave.is_symlink() or not args.wave.is_file() or args.wave.stat().st_size>4*1024**2:raise ValueError('bounded regular recorded WAV required')
    report=json.loads(args.tts_report.read_text())
    if report.get('voice_kind')!='synthetic_fixture':raise ValueError('this qualification helper only accepts declared synthetic-reference recordings')
    requests=[request for request in report['requests'] if request['id']==args.case]
    if len(requests)!=1:raise ValueError('exact declared request required')
    request=requests[0];samples,rate=sf.read(args.wave,dtype='float32')
    if samples.ndim!=1 or rate!=24000 or request['sample_rate']!=24000 or len(samples)!=request['samples'] or not 0<len(samples)<=24000*30:
        raise ValueError('recorded Qwen shape/rate/duration mismatch')
    pcm_hash=hashlib.sha256(samples.astype('<f4').tobytes()).hexdigest()
    if pcm_hash!=request['pcm_sha256']:raise ValueError('recorded PCM hash differs from actual Qwen request evidence')
    resampler=RationalResampler(24000,16000)
    # Exercise one persistent FIR history across non-divisor source quanta.
    values=[]
    for offset in range(0,len(samples),997):values.extend(resampler.push(tuple(float(value) for value in samples[offset:offset+997])))
    values.extend(resampler.finish())
    args.directory.mkdir(parents=True,mode=0o700)
    target=args.directory/'audio-16k.f32'
    with open(target,'xb',opener=lambda path,flags:os.open(path,flags,0o600)) as output:output.write(struct.pack(f'<{len(values)}f',*values))
    record={'source':'existing synthetic-reference Qwen Base 1.7B X-vector P02 '+args.case+' case',
        'source_rate':rate,'source_samples':len(samples),'source_pcm_f32_sha256':pcm_hash,
        'source_wave_sha256':hashlib.sha256(args.wave.read_bytes()).hexdigest(),
        'p02_report_sha256':hashlib.sha256(args.tts_report.read_bytes()).hexdigest(),
        'resampler':'stateful core RationalResampler radius32, exact EOF','target_rate':16000,'target_samples':len(values),
        'target_pcm_f32_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'human_recording':False,'new_tts_model_execution':False}
    with open(args.directory/'input.json','xb',opener=lambda path,flags:os.open(path,flags,0o600)) as output:
        output.write((json.dumps(record,indent=2)+'\n').encode())
    print(json.dumps({'input_prepared':True,'source_samples':len(samples),'target_samples':len(values),'model_execution':False}))


if __name__=='__main__':main()
