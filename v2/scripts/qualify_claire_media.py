"""Actual pinned Python Claire wrapper, EOF and blocked-producer retirement."""
import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time

from talkinghead_v2.contracts import AudioChunk,AudioEOF,Envelope,GenerationKey
from talkinghead_v2.local import digest
from talkinghead_v2.server.claire_media import ClaireMedia,ClaireMediaConfig
from talkinghead_v2.server.config import Settings
from talkinghead_v2.session.device import DeviceLease

ROOT=Path(__file__).resolve().parents[1]


async def main(args):
    if args.directory.exists():raise ValueError('preserve previous wrapper evidence')
    args.directory.mkdir(parents=True,mode=0o700)
    config=await asyncio.to_thread(ClaireMediaConfig.load,args.runtime)
    settings=replace(Settings.load(args.config),native_media_runtime=args.runtime,private_dir=args.directory)
    wave=ROOT/'var/qualification/native-waves-serial/native-000.wav';known=json.loads((ROOT/'var/p03/claire-qwen-input/input.json').read_text())
    if digest(wave)!=known['source_wave_sha256']:raise ValueError('synthetic source identity changed')
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(wave),'-f','f32le','-acodec','pcm_f32le','-'],timeout=10)
    if hashlib.sha256(raw).hexdigest()!=known['source_pcm_f32_sha256']:raise ValueError('synthetic source PCM changed')
    samples=struct.unpack(f'<{len(raw)//4}f',raw)
    lease=DeviceLease(ROOT/'var/gpu-device-0.lock');lease.acquire('claire-python-media-wrapper')
    media=ClaireMedia(settings,'claire-wrapper','claire-wrapper',lease,expected_revision=config.revision)
    cases=[];bindings=[];producer=None
    try:
        await media.connect()
        for epoch in (1,2,3):
            key=GenerationKey('claire-wrapper','turn-'+str(epoch),epoch)
            binding=await media.start(key);bindings.append(binding)
            process=media.state['process'];directory=media.state['directory']
            if binding['visual_mode']!='claire_vendor_research' or binding['clock_mapping']!='new_track_per_epoch':raise ValueError('wrong vendor/epoch binding')
            if epoch==2:
                async def flood():
                    for index in range(1000):
                        if not await media.submit(AudioChunk(Envelope(key,index,'synthetic-flood'),'flood',24000,index*2400,(.1,)*2400)):return
                producer=asyncio.create_task(flood());deadline=time.monotonic()+8
                while process.stdin.transport.get_write_buffer_size()<8192:
                    if producer.done():producer.result();raise ValueError('wrapper did not reach actual input backpressure')
                    if time.monotonic()>deadline:raise ValueError('wrapper backpressure not observed within bound')
                    await asyncio.sleep(.01)
                began=time.monotonic();await media.cancel();elapsed=(time.monotonic()-began)*1000
                producer.cancel();await asyncio.gather(producer,return_exceptions=True);producer=None
                if process.returncode!=0 or directory.exists() or media.state is not None:raise ValueError('owned epoch was not fully retired/purged')
                cases.append({'epoch':epoch,'blocked_producer':True,'owned_process_exit_ms':elapsed,'private_epoch_purged':True,'receiver_flush_claimed':False})
            else:
                offset=sequence=0
                while offset<len(samples):
                    values=samples[offset:offset+997]
                    if not await media.submit(AudioChunk(Envelope(key,sequence,'recorded-qwen'),'fixture',24000,offset,values)):raise ValueError('active wrapper rejected valid PCM')
                    offset+=len(values);sequence+=1
                if not await media.submit(AudioEOF(Envelope(key,sequence,'recorded-qwen'),24000,offset)):raise ValueError('active wrapper rejected exact EOF')
                if media.cursor!=len(samples)*2 or media.padding!=(-media.cursor)%480+960:raise ValueError('wrapper lost source audio/tail')
                cases.append({'epoch':epoch,'valid_samples':media.cursor,'padding_samples':media.padding,'sender_metrics':media.sender_metrics,'presentation_ack':False})
                await media.cancel()
                if process.returncode!=0 or directory.exists():raise ValueError('completed wrapper epoch not fully retired/purged')
    finally:
        if producer:producer.cancel();await asyncio.gather(producer,return_exceptions=True)
        await media.close();lease.release()
    lease.acquire('claire-wrapper-post-exit');lease.release()
    if len({(binding['audio_sid'],binding['video_sid']) for binding in bindings})!=3:raise ValueError('publication pair reused after cancellation')
    paths=['src/talkinghead_v2/server/claire_media.py','src/talkinghead_v2/server/native_media.py',
        'src/talkinghead_v2/providers/motion/paired.py','scripts/describe_claire_media.py','scripts/qualify_claire_media.py']
    report={'schema_version':1,'observation':'actual Python pinned Claire process wrapper / local SFU sender, recorded synthetic Qwen',
        'source_sha256':{path:digest(ROOT/path) for path in paths},'runtime_descriptor_sha256':config.revision,
        'pinned_file_count':len(config.paths),'native_executable_sha256':digest(config.executable),'cases':cases,
        'fresh_publications_after_cancel':True,'post_retirement_admission_reacquired':True,
        'browser_receiver':False,'physical_av_qualified':False,'additional_downloads':False}
    args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'wrapper_epochs':3,'fresh_after_blocked_cancel':True,'models_downloaded':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','runtime','directory','report'):parser.add_argument('--'+name,type=Path,required=True)
    asyncio.run(main(parser.parse_args()))
