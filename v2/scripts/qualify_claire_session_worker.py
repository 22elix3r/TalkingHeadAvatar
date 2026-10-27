"""Actual local SFU/Claire GPU sender and receiver checks, using cached audio."""
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import threading
import time

from livekit import rtc

from talkinghead_v2.audio.resample import RationalResampler
from talkinghead_v2.local import digest
from talkinghead_v2.contracts import AudioChunk,AudioEOF,Envelope,GenerationKey,TextSegment
from talkinghead_v2.providers.motion.paired import ClaireInputMux
from talkinghead_v2.providers.tts.native import QwenNativeProvider,load_native_config
from talkinghead_v2.server.config import Settings
from talkinghead_v2.server.native_media import private_write
from talkinghead_v2.server.transport import CheckedReceiveQueue,grant
from talkinghead_v2.session.device import DeviceLease

ROOT=Path(__file__).resolve().parents[1]


async def main(args):
    if args.directory.exists():raise ValueError('preserve previous worker evidence')
    bridge=json.loads((ROOT/'benchmarks/reports/p03-claire-bridge.json').read_text())
    build=ROOT/'var/build/native-media';library=build/'libtha_claire_bridge.so';binary=build/'tha_claire_session_media';ptx=build/'media_kernels.ptx'
    if digest(library)!=bridge['library_sha256'] or digest(ptx)!=bridge['ptx_sha256']:raise ValueError('qualify the current bridge before session integration')
    engine=json.loads((args.engine/'engine.json').read_text());scene=json.loads((args.scene/'scene.json').read_text())
    if (digest(args.engine/'network.trt')!=engine['engine_sha256'] or engine['engine_sha256']!=bridge['engine_sha256'] or
        digest(args.engine/'model.json')!=engine['model_descriptor_sha256'] or digest(args.scene/'research-scene.bin')!=scene['geometry_sha256'] or
        scene['geometry_sha256']!=bridge['scene_sha256']):raise ValueError('qualified engine/scene provenance changed')
    source=json.loads((ROOT/'var/p03/claire-qwen-input/input.json').read_text())
    wave=ROOT/'var/qualification/native-waves-serial/native-000.wav'
    if digest(wave)!=source['source_wave_sha256']:raise ValueError('cached synthetic Qwen waveform changed')
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(wave),'-f','f32le','-acodec','pcm_f32le','-'],timeout=10)
    if hashlib.sha256(raw).hexdigest()!=source['source_pcm_f32_sha256']:raise ValueError('source PCM differs from Qwen evidence')
    pcm=struct.unpack(f'<{len(raw)//4}f',raw);audio_rs=RationalResampler(24000,48000);motion_rs=RationalResampler(24000,16000)
    audio=audio_rs.push(pcm)+audio_rs.finish();motion=motion_rs.push(pcm)+motion_rs.finish()
    if hashlib.sha256(struct.pack(f'<{len(motion)}f',*motion)).hexdigest()!=bridge['source_pcm_sha256']:raise ValueError('motion resampler/reference parity changed')
    audio=tuple(int(max(-1.,min(1.,value))*32767) for value in audio)
    settings=Settings.load(args.config);runtime=ROOT/'var/toolchains/tensorrt-10.13.3.9'
    if digest(runtime/'runtime.json')!=engine['runtime_manifest_sha256']:raise ValueError('runtime descriptor changed')
    for entry in json.loads((runtime/'runtime.json').read_text())['files']:
        if digest(runtime/entry['path'])!=entry['sha256']:raise ValueError('runtime file changed')
    native=await asyncio.to_thread(load_native_config,settings.tts_runtime) if args.live_qwen else None
    if native and (native.voice_kind!='synthetic_fixture' or native.device_lease_path!=ROOT/'var/gpu-device-0.lock'):raise ValueError('live qualification requires the pinned synthetic voice and shared device lease')
    args.directory.mkdir(parents=True,mode=0o700);lease=DeviceLease(ROOT/'var/gpu-device-0.lock');lease.acquire('claire-session-worker-qualification')
    cases=[];bindings=[];memory=[];query_failures=[];sample_stop=threading.Event();qwen_pids=set();worker_pids=set()
    owned_tts=[];owned_rooms=[];owned_processes=[]
    def sample_memory():
        while not sample_stop.is_set():
            try:
                device=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=3).strip().split(',')
                rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=3)
                processes=[{'pid':int(line.split(',')[0]),'mib':int(line.split(',')[1])} for line in rows.splitlines()]
                memory.append({'monotonic_ns':time.monotonic_ns(),'device_used_mib':int(device[0]),'device_total_mib':int(device[1]),'processes':processes})
            except (OSError,ValueError,subprocess.SubprocessError):query_failures.append(time.monotonic_ns())
            sample_stop.wait(.1)
    sampling=threading.Thread(target=sample_memory);sampling.start()
    try:
        selected_cases=[('qwen_live',None,None)] if args.live_qwen else [('qwen_recorded',len(audio),None),('one',1,None),('short',479,None),('aligned',480,None),
            ('partial',481,None),('quantum',960,None),('tail',961,None),('empty',0,'empty'),('wrong_epoch',960,'epoch'),
            ('gap',960,'gap'),('sequence',960,'sequence'),('wrong_motion',960,'motion'),('backpressure_cancel',960,'cancel')]
        for name,count,fault in selected_cases:
            directory=args.directory/name;directory.mkdir(mode=0o700);sid='claire-worker-'+name;room_name='claire-worker-'+name
            tts=None;live_metrics=None
            receive_room=None;tasks=[];streams=[];receiver={'audio_frames':0,'audio_samples':0,'maximum_rms':0.,'video':[],'idle_frames':0};subscribed=asyncio.Event();kinds=set()
            if name.startswith('qwen_'):
                receive_room=rtc.Room();owned_rooms.append(receive_room)
                @receive_room.on('track_subscribed')
                def subscribed_track(track,publication,participant):
                    async def consume():
                        video=track.kind==rtc.TrackKind.KIND_VIDEO
                        stream=rtc.VideoStream(track) if video else rtc.AudioStream(track,sample_rate=48000,num_channels=1,frame_size_ms=10)
                        if type(stream._queue).__name__!='RingQueue':raise ValueError('unqualified receiver queue implementation')
                        stream._queue=CheckedReceiveQueue(30);streams.append(stream)
                        kinds.add('video' if video else 'audio')
                        if len(kinds)==2:subscribed.set()
                        async for item in stream:
                            if video:
                                if item.metadata is None:raise ValueError('missing received generation metadata')
                                identity=json.loads(item.metadata.user_data)
                                if identity['session_id']!=sid or identity['turn_id']!='test' or identity['epoch']!=7:raise ValueError('stale received epoch')
                                if identity['source_sample']%1600 or not 0<=identity['source_sample']-identity['neural_sample_16k']*3<=2:raise ValueError('received rational source mapping changed')
                                if identity.get('idle_after_eof'):
                                    receiver['idle_frames']+=1
                                    continue
                                receiver['video'].append({'source_sample':identity['source_sample'],'source_pts_us':item.metadata.user_timestamp,
                                    'decoded_sha256':hashlib.sha256(item.frame.data).hexdigest(),'width':item.frame.width,'height':item.frame.height})
                            else:
                                values=item.frame.data;receiver['audio_frames']+=1;receiver['audio_samples']+=len(values)
                                rms=math.sqrt(sum(float(value)**2 for value in values)/max(1,len(values)));receiver['maximum_rms']=max(receiver['maximum_rms'],rms)
                    tasks.append(asyncio.create_task(consume()))
                await receive_room.connect(settings.livekit_url,grant(settings,room_name,'observer-'+sid,'controller'))
            if args.live_qwen:
                tts=QwenNativeProvider(native,diagnostic_log=directory/'qwen-stderr.log',device_lease=lease)
                owned_tts.append(tts)
                await tts.open_session(GenerationKey(sid,'test',7));qwen_pids.add(tts._process.pid)
            private_write(directory/'bootstrap.txt',('\n'.join(('THA_CLAIRE_MEDIA_1',settings.livekit_url,
                grant(settings,room_name,'media-'+sid,'media'),sid,'test','7','300'))+'\n').encode())
            env=dict(os.environ,THA_DEVICE_LEASE_FD=str(lease.fileno),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',
                LD_LIBRARY_PATH=str(runtime/'lib')+':/usr/local/cuda-12.8/lib64',CUDA_CACHE_PATH=str(ROOT/'var/cache/claire-cuda'))
            log=open(directory/'stderr.log','xb',opener=lambda path,flags:os.open(path,flags,0o600))
            process=await asyncio.create_subprocess_exec(str(binary),str(directory/'bootstrap.txt'),str(ptx),str((args.scene/'research-scene.bin').resolve()),
                str((args.engine/'model.json').resolve()),env=env,stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=log,pass_fds=(lease.fileno,),limit=16384)
            owned_processes.append(process);worker_pids.add(process.pid)
            log.close();producer=None
            async def event():
                line=await asyncio.wait_for(process.stdout.readline(),25)
                if not line:raise ValueError('worker exited without event in '+name)
                return json.loads(line)
            def packet(index,start,a,m,final=False):
                return struct.pack('<cQQQII',b'F' if final else b'D',6 if fault=='epoch' else 7,index+int(fault=='sequence'),start+int(fault=='gap'),len(a),len(m))+struct.pack(f'<{len(a)}h',*a)+struct.pack(f'<{len(m)}f',*m)
            try:
                ready=await event()
                if ready.get('kind')!='ready' or ready.get('epoch')!=7:raise ValueError('Claire worker not ready')
                bindings.append((ready['audio_sid'],ready['video_sid']))
                if receive_room:await asyncio.wait_for(subscribed.wait(),10)
                if fault=='cancel':
                    async def flood():
                        for index in range(1200):
                            process.stdin.write(packet(index,index*960,audio[:960],motion[:320]));await process.stdin.drain()
                    process.stdin.transport.set_write_buffer_limits(high=8192,low=4096);producer=asyncio.create_task(flood());await asyncio.sleep(.2)
                    if producer.done():raise ValueError('cancel producer never reached backpressure')
                    began=time.monotonic();process.send_signal(signal.SIGTERM);await asyncio.wait_for(process.wait(),5)
                    if process.returncode:raise ValueError('priority cancellation did not retire worker cleanly')
                    cases.append({'name':name,'blocked_producer':True,'owned_process_exit_ms':(time.monotonic()-began)*1000,'receiver_flush_claimed':False})
                else:
                    selected_audio=audio[:count];selected_motion=motion[:(count+2)//3] if count is not None else ();offset=index=0
                    if args.live_qwen:
                        mux=ClaireInputMux(GenerationKey(sid,'test',7));started=time.perf_counter_ns();first_audio=first_packet=None;early_packets=0
                        text='This is live speech from the cached Qwen voice. The same audio drives Claire while you hear the answer. We are checking the clock, the final sound, and the moving face together.'
                        producer=asyncio.create_task(tts.push(TextSegment(Envelope(mux.key,0,'live-claire'),'live',text,'en',native.voice_id,is_last=True)))
                        async for source_event in tts.outputs():
                            if isinstance(source_event,AudioChunk) and first_audio is None:first_audio=time.perf_counter_ns()-started
                            for paired in await asyncio.to_thread(mux.accept,source_event):
                                if first_packet is None:first_packet=time.perf_counter_ns()-started
                                if not producer.done():early_packets+=1
                                process.stdin.write(paired.packed());await process.stdin.drain()
                        if not await producer or not mux.ledger.closed or early_packets==0:raise ValueError('live incremental Qwen/EOF qualification failed')
                        count=mux.sent_samples;offset=count
                        live_metrics={'voice_kind':native.voice_kind,'model_revision':native.model_revision,'runtime_revision':native.runtime_revision,
                            'source_samples_24k':mux.ledger.next_sample,'motion_samples_16k':mux.sent_motion_samples,
                            'first_audio_received_ns':first_audio,'first_paired_packet_ns':first_packet,'packets_before_tts_complete':early_packets,
                            'controller_clock':'python_perf_counter','provider':tts.last_metrics,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                            'wall_includes_downstream_backpressure':True}
                    elif name=='qwen_recorded':
                        mux=ClaireInputMux(GenerationKey(sid,'test',7));source_offset=source_sequence=0
                        while source_offset<len(pcm):
                            values=pcm[source_offset:source_offset+997]
                            source_event=AudioChunk(Envelope(mux.key,source_sequence,'recorded-qwen'),'fixture',24000,source_offset,values)
                            for paired in await asyncio.to_thread(mux.accept,source_event):
                                process.stdin.write(paired.packed());await process.stdin.drain()
                            source_offset+=len(values);source_sequence+=1
                        for paired in await asyncio.to_thread(mux.accept,AudioEOF(Envelope(mux.key,source_sequence,'recorded-qwen'),24000,source_offset)):
                            process.stdin.write(paired.packed());await process.stdin.drain()
                        offset=count
                    while count-offset>=960:
                        data=packet(index,offset,selected_audio[offset:offset+960],selected_motion[offset//3:offset//3+320])
                        if fault=='motion':data=packet(index,offset,selected_audio[offset:offset+960],selected_motion[:319])
                        process.stdin.write(data);await process.stdin.drain();offset+=960;index+=1
                        if fault:break
                    if not name.startswith('qwen_') and (not fault or fault=='empty'):
                        process.stdin.write(packet(index,offset,selected_audio[offset:],selected_motion[offset//3:],True));await process.stdin.drain()
                    result=await event()
                    if fault:
                        if result!={'kind':'error','code':'claire_media_failed'}:raise ValueError('malformed Claire protocol accepted')
                        await asyncio.wait_for(process.wait(),5)
                        if process.returncode!=1:raise ValueError('protocol fault did not retire worker')
                        cases.append({'name':name,'rejected':True,'process_exit':1})
                    else:
                        if (result.get('kind')!='eof' or result.get('valid_samples')!=count or result.get('padding_samples')!=(-count)%480+960 or
                            result.get('presentation_ack') is not False or result.get('pose_queue_peak_frames',33)>32 or result.get('motion_queue_peak_packets',51)>50 or
                            result.get('source_video_frames',0)+result.get('video_skipped',0)!=(count+1599)//1600):raise ValueError('Claire sender EOF/queue accounting failed in '+name)
                        record={'name':name,'sender':{key:value for key,value in result.items() if key not in {'session_id','turn_id'}},'process_exit':0}
                        if live_metrics:record['live_qwen']=live_metrics
                        if receive_room:
                            await asyncio.sleep(.4)
                            for task in tasks:
                                if task.done():task.result()
                            rows=receiver['video'];positions=[row['source_sample'] for row in rows]
                            if (len(rows)<50 or len(set(row['decoded_sha256'] for row in rows))<10 or receiver['maximum_rms']<100 or
                                positions!=sorted(set(positions)) or positions[-1]!=(count-1)//1600*1600 or receiver['idle_frames']<1 or any(row['width']!=512 or row['height']!=512 for row in rows)):
                                raise ValueError('actual receiver did not decode moving Claire/audio')
                            receiver['missing_source_frame_indices']=[index for index in range((count+1599)//1600) if index*1600 not in positions]
                            record['receiver']=receiver
                        cases.append(record);process.send_signal(signal.SIGTERM);await asyncio.wait_for(process.wait(),5)
                        if process.returncode:raise ValueError('post-EOF owner retirement failed')
                private_write(directory/'result.json',(json.dumps(cases[-1],indent=2)+'\n').encode())
            finally:
                if process.returncode is None:process.kill();await process.wait()
                if producer:producer.cancel();await asyncio.gather(producer,return_exceptions=True)
                if tts:await tts.close()
                process.stdin.close()
                try:await process.stdin.wait_closed()
                except (BrokenPipeError,ConnectionResetError):pass
                for stream in streams:await stream.aclose()
                for task in tasks:task.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
                if receive_room:await receive_room.disconnect()
                (directory/'bootstrap.txt').unlink()
    finally:
        for child in owned_processes:
            if child.returncode is None:child.kill();await child.wait()
        for provider in owned_tts:await provider.close()
        for room in owned_rooms:await room.disconnect()
        sample_stop.set();sampling.join(timeout=10);lease.release()
        private_write(args.directory/'memory.json',(json.dumps({'samples':memory,'query_failures':query_failures},indent=2)+'\n').encode())
    lease.acquire('claire-session-post-exit');lease.release()
    if len(set(bindings))!=len(bindings):raise ValueError('publication IDs reused across retired epochs')
    sources=['native/media/claire_session_worker.cpp','native/motion/claire_bridge.cpp','native/motion/claire_bridge.h','native/media/CMakeLists.txt',
        'src/talkinghead_v2/providers/motion/paired.py','scripts/qualify_claire_session_worker.py']
    report={'schema_version':1,'observation':'local actual Claire SDK/GPU/NVENC/LiveKit sender and Python receiver; '+('live cached Qwen' if args.live_qwen else 'recorded synthetic-reference Qwen PCM'),
        'source_sha256':{path:digest(ROOT/path) for path in sources},'binary_sha256':digest(binary),'bridge_sha256':digest(library),'ptx_sha256':digest(ptx),
        'bridge_report_sha256':digest(ROOT/'benchmarks/reports/p03-claire-bridge.json'),'engine_sha256':engine['engine_sha256'],'scene_sha256':scene['geometry_sha256'],
        'source_wave_sha256':digest(wave),'source_pcm_24k_sha256':hashlib.sha256(raw).hexdigest(),'cases':cases,'fresh_publication_ids':True,
        'admission_reacquired':True,'audio_queue_bound_samples':48000,'host_motion_queue_bound_packets':50,'motion_input_lead_bound_ms':500,
        'live_qwen_streaming':args.live_qwen,'browser_receiver':False,'physical_av_qualified':False,'qwen_coload_qualified':False,'quality_qualified':False,'additional_downloads':False}
    def usage(sample,pids):return sum(process['mib'] for process in sample['processes'] if process['pid'] in pids)
    coload=[sample for sample in memory if usage(sample,qwen_pids)>0 and usage(sample,worker_pids)>0]
    if args.live_qwen and not coload:raise ValueError('no actual Qwen/Claire co-residency memory observation')
    report['memory']={'sample_count':len(memory),'query_failures':len(query_failures),'target_interval_ms':100,'queries_sequential':True,
        'sampled_maxima_are_not_guaranteed_peaks':True,'sampled_device_used_max_mib':max((sample['device_used_mib'] for sample in memory),default=None),
        'device_total_mib':memory[0]['device_total_mib'] if memory else None,'coload_sample_count':len(coload),
        'qwen_process_sampled_max_mib':max((usage(sample,qwen_pids) for sample in memory),default=0),
        'claire_media_process_sampled_max_mib':max((usage(sample,worker_pids) for sample in memory),default=0),
        'receiver_process_sampled_max_mib':max((usage(sample,{os.getpid()}) for sample in memory),default=0),
        'qwen_and_claire_same_query_sampled_max_mib':max((usage(sample,qwen_pids|worker_pids) for sample in coload),default=None),
        'capacity_or_soak_qualified':False}
    if args.live_qwen:
        report['recorded_preflight_fixture_wave_sha256']=report.pop('source_wave_sha256')
        report['recorded_preflight_fixture_pcm_sha256']=report.pop('source_pcm_24k_sha256')
        report['tts_runtime_descriptor_sha256']=digest(settings.tts_runtime)
        report['source_sha256']['src/talkinghead_v2/providers/tts/native.py']=digest(ROOT/'src/talkinghead_v2/providers/tts/native.py')
    args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'worker_cases':len(cases),'receiver_video_frames':len(cases[0]['receiver']['video']),'physical_av':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live-qwen',action='store_true',help='execute one pinned cached-Qwen/SFU/Claire co-load case instead of recorded/fault cohort')
    for name in ('config','engine','scene','directory','report'):parser.add_argument('--'+name,type=Path,required=True)
    asyncio.run(main(parser.parse_args()))
