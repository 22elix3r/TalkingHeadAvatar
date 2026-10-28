"""Actual local SFU/Claire GPU sender and receiver checks, using cached audio."""
import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import threading
import time

from livekit import rtc
from livekit.rtc._ffi_client import FfiClient
from livekit.rtc._proto import ffi_pb2,video_frame_pb2,audio_frame_pb2
from livekit.rtc.log import logger as livekit_logger

from talkinghead_v2.audio.resample import RationalResampler
from talkinghead_v2.local import digest
from talkinghead_v2.contracts import AudioChunk,AudioEOF,Envelope,GenerationKey,TextSegment
from talkinghead_v2.providers.motion.paired import ClaireInputMux
from talkinghead_v2.providers.tts.native import QwenNativeProvider,load_native_config
from talkinghead_v2.server.config import Settings
from talkinghead_v2.server.native_media import private_write
from talkinghead_v2.server.transport import CheckedReceiveQueue,grant
from talkinghead_v2.session.device import DeviceLease
from talkinghead_v2.publish_probe import PublisherTraceConfig,publish_stage_trace

ROOT=Path(__file__).resolve().parents[1]


class QualifiedAudioStream(rtc.AudioStream):
    """Bound the source-matched native receiver without changing PCM cadence."""
    def _create_owned_stream(self):
        if self._track is None or self._audio_filter_module is not None or self._processor is not None:
            raise ValueError('unfiltered explicit receiver track required')
        request=ffi_pb2.FfiRequest();stream=request.new_audio_stream
        stream.track_handle=self._track._ffi_handle.handle
        stream.sample_rate=self._sample_rate;stream.num_channels=self._num_channels
        stream.frame_size_ms=self._frame_size_ms;stream.queue_size_frames=25
        stream.type=audio_frame_pb2.AudioStreamType.AUDIO_STREAM_NATIVE
        return FfiClient.instance.request(request).new_audio_stream.stream


class FfiVideoBoundaryTrace:
    """Qualification-only metadata at the FFI callback, before Python dispatch."""
    def __init__(self,queue,stall_ms=0):
        if type(stall_ms) is not int or stall_ms not in (0,500):raise ValueError('unsupported callback-stall fixture')
        self.queue=queue;self.original=queue.put;self.lock=threading.Lock();self.streams={}
        self.stall_ms=stall_ms;self.stalled=False;self.actual_stall_ns=None
        self.queue.put=self.put
    def register(self,handle,session):
        with self.lock:
            if handle in self.streams or len(self.streams)>=20:raise ValueError('receiver trace stream inventory exceeded')
            self.streams[handle]={'session':session,'rows':[],'preroll':0,'idle':0,'error':False}
    def put(self,event):
        stall=False
        try:
            if event.WhichOneof('message')=='video_stream_event' and event.video_stream_event.HasField('frame_received'):
                video=event.video_stream_event
                with self.lock:
                    record=self.streams.get(video.stream_handle)
                    if record is not None:
                        try:
                            frame=video.frame_received
                            if not frame.HasField('metadata') or len(frame.metadata.user_data)>2048:raise ValueError('metadata bound')
                            identity=json.loads(frame.metadata.user_data)
                            if (identity['session_id']!=record['session'] or identity['turn_id']!='test' or identity['epoch']!=7):raise ValueError('generation mismatch')
                            if identity.get('idle_before_audio'):record['preroll']+=1
                            elif identity.get('idle_after_eof'):record['idle']+=1
                            else:
                                sample=identity['source_sample'];fid=frame.metadata.frame_id
                                if (type(sample) is not int or sample<0 or sample%1600 or not frame.metadata.HasField('frame_id') or
                                    len(record['rows'])>=3600 or (record['rows'] and sample<=record['rows'][-1]['source_sample'])):raise ValueError('source identity bound/order')
                                record['rows'].append({'source_sample':sample,'frame_id':fid})
                                if self.stall_ms and not self.stalled:self.stalled=True;stall=True
                        except (ValueError,TypeError,KeyError):record['error']=True
            if stall:
                began=time.monotonic_ns();time.sleep(self.stall_ms/1000)
                with self.lock:self.actual_stall_ns=time.monotonic_ns()-began
        finally:
            # Observation failure must never suppress or replace an SDK event.
            self.original(event)
    def snapshot(self,handle):
        with self.lock:
            row=self.streams[handle]
            if row['error']:raise ValueError('invalid bounded FFI video trace; private metadata excluded')
            return {'source_frames':deepcopy(row['rows']),'preroll_frames':row['preroll'],'idle_frames':row['idle'],
                    'injected_callback_stall_ns':self.actual_stall_ns}
    def close(self):self.queue.put=self.original


class QualifiedVideoStream(rtc.VideoStream):
    """Set the pinned SDK's native queue, independently of its Python queue."""
    def __init__(self,track,native_queue_frames):
        if type(native_queue_frames) is not int or native_queue_frames not in (1,8):
            raise ValueError('qualified native video queue requires 1 or 8 frames')
        self._native_queue_frames=native_queue_frames
        super().__init__(track)

    def _create_owned_stream(self):
        if self._track is None:raise ValueError('track required')
        request=ffi_pb2.FfiRequest();stream=request.new_video_stream
        if 'queue_size_frames' not in stream.DESCRIPTOR.fields_by_name:
            raise ValueError('pinned SDK native video queue option unavailable')
        stream.track_handle=self._track._ffi_handle.handle
        stream.type=video_frame_pb2.VideoStreamType.VIDEO_STREAM_NATIVE
        if self._format is not None:stream.format=self._format
        stream.normalize_stride=True;stream.queue_size_frames=self._native_queue_frames
        return FfiClient.instance.request(request).new_video_stream.stream


class NativeVideoDropWarnings(logging.Handler):
    """Retain only the SDK's periodic drop count, never arbitrary log text."""
    pattern=re.compile(r'native video stream queue overflow; dropped (\d{1,10}) queued frames')
    def __init__(self):
        super().__init__();self.events=0;self.reported_lower_bound=0
    def emit(self,record):
        match=self.pattern.search(record.getMessage())
        if match:
            self.events+=1;self.reported_lower_bound=max(self.reported_lower_bound,int(match.group(1)))
    def snapshot(self):
        self.acquire()
        try:return self.events,self.reported_lower_bound
        finally:self.release()


def receiver_video_stats(stats):
    """Export only bounded counters, never track, transport or candidate IDs."""
    rows=[]
    for stat in stats:
        if stat.WhichOneof('stats')!='inbound_rtp' or stat.inbound_rtp.stream.kind!='video':continue
        inbound=stat.inbound_rtp.inbound;received=stat.inbound_rtp.received
        # Only known implementation names become public labels. Arbitrary
        # implementation strings, like transport IDs, stay out of evidence.
        implementation=inbound.decoder_implementation.strip().lower()
        family={'nvidia h264 decoder':'nvidia_h264','nvidia h265 decoder':'nvidia_h265',
                'ffmpeg':'ffmpeg','ffmpeg h264 decoder':'ffmpeg','openh264':'openh264'}.get(implementation,'other_or_unreported')
        rows.append({'frames_received':inbound.frames_received,'frames_decoded':inbound.frames_decoded,
            'frames_dropped':inbound.frames_dropped,'packets_received':received.packets_received,
            'packets_lost':received.packets_lost,'packets_discarded':inbound.packets_discarded,
            'retransmitted_packets_received':inbound.retransmitted_packets_received,
            'nack_count':inbound.nack_count,'pli_count':inbound.pli_count,'fir_count':inbound.fir_count,
            'decoder_family':family,'power_efficient_decoder':inbound.power_efficient_decoder})
    return rows


def source_capture_trace(lines,expected):
    """Parse only the worker's bounded, non-biometric capture observations."""
    if type(expected) is not int or not 0<expected<=3600:raise ValueError('invalid source capture trace count')
    rows=[]
    for line in lines:
        if not line.startswith('THA_CAPTURE_TRACE '):continue
        fields=line.split()
        if len(fields)!=8:raise ValueError('invalid source capture trace')
        try:frame,source,pts,start,end,size,key=map(int,fields[1:])
        except ValueError:raise ValueError('invalid source capture trace') from None
        if (frame<0 or source<0 or source%1600 or pts<0 or start<0 or end<start or
            not 1<=size<=64*1024*1024 or key not in (0,1)):
            raise ValueError('invalid source capture trace bounds')
        if rows and (frame<=rows[-1]['frame_id'] or source<=rows[-1]['source_sample']):
            raise ValueError('unordered source capture trace')
        rows.append({'frame_id':frame,'source_sample':source,'source_pts_us':pts,'capture_start_us':start,
                     'capture_end_us':end,'encoded_bytes':size,'keyframe':bool(key)})
    if len(rows)!=expected or len(rows)>3600:raise ValueError('incomplete source capture trace')
    return rows


def read_capture_log(path):
    with open(path,'rb',opener=lambda name,flags:os.open(name,flags|os.O_NOFOLLOW)) as source:
        data=source.read(4*1024*1024+1)
    if len(data)>4*1024*1024:raise ValueError('source capture log exceeded bound')
    return data.decode('utf-8',errors='replace').splitlines()


def read_source_capture_trace(path,expected):
    return source_capture_trace(read_capture_log(path),expected)


def sender_rtp_stats(lines,tag='THA_SENDER_RTP'):
    if tag not in {'THA_SENDER_RTP','THA_SENDER_RTP_AFTER_300MS'}:raise ValueError('invalid sender RTP stats tag')
    rows=[line.split() for line in lines if line.startswith(tag+' ')]
    if not rows:return None
    if len(rows)!=1 or len(rows[0])!=10:raise ValueError('invalid sender RTP stats')
    try:values=list(map(int,rows[0][1:]))
    except ValueError:raise ValueError('invalid sender RTP stats') from None
    if any(value<0 for value in values):raise ValueError('invalid sender RTP stats')
    return dict(zip(('video_frames_accepted_total','frames_sent','frames_encoded','packets_sent',
                     'retransmitted_packets_sent','nack_count','pli_count','fir_count','key_frames_encoded'),values))


async def main(args):
    if args.trace_source_captures and not args.live_qwen:raise ValueError('source capture tracing requires the live-Qwen case')
    if args.directory.exists() or args.report.exists():raise ValueError('preserve previous worker evidence')
    selected=getattr(args,'selected_receiver',None)
    publisher=PublisherTraceConfig.load(args.publisher_trace_runtime) if args.publisher_trace_runtime else None
    if args.trust_encoded_rate_control and (publisher is None or not publisher.supports_external_rate_control):
        raise ValueError('external-rate-control selection requires an explicit compatible publisher probe')
    if publisher and (not args.trace_source_captures or not args.live_qwen):raise ValueError('publisher stage tracing requires the explicit live source trace')
    if args.trace_receiver_ffi and selected is None:raise ValueError('FFI tracing requires an explicit source-matched receiver runtime')
    if args.stall_video_ffi_ms and not args.trace_receiver_ffi:raise ValueError('callback stall requires the explicit FFI trace')
    bridge=json.loads((ROOT/'benchmarks/reports/p03-claire-bridge.json').read_text())
    build=ROOT/'var/build/native-media';library=build/'libtha_claire_bridge.so';binary=build/'tha_claire_session_media';ptx=build/'media_kernels.ptx'
    if publisher:binary=publisher.executable
    sources=['native/media/claire_session_worker.cpp','native/motion/claire_bridge.cpp','native/motion/claire_bridge.h','native/media/CMakeLists.txt',
        'src/talkinghead_v2/providers/motion/paired.py','scripts/qualify_claire_session_worker.py',
        'native/media/publish_trace.h','src/talkinghead_v2/publish_probe.py']
    if publisher:
        from talkinghead_v2.server.inbound_config import private_json
        pinned=private_json(args.publisher_trace_runtime)
        if any(digest(ROOT/name)!=expected for name,expected in pinned['worker_source_sha256'].items()):
            raise ValueError('publisher trace worker source changed since pinning')
    if selected is not None:sources.extend(['src/talkinghead_v2/server/inbound_config.py','src/talkinghead_v2/server/inbound_audio.py'])
    if args.live_qwen:sources.append('src/talkinghead_v2/providers/tts/native.py')
    source_before={path:digest(ROOT/path) for path in sources}
    binaries_before={str(path):digest(path) for path in (binary,library,ptx)}
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
    video_warnings=NativeVideoDropWarnings();livekit_logger.addHandler(video_warnings)
    ffi_trace=FfiVideoBoundaryTrace(FfiClient.instance.queue,args.stall_video_ffi_ms) if args.trace_receiver_ffi else None
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
        selected_cases=[('qwen_live',None,None)] if args.live_qwen else [('qwen_recorded',len(audio),None),('segmented_gap',len(audio),None),('one',1,None),('short',479,None),('aligned',480,None),
            ('partial',481,None),('quantum',960,None),('tail',961,None),('grid_before',1599,None),('grid_exact',1600,None),
            ('grid_after',1601,None),('grid_second',3200,None),('grid_third',4800,None),('empty',0,'empty'),('wrong_epoch',960,'epoch'),
            ('gap',960,'gap'),('sequence',960,'sequence'),('wrong_motion',960,'motion'),('backpressure_cancel',960,'cancel')]
        for name,count,fault in selected_cases:
            directory=args.directory/name;directory.mkdir(mode=0o700);sid='claire-worker-'+name;room_name='claire-worker-'+name
            tts=None;live_metrics=None
            receive_room=None;tasks=[];streams=[];tracks={};queues={};receiver={'audio_frames':0,'audio_samples':0,'maximum_rms':0.,'video':[],'idle_frames':0,'preroll_frames':0};subscribed=asyncio.Event();kinds=set()
            if name.startswith('qwen_'):
                receive_room=rtc.Room();owned_rooms.append(receive_room)
                @receive_room.on('track_subscribed')
                def subscribed_track(track,publication,participant):
                    async def consume():
                        video=track.kind==rtc.TrackKind.KIND_VIDEO
                        audio_stream=QualifiedAudioStream if selected is not None else rtc.AudioStream
                        stream=QualifiedVideoStream(track,args.native_video_queue_frames) if video else audio_stream(track,sample_rate=48000,num_channels=1,frame_size_ms=10)
                        if type(stream._queue).__name__!='RingQueue':raise ValueError('unqualified receiver queue implementation')
                        stream._queue=CheckedReceiveQueue(30);streams.append(stream)
                        kind='video' if video else 'audio';tracks[kind]=track;queues[kind]=stream._queue;kinds.add(kind)
                        if video and ffi_trace:ffi_trace.register(stream._ffi_handle.handle,sid)
                        if len(kinds)==2:subscribed.set()
                        async for item in stream:
                            if video:
                                if item.metadata is None:raise ValueError('missing received generation metadata')
                                identity=json.loads(item.metadata.user_data)
                                if identity['session_id']!=sid or identity['turn_id']!='test' or identity['epoch']!=7:raise ValueError('stale received epoch')
                                if identity.get('idle_before_audio'):
                                    if 'source_sample' in identity or 'neural_sample_16k' in identity:raise ValueError('neutral preroll invented a source timestamp')
                                    receiver['preroll_frames']+=1
                                    continue
                                if identity['source_sample']%1600 or not 0<=identity['source_sample']-identity['neural_sample_16k']*3<=2:raise ValueError('received rational source mapping changed')
                                if identity.get('idle_after_eof'):
                                    receiver['idle_frames']+=1
                                    continue
                                arrival_ns=time.monotonic_ns()
                                first_ns=receiver.setdefault('_first_source_arrival_ns',arrival_ns)
                                receiver['video'].append({'source_sample':identity['source_sample'],'source_pts_us':item.metadata.user_timestamp,
                                    'frame_id':item.metadata.frame_id,'arrival_offset_us':(arrival_ns-first_ns)//1000,
                                    'decoded_sha256':hashlib.sha256(item.frame.data).hexdigest(),'width':item.frame.width,'height':item.frame.height})
                            else:
                                values=item.frame.data;receiver['audio_frames']+=1;receiver['audio_samples']+=len(values)
                                rms=math.sqrt(sum(float(value)**2 for value in values)/max(1,len(values)));receiver['maximum_rms']=max(receiver['maximum_rms'],rms)
                        if not video and selected is not None:raise ValueError('bounded receiver audio ended unexpectedly')
                    tasks.append(asyncio.create_task(consume()))
                await receive_room.connect(settings.livekit_url,grant(settings,room_name,'observer-'+sid,'controller'))
            if args.live_qwen:
                tts=QwenNativeProvider(native,diagnostic_log=directory/'qwen-stderr.log',device_lease=lease)
                owned_tts.append(tts)
                await tts.open_session(GenerationKey(sid,'test',7));qwen_pids.add(tts._process.pid)
            private_write(directory/'bootstrap.txt',('\n'.join(('THA_CLAIRE_MEDIA_2',settings.livekit_url,
                grant(settings,room_name,'media-'+sid,'media'),sid,'test','7',str(args.prebuffer_ms)))+'\n').encode())
            env=dict(os.environ,THA_DEVICE_LEASE_FD=str(lease.fileno),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',
                LD_LIBRARY_PATH=str(runtime/'lib')+':/usr/local/cuda-12.8/lib64',CUDA_CACHE_PATH=str(ROOT/'var/cache/claire-cuda'))
            env.pop('THA_CLAIRE_CAPTURE_TRACE',None)
            for key in ('THA_CLAIRE_PUBLISH_TRACE','THA_CLAIRE_TRACE_FFI_PATH','THA_LK_TRUST_EXTERNAL_RATE_CONTROL','LD_PRELOAD','LD_AUDIT'):env.pop(key,None)
            if publisher:
                env['LD_LIBRARY_PATH']=str(publisher.ffi_library.parent)+':'+env['LD_LIBRARY_PATH']
                env['THA_CLAIRE_PUBLISH_TRACE']='1';env['THA_CLAIRE_TRACE_FFI_PATH']=str(publisher.ffi_library)
                if args.trust_encoded_rate_control:env['THA_LK_TRUST_EXTERNAL_RATE_CONTROL']='1'
            # This experiment changes only the receiver factory, not the
            # external NVENC publisher's independently qualified environment.
            env.pop('LK_DISABLE_NVDEC',None)
            if args.trace_source_captures:env['THA_CLAIRE_CAPTURE_TRACE']='1'
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
                        if name=='segmented_gap' and offset==19200:await asyncio.sleep(.3)
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
                            result.get('presentation_ack') is not False or result.get('preroll_video_frames',0)<3 or result.get('pose_queue_peak_frames',33)>32 or result.get('motion_queue_peak_packets',51)>50 or
                            result.get('source_video_frames',0)+result.get('video_skipped',0)!=(count+1599)//1600):raise ValueError('Claire sender EOF/queue accounting failed in '+name)
                        record={'name':name,'sender':{key:value for key,value in result.items() if key not in {'session_id','turn_id'}},'process_exit':0}
                        if live_metrics:record['live_qwen']=live_metrics
                        if receive_room:
                            await asyncio.sleep(.4)
                            for task in tasks:
                                if task.done():task.result()
                            try:
                                stats=await asyncio.wait_for(tracks['video'].get_stats(),5)
                                receiver['video_rtp_stats']=receiver_video_stats(stats)
                                receiver['video_rtp_stats_available']=bool(receiver['video_rtp_stats'])
                            except Exception:
                                receiver['video_rtp_stats']=[];receiver['video_rtp_stats_available']=False
                            for task in tasks:
                                if task.done():task.result()
                            receiver['checked_queue_peak_frames']={kind:queue.peak for kind,queue in queues.items()}
                            receiver['requested_native_video_queue_frames']=args.native_video_queue_frames
                            events,lower_bound=video_warnings.snapshot()
                            receiver['native_video_drop_warning_events']=events
                            receiver['native_video_drop_reported_lower_bound']=lower_bound
                            receiver['native_video_drop_warning_scope']='process-wide periodic SDK logs; absence does not prove no overflow'
                            receiver.pop('_first_source_arrival_ns',None)
                            rows=receiver['video'];positions=[row['source_sample'] for row in rows]
                            if (len(rows)<50 or len(set(row['decoded_sha256'] for row in rows))<10 or receiver['maximum_rms']<100 or
                                positions!=sorted(set(positions)) or positions[0]!=0 or positions[-1]!=(count-1)//1600*1600 or receiver['preroll_frames']<1 or receiver['idle_frames']<1 or any(row['width']!=512 or row['height']!=512 for row in rows)):
                                raise ValueError('actual receiver did not decode moving Claire/audio')
                            receiver['missing_source_frame_indices']=[index for index in range((count+1599)//1600) if index*1600 not in positions]
                            if ffi_trace:
                                boundary=ffi_trace.snapshot(next(stream._ffi_handle.handle for stream in streams if isinstance(stream,QualifiedVideoStream)))
                                callback_samples=[row['source_sample'] for row in boundary['source_frames']]
                                expected=set(range(0,count,1600));callbacks=set(callback_samples)
                                if not set(positions)<=callbacks or not callbacks<=expected:raise ValueError('FFI callback/consumer source identity mismatch')
                                boundary['missing_before_ffi_callback']=[index for index in range((count+1599)//1600) if index*1600 not in callbacks]
                                boundary['ffi_callbacks_missing_at_consumer']=[sample//1600 for sample in callback_samples if sample not in positions]
                                receiver['ffi_video_boundary']=boundary
                            record['receiver']=deepcopy(receiver)
                        cases.append(record);process.send_signal(signal.SIGTERM);await asyncio.wait_for(process.wait(),5)
                        if process.returncode:raise ValueError('post-EOF owner retirement failed')
                        if args.trace_source_captures:
                            lines=read_capture_log(directory/'stderr.log')
                            trace=source_capture_trace(lines,result['source_video_frames'])
                            sent={row['frame_id']:row['source_sample'] for row in trace}
                            if any(sent.get(row['frame_id'])!=row['source_sample'] for row in receiver['video']):
                                raise ValueError('receiver source identity differs from capture trace')
                            if ffi_trace and any(sent.get(row['frame_id'])!=row['source_sample'] for row in record['receiver']['ffi_video_boundary']['source_frames']):
                                raise ValueError('FFI callback identity differs from source capture trace')
                            record['source_capture_trace']=trace;record['sender_rtp_stats']=sender_rtp_stats(lines)
                            record['sender_rtp_after_300ms']=sender_rtp_stats(lines,'THA_SENDER_RTP_AFTER_300MS')
                            if publisher:record['publisher_stage_trace']=publish_stage_trace(lines,trace)
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
        if ffi_trace:ffi_trace.close()
        sample_stop.set();sampling.join(timeout=10);lease.release();livekit_logger.removeHandler(video_warnings)
        private_write(args.directory/'memory.json',(json.dumps({'samples':memory,'query_failures':query_failures},indent=2)+'\n').encode())
    lease.acquire('claire-session-post-exit');lease.release()
    if len(set(bindings))!=len(bindings):raise ValueError('publication IDs reused across retired epochs')
    if (source_before!={path:digest(ROOT/path) for path in sources} or binaries_before!={str(path):digest(path) for path in (binary,library,ptx)}):
        raise ValueError('source or binary changed during qualification; private observations retained without a passing export')
    if selected is not None and (digest(selected.library)!=selected.library_sha256 or digest(args.receiver_runtime)!=selected.revision):
        raise ValueError('selected receiver binary or descriptor changed during qualification')
    if publisher and PublisherTraceConfig.load(args.publisher_trace_runtime).revision!=publisher.revision:
        raise ValueError('publisher trace runtime changed during qualification')
    report={'schema_version':1,'observation':'local actual Claire SDK/GPU/NVENC/LiveKit sender and Python receiver; '+('live cached Qwen' if args.live_qwen else 'recorded synthetic-reference Qwen PCM'),
        'source_sha256':{path:digest(ROOT/path) for path in sources},'binary_sha256':digest(binary),'bridge_sha256':digest(library),'ptx_sha256':digest(ptx),
        'bridge_report_sha256':digest(ROOT/'benchmarks/reports/p03-claire-bridge.json'),'engine_sha256':engine['engine_sha256'],'scene_sha256':scene['geometry_sha256'],
        'source_wave_sha256':digest(wave),'source_pcm_24k_sha256':hashlib.sha256(raw).hexdigest(),'cases':cases,'fresh_publication_ids':True,
        'admission_reacquired':True,'audio_queue_bound_samples':48000,'host_motion_queue_bound_packets':50,'motion_input_lead_bound_ms':800,
        'eligible_motion_start_buffer_ms':args.prebuffer_ms,'protocol':'THA_CLAIRE_MEDIA_2',
        'live_qwen_streaming':args.live_qwen,'browser_receiver':False,'physical_av_qualified':False,'qwen_coload_qualified':False,'quality_qualified':False,'additional_downloads':False}
    report['source_capture_trace_enabled']=args.trace_source_captures
    report['receiver_ffi_trace_enabled']=args.trace_receiver_ffi
    report['publisher_trace_runtime']=None if publisher is None else {'descriptor_sha256':publisher.revision,
        'ffi_sha256':digest(publisher.ffi_library),'cpp_sha256':digest(publisher.cpp_library),
        'executable_sha256':digest(publisher.executable),'serving_runtime':False,'trusted_external_rate_control':args.trust_encoded_rate_control}
    report['receiver_runtime']={'library_sha256':digest(Path(FfiClient.instance._ffi_lib._name)),
        'source_matched':selected is not None,'descriptor_sha256':selected.revision if selected else None,
        'native_audio_queue_frames':25 if selected else None,'python_audio_queue_frames_10ms':30,
        'native_video_queue_frames':args.native_video_queue_frames,'python_video_queue_frames':30,
        'nvdec_disabled_by_environment':'LK_DISABLE_NVDEC' in os.environ}
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


async def run(args):
    config=None
    if 'LK_DISABLE_NVDEC' in os.environ:raise ValueError('use explicit receiver decoder selection instead of an inherited override')
    if args.receiver_decoder=='disable-nvdec':
        if args.receiver_runtime is None:raise ValueError('decoder selection requires the source-matched receiver runtime')
        os.environ['LK_DISABLE_NVDEC']='1'
    if args.receiver_runtime is not None:
        from talkinghead_v2.server.inbound_config import InboundAudioConfig
        from talkinghead_v2.server.inbound_audio import validate_loaded_runtime
        config=InboundAudioConfig.load(args.receiver_runtime)
        selected=config.environment(os.environ)
        for key in ('LD_PRELOAD','LD_AUDIT'):os.environ.pop(key,None)
        os.environ['LIVEKIT_LIB_PATH']=selected['LIVEKIT_LIB_PATH']
        validate_loaded_runtime(config)
    args.selected_receiver=config
    try:await main(args)
    finally:
        if config is not None:
            from talkinghead_v2.server.inbound_audio import dispose_owned_runtime
            await asyncio.to_thread(dispose_owned_runtime,config)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live-qwen',action='store_true',help='execute one pinned cached-Qwen/SFU/Claire co-load case instead of recorded/fault cohort')
    parser.add_argument('--trace-source-captures',action='store_true',help='live qualification only: privately trace source capture timing and payload sizes')
    parser.add_argument('--native-video-queue-frames',type=int,choices=(1,8),default=1,help='request bounded native receiver queue for research comparison')
    parser.add_argument('--receiver-runtime',type=Path,help='explicit qualified source-matched FFI descriptor; no provisioning')
    parser.add_argument('--publisher-trace-runtime',type=Path,help='explicit isolated SDK/worker stage-trace descriptor; never a serving runtime')
    parser.add_argument('--trust-encoded-rate-control',action='store_true',help='isolated opt-in experiment; external NVENC must honor rate feedback')
    parser.add_argument('--receiver-decoder',choices=('auto','disable-nvdec'),default='auto',help='source-matched receiver factory only; publisher remains unchanged')
    parser.add_argument('--trace-receiver-ffi',action='store_true',help='qualification-only bounded source identities before Python dispatch')
    parser.add_argument('--stall-video-ffi-ms',type=int,choices=(0,500),default=0,help='deliberate one-callback stall for an isolated queue positive control')
    parser.add_argument('--prebuffer-ms',type=int,choices=(300,500),default=500)
    for name in ('config','engine','scene','directory','report'):parser.add_argument('--'+name,type=Path,required=True)
    asyncio.run(run(parser.parse_args()))
