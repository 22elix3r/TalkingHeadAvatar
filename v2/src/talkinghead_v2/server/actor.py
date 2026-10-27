"""One session actor owns turn revisions, epochs and live queues."""

import asyncio
from collections import deque
import json
import logging
import time
import uuid

from livekit import rtc
import webrtcvad

from talkinghead_v2.conversation import TurnDetector,StableSegmenter
from talkinghead_v2.contracts import Envelope,GenerationKey,TextSegment
from talkinghead_v2.providers.tts.native import QwenNativeProvider
from talkinghead_v2.server.transport import Media,checked_audio_stream,grant
from talkinghead_v2.session.device import DeviceBusy,DeviceLease


class Events:
    def __init__(self):
        self.rows=deque(maxlen=256);self.sequence=0;self.changed=asyncio.Condition()
    async def emit(self,kind,epoch,**data):
        async with self.changed:
            self.sequence+=1
            row={'sequence':self.sequence,'kind':kind,'epoch':epoch,'clock_id':'controller_monotonic',
                 'monotonic_ns':time.perf_counter_ns(),**data}
            self.rows.append(row);self.changed.notify_all()
            return row
    async def stream(self,last=0):
        while True:
            heartbeat=False
            async with self.changed:
                if self.sequence<=last:
                    try: await asyncio.wait_for(self.changed.wait(),10)
                    except TimeoutError:
                        heartbeat=True
                rows=[r for r in self.rows if r['sequence']>last]
            if heartbeat:
                yield ': heartbeat\n\n';continue
            for row in rows:
                last=row['sequence']
                yield f'id: {last}\ndata: {json.dumps(row)}\n\n'
                if row['kind']=='closed': return


class Actor:
    def __init__(self,settings,store,row,asr,llm,native_config,avatar_config=None):
        self.settings,self.store,self.row=settings,store,row
        self.asr,self.llm=asr,llm
        lease_path=getattr(native_config,'device_lease_path',None)
        self.device_lease=DeviceLease(lease_path) if lease_path else None
        self.tts=QwenNativeProvider(native_config,diagnostic_log=settings.private_dir/'native-diagnostics.log',device_lease=self.device_lease)
        self.visual_mode='static_procedural_reference'
        if getattr(settings,'media_backend','raw_reference')=='native_claire_research':
            from talkinghead_v2.server.claire_media import ClaireMedia
            if avatar_config is None: raise ValueError('pinned Claire serving revision required')
            self.media=ClaireMedia(settings,row.room,row.id,self.device_lease,expected_revision=avatar_config.revision)
            self.visual_mode='claire_vendor_research'
        elif getattr(settings,'media_backend','raw_reference')=='native_static':
            from talkinghead_v2.server.native_media import NativeMedia
            self.media=NativeMedia(settings,row.room,row.id,self.device_lease)
        else: self.media=Media(settings,row.room,row.id)
        self.controller=rtc.Room()
        self.events=Events();self.detector=TurnDetector();self.vad=webrtcvad.Vad(2)
        self.epoch=0;self.closed=False;self.ready=False;self.responding=False
        self.control_lock=asyncio.Lock();self.generation_lock=asyncio.Lock()
        self.response=None;self.partial=None;self.input_task=None;self.heartbeat=None
        self.users=[];self.revision=0;self.utterance=0;self.next_partial=16000
        self.voiced_run=0;self.metrics=deque(maxlen=64)
        self.client_seen=False;self.ready_at=0.
        self.last_browser_heartbeat=time.monotonic();self.final_task=None
        self.closing=False;self.preparation=None;self.close_done=asyncio.Event();self.closed_persisted=False
    async def prepare(self):
        try:
            if self.device_lease: self.device_lease.acquire(self.row.id)
            await self.tts.open_session(GenerationKey(self.row.id,'warmup',0));await self.tts.cancel()
            await self.media.connect()
            @self.controller.on('track_published')
            def published(publication,participant):
                if participant.identity=='user-'+self.row.id and publication.source==rtc.TrackSource.SOURCE_MICROPHONE:
                    publication.set_subscribed(True)
            @self.controller.on('track_subscribed')
            def subscribed(track,publication,participant):
                if participant.identity=='user-'+self.row.id and track.kind==rtc.TrackKind.KIND_AUDIO:
                    if self.input_task: self.input_task.cancel()
                    self.input_task=asyncio.create_task(self.receive(track))
            @self.controller.on('participant_disconnected')
            def left(participant):
                if participant.identity=='user-'+self.row.id: asyncio.create_task(self.close())
            @self.controller.on('participant_connected')
            def joined(participant):
                if participant.identity=='user-'+self.row.id: self.client_seen=True
            await self.controller.connect(self.settings.livekit_url,
                grant(self.settings,self.row.room,'controller-'+self.row.id,'controller'),
                options=rtc.RoomOptions(auto_subscribe=False))
            await asyncio.to_thread(self.store.update,self.row.id,status='READY')
            self.ready=True;self.ready_at=time.monotonic();self.heartbeat=asyncio.create_task(self.lease())
            await self.events.emit('ready',self.epoch,visual_mode=self.visual_mode)
        except DeviceBusy:
            await self.fail('gpu_capacity_busy')
        except Exception:
            await self.fail('preparation_failed')
    async def lease(self):
        try:
            while not self.closed:
                await asyncio.sleep(2)
                if not self.client_seen and time.monotonic()-self.ready_at>30:
                    await self.fail('client_join_timeout');return
                if self.client_seen and time.monotonic()-self.last_browser_heartbeat>30:
                    await self.fail('client_disconnected');return
                await asyncio.to_thread(self.store.update,self.row.id,epoch=self.epoch)
        except asyncio.CancelledError: pass
        except Exception: await self.fail('lease_lost')
    async def receive(self,track):
        stream=checked_audio_stream(track)
        try:
            async for event in stream:
                if self.closed: break
                pcm=bytes(event.frame.data)
                if len(pcm)!=640: raise RuntimeError('unexpected microphone frame size')
                voiced=self.vad.is_speech(pcm,16000)
                self.voiced_run=self.voiced_run+1 if voiced else 0
                if self.responding and self.voiced_run==9:
                    await self.interrupt('vad_barge_in')
                endpoint,started=self.detector.push(pcm,voiced)
                if started:
                    self.utterance+=1;self.revision=0
                    await self.events.emit('user_speaking',self.epoch,utterance=self.utterance)
                if self.detector.frames and self.detector.cursor>=self.next_partial and (self.partial is None or self.partial.done()):
                    self.next_partial=self.detector.cursor+16000
                    self.partial=asyncio.create_task(self.caption(b''.join(self.detector.frames),self.utterance))
                if endpoint:
                    utterance=self.utterance
                    if self.final_task and not self.final_task.done(): self.final_task.cancel()
                    self.final_task=asyncio.create_task(self.commit_audio(endpoint,utterance))
        except asyncio.CancelledError: pass
        except Exception: await self.fail('microphone_discontinuity')
        finally: await stream.aclose()
    async def caption(self,pcm,utterance):
        try:
            text=await self.asr.transcribe(pcm)
            if not self.closed and utterance==self.utterance and text:
                self.revision+=1
                await self.events.emit('caption',self.epoch,speaker='user',text=text[:2000],final=False,
                                       utterance=utterance,revision=self.revision)
        except Exception:
            await self.events.emit('warning',self.epoch,code='partial_asr_unavailable')
    async def commit_audio(self,endpoint,utterance):
        started=time.perf_counter_ns()
        try:
            text=await self.asr.transcribe(endpoint.pcm)
            if self.closed or utterance!=self.utterance or not text: return
            self.revision+=1
            await self.events.emit('caption',self.epoch,speaker='user',text=text[:2000],final=True,
                utterance=utterance,revision=self.revision,endpoint_patience_samples=endpoint.committed_sample-endpoint.last_voiced_sample,
                endpoint_boundary='inbound decoded 16kHz VAD edge; client capture endpoint unqualified',asr_wall_ns=time.perf_counter_ns()-started)
            await self.answer(text)
        except Exception: await self.fail('asr_failed')
    async def interrupt(self,reason='explicit_interrupt'):
        async with self.generation_lock:
            if self.closed: return
            self.epoch+=1
            self.utterance+=1  # Fence ASR results already executing on the CPU.
            self.media.fence()
            task=self.response;self.response=None;self.responding=False
            # Reject the generation before any await. Source clear alone is not a
            # receiver flush: browser gates old track IDs and waits for replacement.
            await self.events.emit('interrupted',self.epoch,reason=reason,receiver_safe_resume=False)
            ack=await self.tts.cancel()
            if task and task is not asyncio.current_task():
                task.cancel();await asyncio.gather(task,return_exceptions=True)
            await self.media.cancel()
            try: await asyncio.to_thread(self.store.update,self.row.id,epoch=self.epoch,status='READY')
            except Exception:
                if reason!='session_closed': raise
            await self.events.emit('cancel_ack',self.epoch,compute_stopped=ack['compute_stopped'],
                                   source_tracks_replaced_on_next_answer=True,receiver_safe_resume=False)
    async def answer(self,text):
        if self.closed or self.closing or not self.ready: return
        await self.interrupt('new_turn')
        self.users.append(text[:1500]);self.users=self.users[-4:]
        self.responding=True
        key=GenerationKey(self.row.id,uuid.uuid4().hex,self.epoch)
        self.response=asyncio.create_task(self.generate(key))
    async def generate(self,key):
        tasks=[]
        try:
            await self.tts.open_session(key)
            binding=await self.media.start(key)
            await self.events.emit('playback_binding',self.epoch,**{k:v for k,v in binding.items() if k!='epoch'})
            await asyncio.to_thread(self.store.update,self.row.id,epoch=self.epoch,status='ACTIVE')
            segments=asyncio.Queue(maxsize=2)
            async def text_producer():
                segmenter=StableSegmenter()
                last_release=time.monotonic()
                async for token in self.llm.tokens(self.users):
                    if key.epoch!=self.epoch: return
                    released=segmenter.push(token)
                    if not released and time.monotonic()-last_release>=.12:
                        released=segmenter.release_words()
                    for text in released: await segments.put(text)
                    if released: last_release=time.monotonic()
                for text in segmenter.push('',final=True): await segments.put(text)
                await segments.put(None)
            async def speak():
                sequence=0
                while (text:=await segments.get()) is not None:
                    if key.epoch!=self.epoch: return
                    await self.events.emit('caption',self.epoch,speaker='assistant',text=text,final=False,segment=sequence)
                    if not await self.tts.push(TextSegment(Envelope(key,sequence,'conversation'),str(sequence),text,'en',self.tts.config.voice_id)):
                        return
                    self.metrics.append({'epoch':key.epoch,'tts_ttfa_ns':self.tts.last_metrics['callbacks'][0]['received_ns'],
                                         'native_samples':self.tts.last_metrics['samples']})
                    sequence+=1
                await self.tts.flush()
            async def publish():
                async for event in self.tts.outputs():
                    if key.epoch!=self.epoch or not await self.media.submit(event): return
            tasks=[asyncio.create_task(f()) for f in (text_producer,speak,publish)]
            await asyncio.gather(*tasks)
            if key.epoch==self.epoch and not self.closed:
                self.responding=False
                await self.events.emit('sender_complete',self.epoch,valid_audio_samples=self.media.cursor,
                    codec_padding_samples=self.media.padding,presentation_ack=False,
                    native_source_metrics=getattr(self.media,'sender_metrics',None))
                await asyncio.to_thread(self.store.update,self.row.id,status='READY')
        except asyncio.CancelledError: raise
        except Exception:
            logging.getLogger(__name__).exception('P04 response failed')
            if not self.closed and key.epoch==self.epoch:
                await self.events.emit('error',self.epoch,code='response_failed')
                self.responding=False
                await self.tts.cancel();await self.media.cancel()
        finally:
            for task in tasks:
                if not task.done(): task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
    async def fail(self,code):
        await self.events.emit('error',self.epoch,code=code)
        await self.close(error=code)
    async def close(self,error=None):
        if self.closing:
            await self.close_done.wait()
            return self.closed_persisted
        self.closing=True
        try:
            if self.preparation and self.preparation is not asyncio.current_task() and not self.preparation.done():
                self.preparation.cancel();await asyncio.gather(self.preparation,return_exceptions=True)
            try: await self.interrupt('session_closed')
            except Exception: logging.getLogger(__name__).warning('source cancellation failed during close')
            self.closed=True;self.ready=False
            for task in (self.input_task,self.partial,self.final_task,self.heartbeat):
                if task and task is not asyncio.current_task(): task.cancel()
            await asyncio.gather(self.tts.close(),self.media.close(),self.controller.disconnect(),return_exceptions=True)
            try:
                await asyncio.to_thread(self.store.update,self.row.id,epoch=self.epoch,status='FAILED' if error else 'CLOSED',error=error)
                self.closed_persisted=True
            except Exception: logging.getLogger(__name__).warning('source stopped but durable close is pending')
            await self.events.emit('closed',self.epoch,code=error)
            return self.closed_persisted
        finally:
            if getattr(self,'device_lease',None): self.device_lease.release()
            self.closed=True;self.ready=False;self.close_done.set()
