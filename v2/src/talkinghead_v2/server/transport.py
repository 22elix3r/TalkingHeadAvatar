"""Self-hosted LiveKit reference. Track replacement is the receiver resume boundary."""

import asyncio
from collections import deque
from datetime import timedelta
from fractions import Fraction
from importlib.resources import files
from pathlib import Path
import time

import numpy as np
from livekit import rtc,api

from talkinghead_v2.assets.avatar import load_avatar
from talkinghead_v2.audio.resample import RationalResampler
from talkinghead_v2.audio.clock import SampleLedger
from talkinghead_v2.contracts import AudioChunk,AudioEOF,Envelope,MotionFrame
from talkinghead_v2.rendering.mesh import MeshRenderer


def grant(settings,room,identity,role):
    grants=api.VideoGrants(room_join=True,room=room,can_publish=role!='controller',
        can_subscribe=role!='media',can_publish_data=False,
        can_publish_sources=['microphone'] if role=='browser' else ['microphone','camera'] if role=='media' else [])
    return api.AccessToken(settings.livekit_key,settings.livekit_secret).with_identity(identity).with_ttl(timedelta(minutes=2)).with_grants(grants).to_jwt()


class AudioDiscontinuity(RuntimeError): pass


class CheckedReceiveQueue:
    """Pinned SDK queue adapter. Overflow terminates the turn, never evicts silently."""
    def __init__(self,max_frames=25):
        self.queue=deque();self.event=asyncio.Event();self.maximum=max_frames
        self.failed=False;self.lost_frames=0;self.peak=0
    def put(self,item):
        if self.failed:
            self.lost_frames+=int(item is not None);return
        if len(self.queue)>=self.maximum:
            self.lost_frames=len(self.queue)+1;self.failed=True;self.queue.clear()
        else: self.queue.append(item);self.peak=max(self.peak,len(self.queue))
        self.event.set()
    async def get(self):
        while not self.queue and not self.failed:
            self.event.clear();await self.event.wait()
        if self.failed: raise AudioDiscontinuity('inbound audio queue overflow')
        return self.queue.popleft()


def checked_audio_stream(track,inbound_config=None):
    if inbound_config is not None:
        from talkinghead_v2.server.inbound_audio import BoundedNativeAudioStream
        stream=BoundedNativeAudioStream(track,inbound_config)
    else:
        stream=rtc.AudioStream(track,sample_rate=16000,num_channels=1,frame_size_ms=20,capacity=0)
    # livekit 1.1.21 owns this queue. Guard the inspected integration point; do
    # not modify the SDK globally or permit its silent RingQueue eviction policy.
    if type(stream._queue).__name__!='RingQueue': raise RuntimeError('unqualified LiveKit receive queue implementation')
    stream._queue=CheckedReceiveQueue(inbound_config.python_frames if inbound_config is not None else 25)
    return stream


class Media:
    def __init__(self,settings,room_name,sid):
        self.settings,self.room_name,self.sid=settings,room_name,sid
        self.room=rtc.Room();self.audio=self.video=None;self.publications=[]
        self.key=None;self.cancelled=True;self.blocked=False;self.cursor=self.next_video=self.padding=0
        self.base_us=0
        self.asset=load_avatar(Path(str(files('talkinghead_v2')/'fixtures/p03-avatar/avatar.json')),tenant_id='procedural')
        self.renderer=MeshRenderer(self.asset,resolution=128)
    async def connect(self):
        await self.room.connect(self.settings.livekit_url,grant(self.settings,self.room_name,'media-'+self.sid,'media'),
                                options=rtc.RoomOptions(auto_subscribe=False))
    async def start(self,key):
        await self.cancel()
        if self.blocked: raise PermissionError('media_authorization_revoked')
        self.key=key;self.cancelled=False;self.cursor=self.next_video=self.padding=0
        self.ledger=SampleLedger(key,24000);self.resampler=RationalResampler(24000,48000)
        self.base_us=time.monotonic_ns()//1000
        self.audio=rtc.AudioSource(48000,1,queue_size_ms=60)
        self.video=rtc.VideoSource(128,128)
        at=rtc.LocalAudioTrack.create_audio_track('answer-'+str(key.epoch),self.audio)
        vt=rtc.LocalVideoTrack.create_video_track('static-reference-'+str(key.epoch),self.video)
        ap=await self.room.local_participant.publish_track(at,rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        self.publications.append(ap.sid)
        if self.blocked:
            await self.cancel();raise PermissionError('media_authorization_revoked')
        vp=await self.room.local_participant.publish_track(vt,rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA))
        self.publications.append(vp.sid)
        if self.blocked:
            await self.cancel();raise PermissionError('media_authorization_revoked')
        neutral=MotionFrame(Envelope(key,0,'static-reference'),self.asset.rig,0,48000,48000,tuple(0. for _ in self.asset.rig.coefficient_names))
        pixels=self.renderer.render(neutral,duration=Fraction(1,30)).rgb
        self.image=rtc.VideoFrame(128,128,rtc.VideoBufferType.RGB24,pixels)
        self.video.capture_frame(self.image,timestamp_us=self.base_us)
        return {'audio_sid':ap.sid,'video_sid':vp.sid,'publisher':'media-'+self.sid,'epoch':key.epoch,
                'visual_mode':'static_procedural_reference' if self.asset.kind=='procedural_fixture' else 'static_mesh_reference',
                'clock_mapping':'new_track_per_epoch','source_sample_rate':48000}
    async def _pcm(self,values,key,*,tail=False):
        for start in range(0,len(values),960):
            if self.cancelled or self.key!=key: return False
            valid=values[start:start+960]
            data=(np.clip(np.asarray(valid),-1,1)*32767).astype('<i2')
            if tail and len(data)%480:
                padding=480-len(data)%480
                data=np.pad(data,(0,padding));self.padding+=padding
            await self.audio.capture_frame(rtc.AudioFrame(data.tobytes(),48000,1,len(data)))
            if self.cancelled or self.key!=key: return False
            self.cursor+=len(valid)
            while self.next_video<=self.cursor:
                self.video.capture_frame(self.image,timestamp_us=self.base_us+self.next_video*1000000//48000)
                self.next_video+=1600
        return True
    async def submit(self,event):
        key=event.envelope.key
        if self.cancelled or key!=self.key: return False
        self.ledger.accept(event)
        if isinstance(event,AudioChunk):
            return await self._pcm(self.resampler.push(event.samples),key)
        if not await self._pcm(self.resampler.finish(),key,tail=True): return False
        # Flush a final codec block explicitly; silence is excluded from valid PCM.
        await self.audio.capture_frame(rtc.AudioFrame(bytes(480*2),48000,1,480));self.padding+=480
        await self.audio.wait_for_playout()
        return not self.cancelled and self.key==key
    async def cancel(self):
        self.fence()
        for sid in self.publications:
            await self.room.local_participant.unpublish_track(sid)
        self.publications=[]
        if self.audio: await self.audio.aclose()
        if self.video: await self.video.aclose()
        self.audio=self.video=None
    def fence(self,*,permanent=False):
        self.blocked=self.blocked or permanent
        self.cancelled=True
        if self.audio: self.audio.clear_queue()
    async def close(self):
        await self.cancel();await self.room.disconnect()
