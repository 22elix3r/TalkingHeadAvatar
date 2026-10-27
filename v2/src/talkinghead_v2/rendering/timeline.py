"""Bounded motion history selected by audio PTS, independent of inference completion."""

from collections import deque
from fractions import Fraction

from talkinghead_v2.audio.motion import interpolate_motion
from talkinghead_v2.contracts import ContractError, GenerationKey, MotionFrame, RigIdentity, integer


class MotionTimeline:
    def __init__(self, key: GenerationKey, rig: RigIdentity, sample_rate: int, *,
                 max_future=Fraction(1,2), max_items=64):
        integer(sample_rate,"sample rate",1); integer(max_items,"motion item bound",2)
        if not isinstance(max_future,Fraction) or max_future <= 0:
            raise ContractError("motion duration bound must be positive and exact")
        self.key,self.rig,self.sample_rate = key,rig,sample_rate
        self.max_samples = int(sample_rate*max_future)
        self.max_items = max_items
        self.frames = deque()
        self.last_sample = self.last_sequence = -1
        self.playhead = 0
        self.cancelled = False
        self.peak_items = self.rejected_stale = self.expired = 0

    def advance(self,audio_sample: int):
        integer(audio_sample,"audio playhead")
        if audio_sample < self.playhead:
            raise ContractError("audio presentation mapping must be monotonic")
        self.playhead = audio_sample
        while len(self.frames) > 1 and self.frames[1].audio_sample <= audio_sample:
            self.frames.popleft(); self.expired += 1

    def push(self,frame: MotionFrame) -> bool:
        if self.cancelled or frame.envelope.key != self.key:
            self.rejected_stale += 1
            return False
        if frame.rig != self.rig or frame.sample_rate != self.sample_rate:
            raise ContractError("motion timeline rig/timebase mismatch")
        if frame.audio_sample <= self.last_sample or frame.envelope.sequence_no <= self.last_sequence:
            raise ContractError("reordered/duplicate motion")
        if frame.valid_until > self.playhead+self.max_samples or len(self.frames) >= self.max_items:
            raise ContractError("motion timeline backpressure: future time/item bound exceeded")
        self.last_sample,self.last_sequence = frame.audio_sample,frame.envelope.sequence_no
        if frame.valid_until < self.playhead:
            self.expired += 1
            return False
        self.frames.append(frame)
        self.peak_items = max(self.peak_items,len(self.frames))
        return True

    def select(self,audio_sample: int) -> MotionFrame | None:
        self.advance(audio_sample)
        if self.cancelled or not self.frames:
            return None
        left = self.frames[0]
        if not left.audio_sample <= audio_sample <= left.valid_until:
            return None
        if len(self.frames)>1 and audio_sample < self.frames[1].audio_sample:
            return interpolate_motion(left,self.frames[1],audio_sample)
        # No unbounded extrapolation through a missing inference interval.
        from dataclasses import replace
        return replace(left,audio_sample=audio_sample)

    def cancel(self):
        self.cancelled = True
        self.frames.clear()


class FrameMailbox:
    """Completed immutable CPU frames only; no GPU surface reuse or device flush claim."""
    def __init__(self,key: GenerationKey,*,sample_rate=48000,max_items=3,max_duration=Fraction(1,10)):
        integer(sample_rate,"sample rate",1); integer(max_items,"frame bound",1)
        if not isinstance(max_duration,Fraction) or max_duration <= 0:
            raise ContractError("frame mailbox needs a positive time bound")
        self.key,self.sample_rate,self.max_items = key,sample_rate,max_items
        self.max_samples = int(max_duration*sample_rate)
        self.frames = deque()
        self.cancelled = False
        self.last_pts = -1
        self.playhead = 0
        self.dropped_late = self.rejected_stale = self.peak_items = 0

    def advance(self,playhead: int):
        integer(playhead,"audio playhead")
        if playhead < self.playhead:
            raise ContractError("video mailbox audio playhead must be monotonic")
        self.playhead = playhead

    def submit(self,frame,*,playhead: int) -> bool:
        self.advance(playhead)
        if self.cancelled or frame.key != self.key:
            self.rejected_stale += 1
            return False
        if frame.sample_rate != self.sample_rate or frame.pts_samples <= self.last_pts:
            raise ContractError("video timebase/order mismatch")
        if frame.pts_samples+frame.duration*self.sample_rate > playhead+self.max_samples:
            raise ContractError("video frame exceeds future time bound")
        if len(self.frames)>=self.max_items:
            raise ContractError("video frame mailbox item bound exceeded")
        self.last_pts = frame.pts_samples
        if frame.pts_samples+frame.duration*self.sample_rate <= playhead:
            self.dropped_late += 1
            return False
        self.frames.append(frame)
        self.peak_items = max(self.peak_items,len(self.frames))
        return True

    def take_due(self,*,playhead: int):
        self.advance(playhead)
        selected = None
        while self.frames and self.frames[0].pts_samples <= playhead:
            frame = self.frames.popleft()
            if frame.pts_samples+frame.duration*self.sample_rate <= playhead:
                self.dropped_late += 1
            else:
                selected = frame
        return None if self.cancelled else selected

    def cancel(self):
        self.cancelled = True
        self.frames.clear()
