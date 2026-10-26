"""Bounded turn audio and irreversible text segmentation; no model imports."""

from collections import deque
from dataclasses import dataclass
import re

from talkinghead_v2.contracts import ContractError


@dataclass(frozen=True)
class Endpoint:
    pcm: bytes
    first_sample: int
    last_voiced_sample: int
    committed_sample: int
    reason: str


class TurnDetector:
    """20 ms mono s16/16 kHz; pre-roll, conservative silence, bounded utterance.

    VAD decisions are supplied by the CPU adapter. These indices name inbound
    decoded audio, not the user's original capture clock or an annotated endpoint.
    """
    def __init__(self, *, patience_frames=60, max_frames=1500):
        if not 10 <= patience_frames <= 100 or not patience_frames < max_frames <= 3000:
            raise ContractError("invalid bounded endpoint policy")
        self.patience_frames,self.max_frames=patience_frames,max_frames
        self.cursor=0
        self.preroll=deque(maxlen=10)
        self.frames=[]
        self.voiced_count=self.silence=self.run=0
        self.first_sample=self.last_voiced=0

    def push(self,pcm: bytes,voiced: bool):
        if not isinstance(pcm,bytes) or len(pcm)!=640 or type(voiced) is not bool:
            raise ContractError("VAD requires exactly 20 ms of mono PCM16 at 16 kHz")
        self.cursor+=320
        self.run=self.run+1 if voiced else 0
        started=False
        if not self.frames:
            self.preroll.append(pcm)
            if self.run<3: return None,False
            self.frames=list(self.preroll)
            self.first_sample=self.cursor-len(self.frames)*320
            self.voiced_count=3
            self.preroll.clear()
            started=True
        else:
            self.frames.append(pcm)
            self.voiced_count+=int(voiced)
        if voiced: self.last_voiced=self.cursor
        self.silence=0 if voiced else self.silence+1
        reason='maximum_utterance' if len(self.frames)>=self.max_frames else 'silence'
        if len(self.frames)>=self.max_frames or self.silence>=self.patience_frames:
            # Keep 80 ms past the VAD edge; silence patience remains in latency.
            keep=min(len(self.frames),(self.last_voiced-self.first_sample)//320+4)
            result=Endpoint(b''.join(self.frames[:keep]),self.first_sample,self.last_voiced,self.cursor,reason)
            valid=self.voiced_count>=6
            self.frames=[];self.preroll.clear();self.voiced_count=self.silence=self.run=0
            return result if valid else None,started
        return None,started


class StableSegmenter:
    """Speak only complete words/clauses, preserving numbers and abbreviations."""
    abbreviations={'mr.','mrs.','ms.','dr.','prof.','e.g.','i.e.','a.m.','p.m.'}
    def __init__(self,max_words=12):
        self.buffer=''
        self.max_words=max_words
    def push(self,token: str,*,final=False) -> list[str]:
        if not isinstance(token,str) or len(self.buffer)+len(token)>65536:
            raise ContractError("LLM text exceeds the bounded segment buffer")
        self.buffer+=token
        result=[]
        while self.buffer:
            end=None
            for match in re.finditer(r'\S+\s+',self.buffer):
                word=match.group().strip()
                if word[-1:] in '.?!' and word.lower() not in self.abbreviations and not re.fullmatch(r'(?:[A-Za-z]\.)+',word):
                    end=match.end();break
            complete=list(re.finditer(r'\S+\s+',self.buffer))
            if end is None and len(complete)>=self.max_words: end=complete[self.max_words-1].end()
            if end is None:
                if final: end=len(self.buffer)
                else: break
            text=self.buffer[:end].strip()
            self.buffer=self.buffer[end:]
            if text: result.append(text)
        return result

    def release_words(self) -> list[str]:
        complete=list(re.finditer(r'\S+\s+',self.buffer))
        if len(complete)<4: return []
        end=complete[min(self.max_words,len(complete))-1].end()
        text,self.buffer=self.buffer[:end].strip(),self.buffer[end:]
        return [text] if text else []
