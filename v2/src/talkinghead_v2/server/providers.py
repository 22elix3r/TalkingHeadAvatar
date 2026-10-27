"""Pinned cached CPU ASR and explicitly configured CPU Ollama, separate from TTS."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import httpx
import numpy as np


class Asr:
    def __init__(self,path,hashes):
        for name,expected in hashes.items():
            if hashlib.sha256((path/name).read_bytes()).hexdigest()!=expected: raise ValueError('ASR cache identity mismatch')
        from faster_whisper import WhisperModel
        self.model=WhisperModel(str(path),device='cpu',compute_type='int8',cpu_threads=2,num_workers=1,local_files_only=True)
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='tha-cpu-asr')
        self.lock=asyncio.Lock()
    def decode(self,pcm):
        audio=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768
        segments,_=self.model.transcribe(audio,language='en',beam_size=1,vad_filter=False,
            condition_on_previous_text=False,without_timestamps=True)
        return ' '.join(s.text.strip() for s in segments).strip()
    async def transcribe(self,pcm):
        async with self.lock:
            pending=asyncio.get_running_loop().run_in_executor(self.executor,self.decode,pcm)
            try: return await asyncio.shield(pending)
            except asyncio.CancelledError:
                # Keep the single CPU slot until native decode has really returned.
                await pending
                raise
    def close(self): self.executor.shutdown(wait=False,cancel_futures=True)


class Llm:
    def __init__(self,settings):
        self.settings=settings
        self.client=httpx.AsyncClient(base_url=settings.ollama_url,timeout=httpx.Timeout(90,connect=5),trust_env=False)
        self.options={'num_gpu':0,'num_ctx':2048,'num_predict':80,'num_thread':4,'temperature':.3}
    async def prepare(self):
        response=await self.client.get('/api/tags');response.raise_for_status()
        if not any(m['name']==self.settings.llm_model and m['digest']==self.settings.llm_digest for m in response.json()['models']):
            raise ValueError('configured cached LLM identity unavailable; no pull attempted')
        response=await self.client.post('/api/generate',json={'model':self.settings.llm_model,'prompt':'','stream':False,
            'options':self.options,'keep_alive':'5m'})
        response.raise_for_status()
        response=await self.client.get('/api/ps');response.raise_for_status()
        if not any(m['name']==self.settings.llm_model and m.get('size_vram',-1)==0 for m in response.json()['models']):
            raise RuntimeError('CPU-only LLM residency was not verified')
    async def tokens(self,users):
        messages=[{'role':'system','content':'Answer in one short plain English sentence, at most twelve words. Do not use markup. Be helpful and direct.'}]
        # Assistant history is omitted until a qualified receiver presentation
        # acknowledgment exists; generated/cancelled speech is not treated as heard.
        messages.extend({'role':'user','content':t[:1500]} for t in users[-4:])
        async with self.client.stream('POST','/api/chat',json={'model':self.settings.llm_model,'messages':messages,
                'stream':True,'think':False,'options':self.options,'keep_alive':'5m'}) as response:
            response.raise_for_status()
            done=False
            async for line in response.aiter_lines():
                if not line: continue
                if len(line)>65536: raise RuntimeError('oversized LLM stream event')
                event=json.loads(line)
                if event.get('error'): raise RuntimeError('LLM generation failed')
                token=event.get('message',{}).get('content','')
                if token: yield token
                if event.get('done'): done=True;break
            if not done: raise RuntimeError('LLM stream ended without completion')
    async def close(self): await self.client.aclose()
