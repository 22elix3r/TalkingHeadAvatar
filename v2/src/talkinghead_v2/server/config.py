"""Operator-owned, loopback-only reference configuration. Never fetch models."""

from dataclasses import dataclass
import json
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class Settings:
    database_url: str
    livekit_url: str
    livekit_key: str
    livekit_secret: str
    ollama_url: str
    llm_model: str
    llm_digest: str
    asr_path: Path
    asr_files: dict
    tts_runtime: Path
    web_dist: Path
    private_dir: Path
    origin: str='http://127.0.0.1:18700'
    media_backend: str='raw_reference'
    native_media_runtime: Path | None=None

    @classmethod
    def load(cls,path: Path):
        d=json.loads(path.read_text())
        for name in ('database_url','livekit_url','ollama_url','origin'):
            if urlparse(d[name]).hostname not in {'127.0.0.1','localhost','::1'}:
                raise ValueError('P04 development mode refuses non-loopback endpoints')
        for name in ('asr_path','tts_runtime','web_dist','private_dir'): d[name]=(path.parent/d[name]).resolve()
        if d.get('media_backend','raw_reference') not in {'raw_reference','native_static'}:
            raise ValueError('unknown media backend')
        if d.get('native_media_runtime') is not None:
            d['native_media_runtime']=(path.parent/d['native_media_runtime']).resolve()
        if d.get('media_backend')=='native_static' and not d.get('native_media_runtime'):
            raise ValueError('native_static requires an explicit pinned runtime')
        return cls(**d)
