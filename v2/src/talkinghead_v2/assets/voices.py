"""Tenant-scoped voice authorization; filenames and public availability are not consent."""

import json
from pathlib import Path
import re
import wave

from talkinghead_v2.assets.artifacts import ArtifactError, sha256_file


def read_voice(registry: Path, *, tenant_id: str) -> dict:
    if registry.stat().st_size > 1024*1024:
        raise ArtifactError("voice registry exceeds metadata limit")
    data = json.loads(registry.read_text())
    if not isinstance(data, dict):
        raise ArtifactError("voice registry must be a JSON object")
    for name in ("tenant_id", "voice_id"):
        if not isinstance(data.get(name), str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", data[name]):
            raise ArtifactError("invalid tenant/voice identifier")
    if data["tenant_id"] != tenant_id:
        raise ArtifactError("voice belongs to another tenant")
    if type(data.get("schema_version")) is not int or data.get("schema_version") != 1 or data.get("status") != "approved":
        raise ArtifactError("voice is not approved")
    if data.get("kind") not in {"human_consented", "synthetic_fixture"}:
        raise ArtifactError("voice provenance kind is required")
    uses = data.get("allowed_uses")
    if not isinstance(uses, list) or not all(isinstance(use, str) for use in uses):
        raise ArtifactError("allowed uses must be an explicit list")
    consent = data.get("consent_record_id")
    if data["kind"] == "human_consented" and (not isinstance(consent, str) or not consent.strip() or "research_voice_clone" not in uses):
        raise ArtifactError("human voice needs explicit research-cloning consent and allowed use")
    if data["kind"] == "synthetic_fixture" and data.get("allowed_uses") != ["synthetic_plumbing_only"]:
        raise ArtifactError("synthetic fixture cannot establish human cloning qualification")
    relative = Path(data.get("audio_path", ""))
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        raise ArtifactError("voice audio must be registry-relative")
    audio = registry.parent / relative
    if audio.is_symlink() or not audio.is_file() or audio.suffix != ".wav":
        raise ArtifactError("voice reference must be a local regular WAV")
    if sha256_file(audio) != data.get("audio_sha256"):
        raise ArtifactError("voice reference hash mismatch")
    with wave.open(str(audio), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2 or stream.getframerate() != 24000:
            raise ArtifactError("reference must be mono PCM16 at 24 kHz")
        if not 24000 <= stream.getnframes() <= 30*24000:
            raise ArtifactError("reference duration must be between one and thirty seconds")
    data["resolved_audio_path"] = str(audio.resolve())
    return data
