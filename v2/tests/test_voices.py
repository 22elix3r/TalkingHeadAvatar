import hashlib
import json
import wave

import pytest

from talkinghead_v2.assets.artifacts import ArtifactError
from talkinghead_v2.assets.voices import read_voice


@pytest.fixture
def registry(tmp_path):
    audio = tmp_path / "synthetic.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(b"\0\0"*24000)
    data = {"schema_version": 1, "tenant_id": "tenant-a", "voice_id": "synthetic",
            "status": "approved", "kind": "synthetic_fixture", "audio_path": audio.name,
            "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
            "allowed_uses": ["synthetic_plumbing_only"]}
    path = tmp_path / "voice.json"
    path.write_text(json.dumps(data))
    return path, data


def test_registry_scopes_synthetic_voice_and_rejects_cross_tenant(registry):
    path, _ = registry
    voice = read_voice(path, tenant_id="tenant-a")
    assert voice["kind"] == "synthetic_fixture"
    with pytest.raises(ArtifactError, match="another tenant"):
        read_voice(path, tenant_id="tenant-b")


@pytest.mark.parametrize("change", [
    {"status": "revoked"}, {"kind": "human_consented"}, {"audio_sha256": "0"*64},
    {"allowed_uses": ["commercial"]}, {"audio_path": "../synthetic.wav"},
])
def test_voice_requires_current_approval_rights_and_unchanged_bytes(registry, change):
    path, data = registry
    data.update(change)
    path.write_text(json.dumps(data))
    with pytest.raises(ArtifactError):
        read_voice(path, tenant_id="tenant-a")
