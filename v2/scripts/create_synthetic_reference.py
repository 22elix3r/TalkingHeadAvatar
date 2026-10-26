"""Explicit eSpeak fixture creation, never a substitute TTS serving provider."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

root = Path("var/artifacts/tenants/synthetic/voices/espeak-reference")
root.mkdir(parents=True, exist_ok=True)
target = root / "reference.wav"
if target.exists():
    raise SystemExit("fixture already exists; refuse to overwrite its immutable reference")
text = "This is a synthetic reference for an audio pipeline test. It does not represent a real person's voice."
with tempfile.TemporaryDirectory(prefix="tha-reference-") as folder:
    raw = Path(folder) / "reference.wav"
    subprocess.run(["espeak-ng", "-v", "en-us", "-s", "155", "-w", str(raw), text], check=True)
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(raw),
                    "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(target)], check=True)
registry = {"schema_version": 1, "tenant_id": "synthetic", "voice_id": "espeak-reference",
            "kind": "synthetic_fixture", "status": "approved", "audio_path": target.name,
            "audio_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "reference_transcript": text, "allowed_uses": ["synthetic_plumbing_only"],
            "provenance": "Locally generated eSpeak NG en-us; not a human recording or identity",
            "tool_terms": "eSpeak NG code/data GPL-3.0-or-later; generated audio distribution clearance not asserted"}
(root / "voice.json").write_text(json.dumps(registry, indent=2)+"\n")
target.chmod(0o444)
print("Created private synthetic reference registry; no human cloning quality claim")
