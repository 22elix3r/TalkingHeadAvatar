"""Explicit pinned ASR evaluator provisioning; not a serving ASR selection."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REPO = "openai/whisper-base.en"
REVISION = "911407f4214e0e1d82085af863093ec0b66f9cd6"
data = json.load(urlopen(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true", timeout=30))
root = Path("var/evaluators/whisper-base.en")
root.mkdir(parents=True, exist_ok=True)
entries = []
for item in data["siblings"]:
    name = item["rfilename"]
    if not name.endswith((".json", ".txt", ".safetensors")):
        continue
    if "/" in name or name.startswith("."):
        raise ValueError("unexpected evaluator path")
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
    target = root / name
    if target.exists():
        raise SystemExit("evaluator cache already exists; verify its manifest instead of overwriting")
    digest, count = hashlib.sha256(), 0
    with urlopen(url, timeout=60) as source, target.open("xb") as output:
        while block := source.read(1024*1024):
            output.write(block); digest.update(block); count += len(block)
    if count != item["size"] or (item.get("lfs") and digest.hexdigest() != item["lfs"]["sha256"]):
        raise ValueError("evaluator size/hash mismatch")
    if not item.get("lfs"):
        blob = b"blob "+str(count).encode()+b"\0"+target.read_bytes()
        if hashlib.sha1(blob).hexdigest() != item["blobId"]:
            raise ValueError("evaluator Git blob identity mismatch")
    target.chmod(0o444)
    entries.append({"path": name, "size": count, "sha256": digest.hexdigest(), "url": url})
manifest = {"schema_version": 1, "role": "offline_quality_evaluator_not_serving_asr",
            "model_id": REPO, "revision": REVISION, "files": entries,
            "terms": {"hf_model_card": "Apache-2.0", "original_openai_code": "MIT",
                      "transformers_runtime": "Apache-2.0"},
            "sources": [f"https://huggingface.co/{REPO}/blob/{REVISION}/README.md",
                        "https://github.com/openai/whisper/blob/main/LICENSE"]}
Path("configs/quality/whisper-base.manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
print("Pinned evaluator files verified; no remote model code or pickle loaded")
