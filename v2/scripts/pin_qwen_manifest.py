"""Explicit metadata-only pinning; never downloads model weights. Run from v2/."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
data = json.load(urlopen(f"https://huggingface.co/api/models/{MODEL}?blobs=true", timeout=30))
revision = data["sha"]
entries = []
for item in data["siblings"]:
    name, size = item["rfilename"], item["size"]
    if name in {".gitattributes", "README.md"}:
        continue
    url = f"https://huggingface.co/{MODEL}/resolve/{revision}/{name}"
    digest = item.get("lfs", {}).get("sha256")
    if digest is None:
        if size > 10*1024**2:
            raise ValueError("large artifact without upstream SHA-256")
        content = urlopen(url, timeout=30).read()
        if len(content) != size:
            raise ValueError("metadata size mismatch")
        digest = hashlib.sha256(content).hexdigest()
    entries.append({"path": name, "size": size, "sha256": digest, "url": url,
                    "role": "speech_tokenizer" if name.startswith("speech_tokenizer/") else "tts_model"})
manifest = {"schema_version": 1, "model_id": MODEL, "revision": revision, "files": entries,
            "terms": {"weights": "Apache-2.0", "model_card": f"https://huggingface.co/{MODEL}/blob/{revision}/README.md",
                      "tokenizer": "Apache-2.0; bundled file identities recorded separately", "reference_voice": "not included"},
            "tokenizer_registry": {"model_id": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
                                   "revision": "7dd38ad4e9bad454aae9cd937d0cd577604fe229",
                                   "path": "speech_tokenizer", "weights_sha256": "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"}}
target = Path("configs/qwen/base-1.7b.manifest.json")
target.write_text(json.dumps(manifest, indent=2)+"\n")
print(f"Pinned {len(entries)} files, {sum(e['size'] for e in entries)} bytes; no weights fetched")
