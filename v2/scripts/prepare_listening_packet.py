"""Create private blinded, gain-matched reference/native pairs; no ratings invented."""

import hashlib
import json
from pathlib import Path
import random

import numpy as np
import soundfile as sf

root = Path("var/qualification")
reference = json.loads((root / "reference/reference-report.json").read_text())
if reference["voice_kind"] != "synthetic_fixture":
    raise SystemExit("this helper is scoped to the synthetic pilot, not human reference distribution")
output = root / "listening"
if output.exists():
    raise SystemExit("listening packet already exists; retain immutable pilot inputs")
output.mkdir()
rng = random.Random(17)
texts = json.loads(Path("configs/qwen/reference-texts.json").read_text())
rows, key = [], []
for index, request in enumerate(texts):
    providers = [("official_bf16", root / "reference" / f"reference-{index:03}.wav"),
                 ("native_f16_streaming", root / "native-waves" / f"native-{index:03}.wav")]
    rng.shuffle(providers)
    pair = {"case": request["id"], "expected_text": request["text"], "clips": []}
    for label, (provider, path) in zip(("A", "B"), providers):
        samples, rate = sf.read(path, dtype="float32")
        rms, peak = float(np.sqrt(np.mean(samples*samples))), float(np.max(np.abs(samples)))
        gain = min(.1/max(rms, 1e-9), .89/max(peak, 1e-9))
        name = f"case-{index:02}-{label}.wav"
        target = output / name
        sf.write(target, samples*gain, rate, subtype="FLOAT")
        pair["clips"].append({"label": label, "file": name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        key.append({"case": request["id"], "label": label, "provider": provider, "gain": gain,
                    "original_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    rows.append(pair)
packet = {"schema_version": 1, "voice_kind": "synthetic_fixture", "rated": False, "cases": rows,
          "instructions": "Rate each clip intelligibility/naturalness 1-5; note missing/repeated words, clicks and tail truncation. Prefer A, B or tie. Do not view the key before rating.",
          "limits": "One synthetic voice; cannot qualify human cloning or replace independent raters."}
(output / "packet.json").write_text(json.dumps(packet, indent=2)+"\n")
(output / "private-key.json").write_text(json.dumps(key, indent=2)+"\n")
print("Private blinded pilot packet created; ratings remain pending")
