"""Export synthetic-only metrics, never model weights, raw PCM or voice embeddings."""

import hashlib
import json
from pathlib import Path

root = Path("var/qualification")
public = Path("benchmarks/reports")
public.mkdir(parents=True, exist_ok=True)
for source, target in (
    (root / "reference/reference-report.json", "p02-official-reference-initial.json"),
    (root / "reference-native-matched/reference-report.json", "p02-official-reference.json"),
    (root / "native-cohort.json", "p02-native-cohort-graphs-async.json"),
    (root / "native-cohort-serial.json", "p02-native-cohort.json"),
    (root / "intelligibility.json", "p02-intelligibility-graphs-async.json"),
    (root / "intelligibility-serial.json", "p02-intelligibility-initial-reference.json"),
    (root / "intelligibility-matched.json", "p02-intelligibility.json"),
    (root / "device-calibration.json", "p02-device-calibration.json"),
    (root / "native-soak-fixed.json", "p02-native-soak-graphs-async.json"),
    (root / "native-soak-serial.json", "p02-native-soak.json"),
    (root / "local-speech.json", "p02-local-speech.json"),
    (root / "local-interruption.json", "p02-local-interruption.json"),
    (root / "segments.json", "p02-segments.json"),
):
    if not source.exists():
        continue
    data = json.loads(source.read_text())
    if data.get("voice_kind", "synthetic_fixture") != "synthetic_fixture":
        raise SystemExit("refuse automatic publication of human voice reports")
    (public / target).write_text(json.dumps(data, indent=2)+"\n")
failed = json.loads((root / "native-soak-failed.progress.json").read_text())
runtime = json.loads((root / "native-soak-failed.runtime.json").read_text())
failure = {"schema_version": 1, "evidence_category": "local_observation", "phase": "P02",
           "observed_duration_seconds": failed["elapsed_seconds"], "completed_requests": failed["completed"],
           "failures": failed["failures"], "stability_30_minutes_passed": False,
           "status": "interrupted_after_failure", "worker_sha256": runtime["worker_sha256"],
           "notes": ["Original run had one invalid protocol frame on epoch 184; exact emitting bytes were not retained",
                     "Bridge shared stdout with libraries; replacement uses a dedicated descriptor and redirects library stdout",
                     "Only intermediate aggregate/last-request progress survived this interrupted first run; new runs persist every request"]}
(public / "p02-initial-soak-failure.json").write_text(json.dumps(failure, indent=2)+"\n")
runtime = json.loads(Path("var/native-runtime-serial.json").read_text())
voice = json.loads(Path(runtime["voice_registry"]).read_text())
if voice["kind"] != "synthetic_fixture":
    raise SystemExit("refuse automatic export of human voice/embedding metadata")
safe = {name: runtime[name] for name in ("backend", "worker_sha256", "derived_models", "embedding_sha256",
        "embedding_dimension", "runtime_revision", "ggml_tree", "upstream_patch_sha256",
        "disable_cuda_graphs", "async_decode")}
safe["reference_voice_kind"] = "synthetic_fixture"
safe["runtime_lock_sha256"] = hashlib.sha256(Path("runtimes/qwen-reference/uv.lock").read_bytes()).hexdigest()
safe["bridge_source_sha256"] = hashlib.sha256(Path("native/qwen/worker.cpp").read_bytes()).hexdigest()
safe["sampling_controls"] = {"temperature": 0.75, "top_k": 16, "codebook_0_top_p": 1.0,
    "residual_top_p": 0.9, "repetition_penalty": 1.02, "seed_per_request": 17, "max_audio_tokens": 1024,
    "provenance": "pinned engine/src/qwen3_tts.h defaults plus checked-in bridge overrides"}
safe["build_reproduction"] = json.loads(Path("var/build/qwen-build.json").read_text())
(public / "p02-native-build.json").write_text(json.dumps(safe, indent=2)+"\n")
print("Exported available synthetic metrics only; raw audio/embeddings/models remain private")
