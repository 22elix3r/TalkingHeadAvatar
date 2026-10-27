"""Bind locally built/converted files and the approved reference into a private descriptor."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from talkinghead_v2.assets.artifacts import sha256_file

root = Path.cwd().resolve()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference-report", type=Path, default=root / "var/qualification/reference/reference-report.json")
parser.add_argument("--voice-registry", type=Path, default=root / "var/artifacts/tenants/synthetic/voices/espeak-reference/voice.json")
parser.add_argument("--output", type=Path, default=root / "var/native-runtime.json")
parser.add_argument("--disable-cuda-graphs", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--async-decode", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()
reference = json.loads(args.reference_report.read_text())
if reference["identity"]["conditioning"] != "x-vector":
    raise SystemExit("this streaming runtime supports X-Vector only; no silent ICL substitution")
worker = root / "var/build/bridge/tha_qwen_worker"
models = root / "var/models"
embedding = args.reference_report.parent / reference["embedding"]["path"]
descriptor = {"schema_version": 1, "backend": "qwen_cpp_cuda_f16", "worker": str(worker),
    "worker_sha256": sha256_file(worker), "models": str(models),
    "derived_models": {p.name: sha256_file(p) for p in models.iterdir()},
    "embedding": str(embedding), "embedding_sha256": sha256_file(embedding),
    "embedding_dimension": reference["embedding"]["dimension"], "reference_identity": reference["identity"],
    "tenant_id": reference["identity"]["tenant_id"], "voice_registry": str(args.voice_registry.resolve()),
    "manifest": str(root / "configs/qwen/base-1.7b.manifest.json"), "cache": str(root / "var/artifacts"),
    "runtime_revision": "74422bab4998ad69e0a67ba105654dadbeec9b3a+linux-player-metrics-v1",
    "ggml_tree": "b1e00d0467ff3e8a7ade3bc6031b756b4cfe0c13",
    "upstream_patch_sha256": sha256_file(root / "native/qwen/linux-player-metrics.patch"),
    "disable_cuda_graphs": args.disable_cuda_graphs, "async_decode": args.async_decode}
target = args.output
target.write_text(json.dumps(descriptor, indent=2)+"\n")
print("Pinned private local native runtime/derived models/voice; no binaries or embeddings published")
