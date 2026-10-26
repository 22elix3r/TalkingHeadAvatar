"""Official Qwen offline reference. Execute only in the locked inference environment."""

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from talkinghead_v2.assets.artifacts import read_manifest, sha256_file, verify_snapshot
from talkinghead_v2.assets.voices import read_voice

OFFICIAL_SOURCE = "022e286b98fbec7e1e916cb940cdf532cd9f488e"
# Match the pinned native tts_params controls, including its separate codebook-0
# and residual samplers. RNGs/dtypes/implementations still differ across runtimes.
NATIVE_MATCHED_SAMPLING = dict(do_sample=True, temperature=0.75, top_k=16, top_p=1.0,
    repetition_penalty=1.02, subtalker_dosample=True, subtalker_temperature=0.75,
    subtalker_top_k=16, subtalker_top_p=0.9)


def run(args) -> dict:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    manifest = read_manifest(args.manifest)
    voice = read_voice(args.voice, tenant_id=args.tenant)
    if args.mode == "icl" and not voice.get("reference_transcript", "").strip():
        raise ValueError("ICL requires an accurate authorized reference transcript")
    verification_start = time.perf_counter_ns()
    snapshot = verify_snapshot(args.cache, manifest)
    verification_ns = time.perf_counter_ns()-verification_start
    args.output.mkdir(parents=True, exist_ok=True)
    with redirect_stdout(sys.stderr):
        import numpy as np
        import soundfile as sf
        import torch
        from qwen_tts import Qwen3TTSModel

        torch.set_num_threads(4)
        torch.manual_seed(17)
        if not torch.cuda.is_available():
            raise RuntimeError("reference requires explicitly qualified CUDA; no silent CPU/dtype substitution")
        baseline_free, total = torch.cuda.mem_get_info()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter_ns()
        model = Qwen3TTSModel.from_pretrained(str(snapshot), device_map="cuda:0", dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True, use_safetensors=True, trust_remote_code=False)
        torch.cuda.synchronize()
        load_ns = time.perf_counter_ns()-started
        reference, rate = sf.read(voice["resolved_audio_path"], dtype="float32")
        prompt_start = time.perf_counter_ns()
        prompt = model.create_voice_clone_prompt(ref_audio=(reference, rate),
            x_vector_only_mode=args.mode == "x-vector",
            ref_text=voice.get("reference_transcript") if args.mode == "icl" else None)
        torch.cuda.synchronize()
        prompt_ns = time.perf_counter_ns()-prompt_start
        identity = {"tenant_id": args.tenant, "voice_id": voice["voice_id"], "reference_sha256": voice["audio_sha256"],
                    "model_id": manifest["model_id"], "model_revision": manifest["revision"],
                    "official_source": OFFICIAL_SOURCE, "torch": torch.__version__,
                    "conditioning": args.mode, "dtype": "bfloat16"}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        embedding = prompt[0].ref_spk_embedding.detach().float().cpu().flatten().tolist()
        embedding_path = args.output / "speaker-embedding.json"
        embedding_path.write_text(json.dumps(embedding)+"\n")
        requests = json.loads(args.texts.read_text())
        results = []
        for index, request in enumerate(requests):
            text = request["text"]
            if not isinstance(text, str) or not text.strip() or len(text.encode()) > 65536:
                raise ValueError("invalid reference text")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            waves, native_rate = model.generate_voice_clone(text=text, language="English", voice_clone_prompt=prompt,
                non_streaming_mode=False, **NATIVE_MATCHED_SAMPLING, max_new_tokens=args.max_tokens)
            torch.cuda.synchronize()
            elapsed_ns = time.perf_counter_ns()-start
            samples = np.asarray(waves[0], dtype=np.float32)
            if native_rate != 24000 or samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
                raise RuntimeError("official reference returned invalid native PCM")
            output = args.output / f"reference-{index:03}.wav"
            sf.write(output, samples, native_rate, subtype="FLOAT")
            free, total = torch.cuda.mem_get_info()
            results.append({"id": request["id"], "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "sample_rate": native_rate, "samples": int(samples.size), "wall_ns": elapsed_ns,
                            "rtf": elapsed_ns/1e9/(samples.size/native_rate), "wave_sha256": sha256_file(output),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "device_used_bytes": total-free,
                            "output": output.name, "streaming": False})
            (args.output / "progress.json").write_text(json.dumps(results, indent=2)+"\n")
    report = {"schema_version": 1, "evidence_category": "local_observation", "adapter": "official_offline_reference",
              "incremental_audio_output": False, "incremental_text_input": False,
              "identity": identity, "speaker_cache_key": key, "voice_kind": voice["kind"],
              "sampling": {**NATIVE_MATCHED_SAMPLING, "max_new_tokens": args.max_tokens, "seed": 17},
              "sampling_profile": "native_matched_controls_not_numerical_equivalence",
              "adapter_source_sha256": sha256_file(Path(__file__)),
              "human_cloning_qualified": False, "artifact_verification_ns": verification_ns,
              "model_load_ns": load_ns, "prompt_ns": prompt_ns, "baseline_device_used_bytes": total-baseline_free,
              "embedding": {"path": embedding_path.name, "dimension": len(embedding),
                            "sha256": sha256_file(embedding_path)}, "requests": results,
              "conditions": "new process; filesystem cache warm; CUDA bf16 SDPA; no concurrent inference worker"}
    (args.output / "reference-report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "cache", "voice", "output", "texts"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--mode", choices=("x-vector", "icl"), default="x-vector")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": "complete", "requests": len(report["requests"]),
                      "streaming": False, "voice_kind": report["voice_kind"]}))


if __name__ == "__main__":
    main()
