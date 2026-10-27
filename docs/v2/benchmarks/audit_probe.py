"""Read-only V1 audit microbenchmarks; no downloads, training, playback or rendering.

Run from the repo: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3
  docs/v2/benchmarks/audit_probe.py --output /tmp/avatar-audit.json
Synthetic inputs measure execution only, never animation quality or conversation latency.
"""
from __future__ import annotations
import argparse
import contextlib
import gc
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from audio_driver.motion_translator import MotionTranslator
    from audio_driver.audio_encoder import AudioEncoder
    from audio_driver.inference import StreamingAudioBuffer
    torch.manual_seed(1234)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    report = {"kind": "isolated synthetic microbenchmark, not end-to-end", "date": "2026-10-25",
              "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "cpu_threads": 2, "warmup": 10, "samples": 100,
              "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv"], text=True),
              "dependencies": {}, "checkpoints": {}, "measurements": {}}
    for package in ("transformers", "torchaudio", "chatterbox-tts", "numpy", "pytest", "llama-cpp-python", "pyvirtualcam", "diff-gaussian-rasterization", "simple-knn"):
        try: report["dependencies"][package] = metadata.version(package)
        except metadata.PackageNotFoundError: report["dependencies"][package] = "not installed as distribution"
    for path in sorted((ROOT / "audio_driver/checkpoints").glob("*/best_model.pt")):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            result = {"config": ckpt.get("config"), "bytes": path.stat().st_size}
            try:
                with contextlib.redirect_stdout(io.StringIO()): MotionTranslator.load(path, device="cpu")
                result["current_loader"] = "success"
            except Exception as exc: result["current_loader"] = f"{type(exc).__name__}: {exc}"
            report["checkpoints"][str(path.relative_to(ROOT))] = result
            del ckpt
        except Exception as exc: report["checkpoints"][str(path.relative_to(ROOT))] = {"error": str(exc)}
    buffer = StreamingAudioBuffer(chunk_size=12800)
    report["oversize_warmup_buffer_shape"] = list(buffer.push(torch.zeros(16000)).shape)
    try:
        dist = metadata.distribution("chatterbox-tts")
        source = Path(dist.locate_file("chatterbox/tts.py")).read_text()
        report["installed_chatterbox_original_generate_stream_definition"] = "def generate_stream(" in source
    except Exception as exc: report["chatterbox_source_error"] = str(exc)
    if torch.cuda.is_available():
        def measure(name, fn):
            with torch.inference_mode():
                for _ in range(10): fn()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                timings = []
                for _ in range(100):
                    start = time.perf_counter()
                    fn()
                    torch.cuda.synchronize()
                    timings.append((time.perf_counter() - start) * 1000)
            report["measurements"][name] = {"wall_ms_p50_p95_p99": np.percentile(timings, [50,95,99]).tolist(),
                "peak_allocated_MiB": torch.cuda.max_memory_allocated()/2**20,
                "peak_reserved_MiB": torch.cuda.max_memory_reserved()/2**20}
        model = MotionTranslator(causal=True).eval().cuda()
        report["translator_parameters"] = sum(p.numel() for p in model.parameters())
        features = torch.zeros((1,49,1024), device="cuda")
        measure("untrained_current_translator_fp32_49_features", lambda: model(features))
        del model, features
        gc.collect(); torch.cuda.empty_cache()
        frame = torch.rand((3,512,512), device="cuda")
        measure("v1_frame_export_float32_cpu_then_u8", lambda: (frame.permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8))
        measure("comparison_gpu_u8_then_cpu", lambda: (frame.permute(1,2,0).clamp(0,1)*255).to(torch.uint8).contiguous().cpu().numpy())
        del frame
        gc.collect(); torch.cuda.empty_cache()
        encoder = AudioEncoder(device="cuda", fp16=True)
        try:
            start = time.perf_counter()
            encoder._ensure_loaded()
            torch.cuda.synchronize()
            report["hubert_cached_load_seconds"] = time.perf_counter() - start
            for seconds in (0.8,1.0):
                waveform = torch.zeros((1,int(16000*seconds)))
                measure(f"cached_hubert_fp16_{seconds}s_cpu_input", lambda: encoder(waveform))
        except Exception as exc: report["hubert_error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
