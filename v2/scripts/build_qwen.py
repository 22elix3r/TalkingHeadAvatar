"""Reproduce the pinned Fedora/CUDA native build in v2/var, with no global changes."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--cxx", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "native/qwen/source-lock.json").read_text())
    source = root / "var/sources/qwen3-tts-cpp-streaming"
    build = root / "var/build/qwen-cuda-overlay"
    bridge = root / "var/build/bridge"
    overlay = root / "var/toolchains/cuda-12.8-include"
    def command(argv, log=None):
        if log:
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w") as output:
                subprocess.run([str(x) for x in argv], stdout=output, stderr=subprocess.STDOUT, check=True)
        else:
            subprocess.run([str(x) for x in argv], check=True)
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        command(["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", lock["source_url"], source])
        command(["git", "-C", source, "checkout", "--quiet", "--detach", lock["source_revision"]])
    revision = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
    ggml = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD:engine/ggml"], text=True).strip()
    if revision != lock["source_revision"] or ggml != lock["ggml_tree"]:
        raise RuntimeError("source/GGML identity mismatch; do not build an unreviewed engine")
    pipeline = source / "engine/src/pipeline/pipeline_synthesize.cpp"
    if digest(pipeline) != lock["patched_pipeline_sha256"]:
        command(["git", "-C", source, "apply", "--check", root / "native/qwen/linux-player-metrics.patch"])
        command(["git", "-C", source, "apply", root / "native/qwen/linux-player-metrics.patch"])
    if digest(pipeline) != lock["patched_pipeline_sha256"]:
        raise RuntimeError("patched source differs from reviewed Linux fix")
    changed = subprocess.check_output(["git", "-C", source, "diff", "--name-only"], text=True).splitlines()
    if changed != ["engine/src/pipeline/pipeline_synthesize.cpp"]:
        raise RuntimeError("unexpected upstream source modifications")
    for path, expected in lock["converter_sha256"].items():
        if digest(source / path) != expected:
            raise RuntimeError("converter source identity mismatch")
    if not overlay.exists():
        command([sys.executable, root / "scripts/cuda_header_overlay.py", "--source",
                 args.cuda_root / "targets/x86_64-linux/include", "--output", overlay])
    command(["cmake", "-S", source, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
             f"-DCMAKE_CXX_COMPILER={args.cxx.resolve()}", f"-DCMAKE_CUDA_COMPILER={args.cuda_root.resolve()}/bin/nvcc",
             f"-DCMAKE_CUDA_HOST_COMPILER={args.cxx.resolve()}", f"-DCUDAToolkit_ROOT={args.cuda_root.resolve()}",
             f"-DCMAKE_CUDA_FLAGS=-I{overlay}", "-DQWEN3_TTS_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=89",
             "-DGGML_CUDA_GRAPHS=ON", "-DBUILD_TESTING=ON"], root / "var/build/configure-reproduced.log")
    command(["cmake", "--build", build, "--target", "qwen3_streaming_cli", "qwen3_wrapper_contract_test",
             "--parallel", str(args.jobs)], root / "var/build/build-reproduced.log")
    command([build / "tests/qwen3_wrapper_contract_test"])
    command(["cmake", "-S", root / "native/qwen", "-B", bridge, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
             f"-DQWEN_SOURCE={source}", f"-DQWEN_BUILD={build}"], root / "var/build/bridge-configure.log")
    command(["cmake", "--build", bridge], root / "var/build/bridge-build.log")
    evidence = {"source_revision": revision, "ggml_tree": ggml,
                "worker_sha256": digest(bridge / "tha_qwen_worker"),
                "bridge_source_sha256": digest(root / "native/qwen/worker.cpp"),
                "upstream_patch_sha256": digest(root / "native/qwen/linux-player-metrics.patch"),
                "cuda_header_original_sha256": digest(args.cuda_root / "targets/x86_64-linux/include/crt/math_functions.h"),
                "cuda_header_overlay_sha256": digest(overlay / "crt/math_functions.h"),
                "cuda_version": subprocess.check_output([args.cuda_root / "bin/nvcc", "--version"], text=True),
                "host_compiler": subprocess.check_output([args.cxx, "--version"], text=True).splitlines()[0]}
    (root / "var/build/qwen-build.json").write_text(json.dumps(evidence, indent=2)+"\n")
    print("Pinned native build and wrapper test passed; evidence is private under var/build/qwen-build.json")


if __name__ == "__main__":
    main()
