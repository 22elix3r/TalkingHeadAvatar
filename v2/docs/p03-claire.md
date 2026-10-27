# Claire model, vendor geometry and actual GPU inference

Updated 2026-10-27. The user explicitly authorized sourcing research/noncommercial avatar assets, then authorized **the pinned Claire v2.3.1 package and its isolated TensorRT runtime** as an exception to the earlier weight-download ban. This exception does not authorize other TTS/ASR/LLM/emotion model downloads. Qwen voice polish remains deferred. V1, original checkpoints, global Python/driver settings and unrelated user changes are preserved.

## Sources and rights

- Model: [`nvidia/Audio2Face-3D-v2.3.1-Claire`](https://huggingface.co/nvidia/Audio2Face-3D-v2.3.1-Claire), revision `a46eafb067adfdd7cc8f5c7941586b309004561c`, 14 files / **307,466,971 bytes**. [Artifact inventory](../configs/motion/claire-artifacts.json) records exact URL/size/SHA-256 values, including the 159,044,772-byte ONNX network and numeric NPZ data. The pinned card links the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) and reports that training-data personal consent was obtained. That is a **vendor assertion**, not an independently inspected consent record or a new user's enrollment consent. Commercial release clearance is not established here.
- Avatar: [NVIDIA Maya-ACE sample project](https://github.com/NVIDIA/Maya-ACE/tree/f0dc13670952c495450dab1655c94e9056c9d7d9/sample_project), revision `f0dc13670952c495450dab1655c94e9056c9d7d9`. Its documented matching asset is `maya_blendshape_v2/claire_arkit_v2_topo1.ma`. [Source inventory](../configs/motion/claire-maya-source.json) pins the **78,070,108-byte** ASCII scene, root MIT license and sample-project compatibility README. These original vendor files are retained privately and immutable. Do not infer rights for external models, sample sounds or other datasets from the repository license. The separate NVIDIA training dataset has an evaluation-specific license and was not fetched.
- SDK: `NVIDIA/Audio2Face-3D-SDK`, revision `1ca0f02535ed774f5dbcd724a31cd486368dc783`, MIT source. Checkout used `GIT_LFS_SKIP_SMUDGE=1`; upstream model/test-data download helpers were not run. The software-only Packman dependency manifest explicitly excludes `audio2x-sdk-test-data`. Packman 7.31 runs with existing `/usr/bin/python3.12` and a project-local cache; its bundled Python/compiler/bootstrap paths and global caches are not used. This tested override does not claim that upstream's Python model-conversion helpers support Python 3.12.
- TensorRT: CUDA-12 **10.13.3.9** official NVIDIA wheel, 3,115,257,094 bytes, SHA-256 `bf2008eca911411aa93b852825ea992de33396451ea11713a6eb97c411d3b2e9`. The complete wheel was verified before extracting four Linux libraries and retaining the binary license. Its unused Windows builder resource and Python package files were omitted; the verified temporary wheel was then deleted. Public headers are pinned to TensorRT `94e2b9ef6d2cce74c76cdad499cca36cc4949197` and its ONNX parser submodule `9a9f7883dd7b8cb0a718395bac2075fab6f97da8`. Binary runtime terms and public header/code terms remain separate. No unknown/prebuilt engine was downloaded or loaded.

Provisioning is an explicit network operation. `verify` never fetches. Source-specific allowlists reject other repositories, revisions, paths and artifact types; exact hashes/byte limits and a disk reserve apply. Failed downloads cannot publish a serving snapshot. CPU tests cover corrupt/oversized/interrupted downloads, resumption from verified blobs, path escape, symlinks, extra files, offline mode and disk reservation. Model NPZ files are bounded by archive metadata and loaded with `allow_pickle=False`; object/nonfinite arrays are rejected. Maya ASCII is parsed as data: no MEL, Python, plugin, external reference or Maya executable is run.

## Geometry compatibility

The data-only reader preserves source vertices and oriented polygon edge loops. Quads are explicitly split on a fixed diagonal; triangle-to-source-polygon identity is retained. This is a named derived topology, not a claim of identical source triangulation. There is no decimation, face clamping, coefficient padding or silent vertex truncation. Textures/UV/color rendering are not implemented by this extractor.

Actual source/model comparison:

| Component | Matching scene node | Vertices | Source quads | Derived triangles | Maximum neutral-coordinate error |
|---|---|---:|---:|---:|---:|
| Skin | `neutralShapeOrig` | 61,520 | 61,518 | 123,036 | ~5.0e-9 cm |
| Tongue | `polySurfaceShape1_rest` | 5,602 | 5,600 | 11,200 | ~5.0e-9 cm |

`c_tongue_hiShapeOrig` has the same 5,602 vertices but differs from the model neutral by **0.29993 cm**; it is rejected. This demonstrates why equal dimensions do not establish a matching basis. Source coordinates are centimeters; conversion to meters is an explicit factor of 0.01. The small product fixture's 512-vertex/1,024-face native profile and bounded enrollment profile are preserved; these large assets have not been forced through them. A full-size renderer/asset profile is the next integration task.

## Build and run

From the repository root, after the explicit user authorization and with existing CUDA 12.8.93 / GCC 14.3.1 / CMake / Ninja:

```sh
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 v2/.venv/bin/python \
  v2/scripts/provision_claire.py provision --target both
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/.venv/bin/python \
  v2/scripts/provision_claire.py verify --target both
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 v2/.venv/bin/python \
  v2/scripts/provision_claire_runtime.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/.venv/bin/python \
  v2/scripts/build_claire_sdk.py --report v2/benchmarks/reports/p03-claire-runtime.json
cmake -S v2/native/motion -B v2/var/build/claire-motion -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DTHA_TENSORRT_ROOT="$PWD/v2/var/toolchains/tensorrt-10.13.3.9" \
  -DTHA_CLAIRE_SDK_BUILD="$PWD/v2/var/build/audio2face-sdk"
cmake --build v2/var/build/claire-motion --parallel 2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/.venv/bin/python \
  v2/scripts/build_claire_engine.py --directory v2/var/p03/claire-engine-fp32-n1-final
```

SDK source acquisition is documented in the build report. On a fresh software cache, use `build_claire_sdk.py --provision-dependencies` deliberately before the offline build; it preserves the tested package versions and omits test data/models. `build_claire_sdk.py` rejects modified tracked upstream source, verifies every TensorRT file, uses only architecture 89 and the previously qualified local CUDA include overlay, and builds target `audio2x`. It records actual package-tree and linked-library hashes. Upstream CMake helpers print misleading TensorRT `?` / magic_enum `5.5.5` version diagnostics; pinned header values and library versions are checked separately. Full dataset-dependent upstream tests have not passed and are not claimed.

The engine builder checks matching TensorRT headers/runtime, fixes batch min/opt/max to **one**, disables TF32, sets 256 MiB workspace and 48 KiB tactic shared-memory limits, and bounds builder CPU threads to four. The wrapper verifies source/runtime hashes and takes the existing managed device lease, inherited by the child. It writes only a new private engine and descriptor. The final engine is **160,615,948 bytes**, built locally in **28.02 s**. Its manifest binds original model hashes, builder/SDK/runtime hashes, GPU identity and driver. It has no shared-weight pool or capacity claim.

The geometry oracle uses the previously recorded synthetic-reference Qwen short request: 66,645 samples at 24 kHz, verified against its P02/P03 waveform provenance. The core stateful FIR maps it to exactly **44,430 samples at 16 kHz**. No new TTS inference, human recording or model download happens during these offline checks. The private input record pins both source and target hashes. Run:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  v2/runtimes/session-api/.venv/bin/python v2/scripts/qualify_claire_assets.py \
  --directory v2/var/p03/claire-geometry \
  --report v2/benchmarks/reports/p03-claire-assets.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  v2/runtimes/session-api/.venv/bin/python v2/scripts/qualify_claire_neural.py \
  --engine v2/var/p03/claire-engine-fp32-n1-final \
  --directory v2/var/p03/claire-neural-final \
  --report v2/benchmarks/reports/p03-claire-neural.json
```

Prepare a fresh copy of the exact recorded source input with the existing pinned Qwen reference environment (it imports the audio-file reader, not the model):

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  v2/runtimes/qwen-reference/.venv/bin/python v2/scripts/prepare_claire_input.py \
  --wave v2/var/qualification/native-waves-serial/native-000.wav \
  --tts-report v2/benchmarks/reports/p02-native-cohort.json --case short \
  --directory v2/var/p03/claire-qwen-input
```

The script reproduced both the original private input metadata and PCM **exactly**, using 997-sample source quanta instead of the initial whole-source push. It checks the original request PCM hash and declared synthetic voice before conversion. Evidence destinations refuse overwrite; use fresh paths for new runs. The private recorded input was prepared with the existing `soundfile` reader and core FIR, with its source WAV hash checked again by the neural qualifier. Qualification sources, binary hashes and all command results are in [asset](../benchmarks/reports/p03-claire-assets.json), [runtime](../benchmarks/reports/p03-claire-runtime.json) and [neural](../benchmarks/reports/p03-claire-neural.json) evidence.

## Actual neural evidence and limits

Five real GPU runs passed: whole-input reference, 320-sample/20 ms chunks, a 512-sample grid, reset at 16,000 offered samples and a fresh process after reset. Every complete run produced **167 frames**, 152 before EOF and 15 after explicit closure. Source PTS is exactly `floor(frame_index*16000/60)`; the last valid target is sample 44,266. No target is in padded EOF audio. All complete chunked/fresh runs are **byte-identical** to the reference across skin, tongue, jaw and eye outputs; measured maximum/RMS error is zero. Reset bounded output to 45 frames, and owned processes retired before admission was reacquired. This is not live priority-interruption or browser-buffer suppression evidence.

First output required 4,160 input samples in 20 ms mode (the real **260 ms future context**), or 4,608 samples on the coarser grid. The SDK's actual retained-audio peaks were **24,320 / 24,576 samples**, larger than the mathematical context because its pool drops whole buffers. The qualifier enforces a pool-plus-context-plus-quantum bound. The whole-input reference retains its complete short input by design. These are unpaced functional partition checks, not measured TTFA, streaming FPS, physical A/V, system p95 or quality scores.

Each result explicitly copies full geometry/pose data to CPU and synchronizes its producer streams before buffer reuse. This is qualification readback, not the native serving path and not zero-copy. Private geometry dumps contain the vendor sample and stay out of public reports. The final cohort sampled device memory **51 times**, no query failures, with total-device maximum **990 MiB**, including other processes. This is a sampled observation, not a guaranteed loading peak or Qwen/render/encode co-load result.

Offline core validation: **287 passed in 5.46 s** (including 13 artifact and 14 Maya geometry checks); C++ probes compile with first-party warnings-as-errors. The original preview remains ready. P03/P06 remain in progress: rendered articulation and teeth, explicit GA/FLAME retarget/calibration, live actor integration, cancellation/presentation, Qwen+motion+render/encode co-load, independent Q0, human voice/enrollment, deployment images and release gates are still open. A vendor mesh/neural geometry success is not a photorealistic avatar winner or a completed product.
