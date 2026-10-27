# Full-size Claire GPU render qualification

Updated 2026-10-27. The source-matched vendor skin/tongue now renders through CUDA → NV12 → NVENC at 512×512. The direct path consumes actual Claire device tensors, with the recorded synthetic Qwen input from [neural qualification](p03-claire.md). This completes the current renderer checkpoint, not P03/P06 or the speaking-avatar product gate. The normal local preview remains the raw reference.

The separate research profile preserves **67,122 vertices / 134,236 triangles**, original component/vertex ordering and the explicitly derived source triangulation. It uses the declared centimeter-to-meter conversion and proper camera rotation. Small fixture/import limits are unchanged. Flat two-sided skin/tongue materials are a functional reference; source textures, eyeball/denture meshes, Gaussian appearance and independent quality acceptance remain absent. The rendered head was visually inspected at first/mid-utterance snapshots; those images and videos remain private.

One owner thread runs the SDK and encoder in the SDK's current CUDA context. FFmpeg's `AV_CUDA_USE_CURRENT_CONTEXT` borrows that context without changing its scheduling flags or destroying it. Separate producer events precede owned skin/tongue D2D copies; completion before returning the callback protects SDK tensor lifetime. Topology/material/camera state is immutable. Projection, counted bins, compact prefix offsets and tile filling precede rasterization. The 8-pixel tiles have a fixed **4,194,304-reference / 16 MiB** storage bound and **8,192-face per-tile** work limit. Overflow fails before rendering; no triangle deletion, index clamp or per-frame list growth is permitted. Twenty status bytes are explicitly read back per update. Existing RGBA/NV12 surface references and stream/encode fences remain in force.

A fixed 16-pixel / 4,096-entry list initially overflowed. The compact implementation also exposed a pageable host-reference upload race: default-stream DMA was not explicitly fenced against the nonblocking render stream. The recorded host path now synchronizes that upload; the direct path uses producer events and owned D2D copies. All failed artifacts are retained privately. The final workload peaks at **1,985 faces per tile / 262,658 references**, with the full geometry preserved.

Actual commands from the repository root:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  v2/runtimes/session-api/.venv/bin/python v2/scripts/prepare_claire_render_scene.py \
  --geometry-directory v2/var/p03/claire-geometry \
  --directory v2/var/p03/claire-render-scene
cmake -S v2/native/media -B v2/var/build/native-media \
  -DTHA_CLAIRE_SDK_BUILD="$PWD/v2/var/build/audio2face-sdk" \
  -DTHA_TENSORRT_ROOT="$PWD/v2/var/toolchains/tensorrt-10.13.3.9"
cmake --build v2/var/build/native-media --parallel 4
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/.venv/bin/python \
  v2/scripts/run_claire_render.py --mode direct \
  --engine v2/var/p03/claire-engine-fp32-n1-final \
  --scene v2/var/p03/claire-render-scene \
  --neural-directory v2/var/p03/claire-neural-final \
  --directory v2/var/p03/claire-direct-render-final
```

The configure command extends the already pinned P06 native build; [P06 setup](p06.md) records its full compiler/CUDA/LiveKit arguments. SDK ABI 0 is an explicit second encoder-library target; the existing standalone encoder ABI is preserved. Omit `--mode direct --engine ...` for the host-recorded reference. Evidence directories refuse overwrite. Each wrapper verifies input/engine/runtime provenance and holds the inherited managed device lease. No model download occurs here.

Actual validation:

- **84/84** encoded/decoded video frames at 30 Hz from 167 neural samples at 60 Hz. All 84 decoded YUV frame hashes match the host-recorded reference exactly; first/mid raw RGBA snapshots also match exactly. CPU/GPU camera projection differs by at most **2.27e-13 pixels / 2.22e-16 m** in depth. These are numerical/reference results, not a subjective avatar quality score.
- New GPU contracts pass **four owner lifetimes / ten successful frames**, including invalid indices/materials/camera/units, missing/invalid updates, equal-depth face identity, producer parts, foreign thread/context rejection, global/per-tile overflow fencing and borrowed-context survival. Existing private-context contracts still pass **three lifetimes / 270 frames / three failed initializations**. Core regression: **287 passed in 5.33 s**. First-party C++ warnings-as-errors and whitespace checks pass.
- A direct no-readback run was captured with `nsys profile --trace=cuda,nvtx --sample=none --cuda-memory-usage=true ... run_claire_render.py --mode direct --no-readback ...` and exported to SQLite. It contains 84 each of project/count/scan/fill/raster/NV12 kernels and 168 internal NVENC `Convert_PL2BL` calls. Observed CUDA D2H copies are only **4 bytes × 12, 20 bytes × 84 and 40 bytes × one**; no full geometry/image CUDA readback is observed in that process. Skin/tongue owned D2D copies are **738,240 / 67,224 bytes × 84** each. Initial model/topology uploads, ongoing audio uploads, small metadata transfers and encoded CPU/driver copies remain: this is not zero-copy. Nsight's event-completion instrumentation warning is retained; event dependencies are not interpreted as latency/concurrency evidence.
- The separate unpaced direct case reports ~**7.44 ms GPU raster service/frame** and **16.83 ms total geometry-copy/projection/bin service across 84 updates**. Service averages and instrumented observations are not end-to-end latency or capacity quantiles. It produced frames before EOF and retained the same bounded SDK audio state as neural qualification.
- Fifteen sequential NVML device/process samples, no query failures: **1,121 MiB** sampled total-device maximum and **541 MiB** combined SDK/render/NVENC worker maximum. This is a shared-context combined workload, not a matched isolated 512 MiB encoder-budget comparison, guaranteed loading peak or Qwen co-load/capacity pass.

[Structured evidence](../benchmarks/reports/p03-claire-render.json) binds source/binary/PTX/scene/trace identities. The `qualify_claire_render.py` exporter rejects failed/stale runs, mismatched full decoded video, camera errors, oversized D2H transfers, missing source D2D/kernel counts, contract failures and incomplete regression/memory evidence.

No further model/software downloads, training, V1 source/global settings or cloud deployment changed in this checkpoint. Live actor/LiveKit integration of this motion path, receiver-safe cancellation, physical A/V/latency, Qwen co-load, FLAME/Gaussian retarget, teeth/eye/texture quality, independent Q0 and the prior lifecycle/deployment gates remain open. The user next requested scoped V2 author/signature repair and main/V1 branch separation; perform that repository task before continuing live integration.
