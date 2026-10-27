# P06 static native mesh qualification

The native encoder now rasterizes bounded projected canonical meshes on CUDA before BT.709 NV12 conversion and NVENC. It preserves neutral pose, camera projection, vertex/face ordering and input hashes; it rejects invalid indices, nonfinite/nonpositive-depth coordinates, excess geometry and mutation after ownership begins. It never pads/truncates vertices or repairs face indices. This is a **static reference**, not neural motion, Gaussian appearance or a real-avatar quality pass. The product API still uses the raw reference until native session integration is qualified.

`rendering/projected.py` defines a little-endian `THAMESH1` format bounded to 512 vertices, 1,024 triangles and even output dimensions through 512. The native profile is explicitly **512×512**: actual NVENC initialization rejected 128×128 on this device/configuration. Output resolution is selected before projection; geometry/topology/camera conventions are not changed to bypass that limit.

One GPU owner retains immutable double-precision projected vertices, exact face indices and derived pixel bounds. Initial H2D uploads complete before the nonblocking render stream starts. Pixel bounds follow the CPU rasterizer's floor/ceil clipping convention; only covered triangles incur barycentric/depth work. The existing explicit render → conversion → encoder dependencies and three retained input slots remain. A qualification-only readback method is named separately; publishing does not invoke it.

Actual local deterministic checks, with the shared device lease held:

- Packaged procedural neutral mesh: GPU RGBA matches the CPU rasterizer's RGB byte-for-byte at 512×512; all alpha bytes are 255. This tests the same neutral pose/camera/material, not identity or photorealism.
- First-party synthetic grid: 512 vertices / 930 triangles match a separate rectangular-grid pixel oracle exactly. This is a structural/performance fixture with no face/model/voice asset.
- Both encode/decode 60 H.264 frames, Constrained Baseline, 512×512, BT.709 limited range. Pixel parity is checked **before lossy NV12/H.264 coding**, not claimed for decoded RGB.
- Three GPU owner lifetimes / 270 frames and three failed initializations pass, including immutable mesh binding, invalid/duplicate indices, invalid depth, foreign-thread/PTS rejection and forced keyframes.
- **248 offline CPU checks passed in 5.17 s**, including seven projection/serialization/bounds checks. No inference dependency is added to the core.

Dense-grid development observations improved from approximately **42.75 ms/render** without triangle bounds to approximately **0.35 ms/render** with them. These are isolated unpaced 60-frame GPU-service averages from development versions, not a formal source-pinned latency comparison, quantiles, admitted capacity or conversational latency. Cold context/encoder initialization is excluded, and Qwen/motion co-load is not measured here. The final source/binary/PTX hashes and actual final averages are in the [report](../benchmarks/reports/p06-native-mesh.json).

The parity probe intentionally copies one complete **1 MiB RGBA frame to host per case**. A separate `--no-readback` trace contains 60 mesh renders, 60 NV12 conversions and 120 internal NVENC conversion kernels; three initial H2D transfers upload vertices (12,288 bytes), faces (11,160 bytes) and pixel bounds (14,880 bytes). No copy follows the first kernel and no D2H activity is observed in that isolated CUDA-visible boundary. Encoded CPU payload copies remain. This is not zero-copy transport or a claim about another process.

From the repository root, using the existing qualified CUDA/SDK build configuration:

```sh
cmake --build v2/var/build/native-media --parallel 4
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/.venv/bin/python -m pytest -q \
  --rootdir=v2 v2/tests --junitxml=v2/var/p06/mesh-core-tests.xml
```

`tha_mesh_probe kernels.ptx projected-mesh.bin rgba-output h264-output [--no-readback]` is the actual built native probe, under `v2/var/build/native-media/`. New/private outputs are mandatory. `scripts/qualify_native_mesh.py --directory <new-private-directory> --trace <Nsight-SQLite-export> --report <new-report>` runs both parity oracles, decoder checks and GPU owner contracts, and rejects incomplete CPU checks or an empty/wrong/copying trace. The actual command used is recorded in implementation status. It acquires/inherits the managed device-0 lease rather than competing with an active conversation. No fixture/report overwrite or model provisioning is performed.

The generated grid, geometry blobs, decoded pixels and elementary streams stay private. Source meshes/checkpoints are unchanged. Existing GPU SDK/driver/FFmpeg license and resource limits still apply; no dependency/weight download, training or driver change occurred. Native Qwen/actor/epoch/PCM integration, physical A/V, real motion/appearance, co-load/loading peaks and the existing encoder-memory planning miss remain open P06/release gates.
