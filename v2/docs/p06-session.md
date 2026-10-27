# Native static-reference conversation

The opt-in `native_static` backend now takes actual Qwen PCM through a bounded pipe into a per-epoch LiveKit C++ publisher. CUDA renders the approved canonical mesh's neutral projection at 512×512, converts RGBA to NV12 and feeds the existing NVENC surface owner. The normal `18700` preview retains `raw_reference`; V1 is unchanged. This is a functional static reference, not a speaking-avatar or quality qualification.

`server/native_media.py` verifies an operator-owned descriptor containing exact executable/PTX/LiveKit/FFI hashes and the pinned SDK revision. `scripts/describe_native_media.py` describes an already-built runtime and never builds or fetches inputs. Add `"media_backend": "native_static"` and an absolute `"native_media_runtime"` descriptor path to an operator-owned source configuration before `tha local init`. Profile initialization preserves that explicit selection. `local doctor` checks its runtime hashes. Do not modify an initialized profile's checked configuration in place.

## Clocks and ownership

The parent validates ordered 24 kHz mono Qwen samples, resamples with one FIR state per epoch and coalesces 960-sample 48 kHz packets. The 29-byte little-endian `<cQQQI` header carries data/final kind, epoch, sequence, source start and count. A final packet atomically carries the short tail and EOF. The worker rejects holes, overlaps, stale epochs, malformed quanta and input beyond 120 seconds; the Python adapter rejects fractional source EOF rather than silently rounding it.

One C++ CPU thread offers 480 audio samples every 10 ms; one other thread owns CUDA/render/encode. The PCM ring is bounded at 48,000 samples, with backpressure. Before speech, the worker offers explicit silent preparation until 300 ms is buffered or EOF arrives. Active underrun fails the source; it never deletes words or inserts an unreported speech gap. EOF counts valid PCM separately from partial-block padding and two 480-sample codec flush blocks. Silent preparation/post-EOF refresh is excluded from valid PCM. `first_pcm_stream_sample` maps the valid start to the source stream; startup remains part of real response delay.

Video uses the offered-audio source sample cursor, with a 1,600-sample tick. Late video may skip obsolete render slots; audio cannot. Frame metadata carries session, turn, epoch and source sample. Encoder feedback can change bitrate/encoder frame-rate parameters and request a keyframe without changing the render clock. Source, controller and browser clocks remain distinct; these checks do not infer acoustic presentation or lip sync.

Each epoch gets a new owned process and new audio/video publications. The worker inherits the actor's admitted device-lease descriptor. Priority SIGTERM bypasses a full PCM pipe. The parent waits for owned-process retirement, using SIGKILL only after the five-second cooperative deadline, before starting replacement publications. Startup cancellation retains the actual child handle; revocation during spawn cannot publish later. A native crash fails pending EOF rather than reporting success. A browser cancel acknowledgment still does not authorize resume: the existing exact-track gate requires the new SIDs. Per-epoch bootstrap, geometry and diagnostics are private and removed at retirement; qualification stderr remains privately retained. This checkpoint does not add a durable diagnostic archive.

## Actual checks

From the repository root, using the already configured pinned native build:

```sh
cmake --build v2/var/build/native-media --parallel 4
v2/.venv/bin/python v2/scripts/describe_native_media.py \
  --output v2/var/p06/native-media-runtime.json
v2/.venv/bin/tha local provision --root v2 --name p06-native
v2/.venv/bin/tha local up --root v2 --name p06-native
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  v2/runtimes/session-api/.venv/bin/python v2/scripts/qualify_native_worker.py \
  --config v2/var/local/p06-native/runtime.json \
  --directory v2/var/p06/worker-protocol-final \
  --report v2/benchmarks/reports/p06-native-worker.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  v2/.venv/bin/python v2/scripts/run_native_session_checks.py \
  --directory v2/var/p06/native-session-final
```

Evidence commands refuse to overwrite their destinations. Initialization used separate loopback ports: API 18720, PostgreSQL 15452, SFU signal/TCP/UDP 17900/17901/17902. The copied source selected existing cached providers and the pinned descriptor; it contains private credentials and is not published. From `v2/web/`, run the separate `playwright.native-session.firefox.config.ts` typed cohort. Preserve each browser evidence directory before running another cohort: the spec writes its private observations to `var/p06/native-session-browser/`. Then export with `scripts/qualify_native_session.py --directory .../native-session-final --report .../p06-native-session.json` and stop the temporary profile with `tha local down`. The normal preview remains available.

Actual results on 2026-10-27:

- GCC warnings-as-errors build passed. **260 core tests / 5.71 s**, **24 API/runtime tests / 3.96 s**, two web unit tests and TypeScript/build passed. Process tests include cancellation and external revocation while exec is pending, failed exec cleanup and unexpected exit before EOF.
- **11 actual SFU/GPU worker cases passed**: zero/one/479/480/481/960/961-sample final tails, wrong epoch, gap, wrong sequence and cancellation during producer backpressure. Every process retired; device admission reacquired. These are sender checks, separate from browser evidence.
- **Two Chromium checks / 36.38 s**, zero retries: typed interruption with new tracks and browser microphone → actual CPU ASR/LLM/Qwen → native video/audio. **One Firefox check / 17.03 s**, zero retries: typed Qwen/native playback, interruption and fresh tracks. All observed 512×512 video, presented-video callbacks, decoded audio and final valid/padding counts. These are browser graph observations, not physical speakers or an independent quality test.
- Completed Chromium answers contained **131,070 / 74,580 valid 48 kHz samples**, with **1,410 / 1,260** separately counted padding samples. Native queue peaks were **48,000 / 31,860 samples**, maximum observed source-clock lateness **2.62 / 3.61 ms**, and zero skipped video slots. Preparatory stream silence before valid speech was **123,840 / 168,480 samples**; it is explicitly not response audio. No subsecond response claim follows.
- **267 NVML samples**, target 100 ms between sequential device/process queries, no query failures. Total-device sampled maximum **6,451 MiB** includes non-PyTorch allocations and other processes. Qwen process sampled maximum **4,364 MiB**; native media **859 MiB**, exceeding the planned 512 MiB budget. Report the same-query combined maximum separately; do not sum independent maxima. These are sampled observations, not guaranteed peak or complete motion co-load/capacity qualification.

A host reboot during the gap stopped all project services. Existing databases/caches were retained; only this project's stacks were restored. API checks initially failed because PostgreSQL was down, then passed after restoration. Earlier browser attempts failed at an evidence assertion because the client omitted the transport field; both media paths had played. The field is now explicitly included. These failed artifacts remain private and are not counted as passing runs.

[Worker evidence](../benchmarks/reports/p06-native-worker.json), [conversation evidence](../benchmarks/reports/p06-native-session.json). The prior mesh trace establishes only its isolated encoder boundary; no new product-process no-readback trace is asserted here. Encoded payload CPU copies remain. Existing FFmpeg GPLv3+ and LiveKit/CUDA license boundaries are unchanged; no new dependency, model-weight download, training, human recording or cloud deployment occurred.

P06 remains in progress. Real Claire inference, approved speaking-avatar/retarget/Gaussian quality, independent Q0, human voice lifecycle, physical A/V/latency, API/inference images, live virtual output/third-party conference and fail-stop privacy remain open. The user has now authorized sourcing research/noncommercial avatar assets from the web; inspect provenance and actual terms before use, while retaining the model-weight download constraint until explicitly changed.
