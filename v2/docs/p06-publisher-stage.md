# Pre-encoded publisher handoff

Local observation, 2026-10-28. Per-frame SDK hooks now locate the observed
source-frame gaps between the accepted/uploaded frame and WebRTC's output
transformer. An isolated, default-off external-rate-controller selection
conserved every source frame in four measured trusted-controller turns.
This is a local candidate correction, not a serving rollout or a completed
P03/P06 delivery, latency, quality or release gate.

## Boundary evidence

The first trace used the pinned SDK revision
`bae4df2f3d6c8f0f6ce81a7370c445779f1948d2` and a trace-only overlay. NVENC
had already produced the access units before source capture. The SDK hook
named `EncoderUpload` therefore describes the downstream WebRTC path, not
upload to the NVIDIA hardware encoder.

All **324** source captures reached that hook. WebRTC's `EncoderOutput` and
`WebrtcPacketize` hooks each observed **323**. Missing frame ID **245** maps
exactly to source index **224**, also missing at the receiver. The hook named
`WebrtcPacketize` observes a handoff to the packetizer; it does not acknowledge
RTP transmission or physical presentation. All counts apply to the explicitly
sealed observation window after capture stops and sender counters settle.

The passthrough encoder left `has_trusted_rate_controller` false. WebRTC's
[encoder API](https://webrtc.googlesource.com/src/+/refs/heads/main/api/video_codecs/video_encoder.h)
documents that enabling this property disables the media optimizer's input
frame dropper and makes the encoder responsible for matching the target rate.
The locally pinned header documents the same contract, and its hash is recorded
in the build evidence. The external NVENC path already consumes SDK bitrate/FPS
feedback. The candidate selects trusted control only when the exact process
flag `THA_LK_TRUST_EXTERNAL_RATE_CONTROL=1` is present.

Three alternating pairs used the **same patched SDK and worker**, changing
only that flag:

| Mode | Accepted / upload | Output / packetizer / receiver |
|---|---|---|
| SDK default, turn 1 | 324 / 324 | 323 / 323 / 323 |
| External trusted, turn 1 | 324 / 324 | 324 / 324 / 324 |
| SDK default, turn 2 | 324 / 324 | 323 / 323 / 323 |
| External trusted, turn 2 | 324 / 324 | 324 / 324 / 324 |
| SDK default, turn 3 | 324 / 324 | 319 / 319 / 319 |
| External trusted, turn 3 | 324 / 324 | 324 / 324 / 324 |

Every missing source index matched across the output hook, packetizer hook
and receiver. A final pair with the current interruption-safe runner observed
**319/324** in default mode and **324/324** in trusted mode, again with identical
missing-index sets across those boundaries. All eight comparison processes
exited 0 without panic. This supports the input frame-dropper explanation for
these observed gaps; it does not establish that all possible delivery losses
share that cause.

Trusted-turn source payload averages in the three-pair cohort were
**1.1925, 1.2057 and 1.2333 Mbps**, against the local profile's 1.2 Mbps
publish setting. These are whole-source payload averages, not RTP overhead,
peak rates, pacer queue delay, downshift response or constrained-network
qualification. No network-rate or latency acceptance follows from them.

## Isolated SDK overlays

Both overlays apply directly to the cached base revision; do not stack them:

- [Trace overlay](../patches/livekit-publish-stage-trace.patch), SHA-256
  `5aca3d55bcf694000757e675240d4a23df03590953da60aa7fe2638aca0604c6`.
- [Trace plus opt-in rate-control overlay](../patches/livekit-publish-rate-control.patch), SHA-256
  `a08f5363be5ff8166b495c21e738d0647f022d774fec48e070b398a297211cd7`.

The trace exposes existing SDK timing hooks through an isolated local C ABI,
using a fixed **16,384-row** buffer and at most eight handles. Each worker
qualifies one controlled published track. Callback rows contain only stage,
frame identity, application PTS and SDK Unix time. They contain no session/track
IDs, images, coefficients or speech. Unknown/non-source identities stay counted
separately. Overflow, truncation, duplicate source stages and source-PTS mismatch
reject the observation. Application PTS is not subtracted from the Unix clock.

The direct observer avoids the SDK timing stream's documented silent-overrun
behavior. Snapshot sealing defines the end of observation; late callbacks
cannot modify copied records. It does not turn a stage hook into a delivery
acknowledgment. Native ABI checks reject missing symbols, a different loaded
library and an invalid track before a probe can continue.

The original SDK checkout, serving microphone runtime and pinned preview
binaries remain unchanged. The publisher probe uses a separate SDK/library
snapshot; it cannot be selected as the fail-closed microphone runtime. The
serving adapter removes diagnostic and trust flags from inherited environments.

## Build and reproduction

`build_publish_trace_ffi.py` checks the complete SDK diff, existing Cargo lock,
clean pinned submodules and selected Rust toolchain before and after building.
The observed protocol/libyuv revisions are
`9a7c5cf96bbbcbc30a8ce76bca16815ba5b7bb33` and
`917276084a49be726c90292ff0a6b0a3d571a6af`. It uses cached WebRTC/protoc,
`--locked --offline`, an isolated Cargo cache and existing Rust 1.99.0; no
Rustup, global dependency or driver change is performed.

The initial offline test builds identified five missing Rust test crates.
Only their existing lockfile versions/checksums were provisioned into the
isolated cache: hmac 0.12.1, jsonwebtoken 10.4.0, rand_core 0.6.4, sha2 0.10.9
and signature 2.2.0, totaling **175,426 bytes**. Those are software archives,
not model weights. The failed offline attempts remain private. Subsequent
trace and rate-control builds passed their five selected native tests offline.

The actual rate-control SDK build, from the repository root, was:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/.venv/bin/python \
  v2/scripts/build_publish_trace_ffi.py --variant rate-control \
  --source v2/var/sources/livekit-ffi-publish-rate-control-20261028 \
  --target v2/var/build/livekit-ffi-publish-rate-control-20261028 \
  --cargo-home v2/var/p06-claire/cargo-publish-rate-control \
  --webrtc v2/var/sources/livekit-cpp-v1.11.0/client-sdk-rust/target/release/build/scratch-7323677a5bf9c8a5/out/livekit_webrtc/livekit/linux-x64-release-webrtc-89d790b/linux-x64-release \
  --protoc v2/var/build/livekit-cpp/bin/protoc-25.3.0 \
  --rust-sysroot /home/elix3r/.rustup/toolchains/stable-x86_64-unknown-linux-gnu \
  --report v2/var/p06-claire/publish-rate-control-ffi-build-1.json
cmake --build v2/var/build/native-media \
  --target tha_claire_session_media tha_publish_trace_contract --parallel 2
```

The rate-control library SHA-256 is
`084e4f30b6748b3c2434eafedbef142856b56bbb8cc0d97c5353992dcb1e1188`.
The earlier trace-only library is
`c2eecb75610a48317cda942a841b974ee45fb4db2948b8438921d6455071017d`;
its report identifies the preceding helper revision. The native ABI contract
passed against the checksum-verified experimental library without a room or
model. `describe_publish_trace.py` snapshots the worker, SDK libraries and
build receipt into a private descriptor; it does not provision or fetch them.

The rate-control probe descriptor is
`e0726d09c2dc243a75fa63645bd18c9db1d465f93bda4799c47c9e64480b3968`.
The final comparison command used new output paths:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/runtimes/session-api/.venv/bin/python \
  v2/scripts/qualify_claire_publisher_matrix.py --repeats 1 \
  --publisher-runtime v2/var/p06-claire/publisher-rate-control-runtime-1.json \
  --receiver-runtime v2/var/p06-claire/inbound-runtime-reason-v1.json \
  --config v2/var/p04/test-runtime.json --engine v2/var/p03/claire-engine-fp32-n1-final \
  --scene v2/var/p03/claire-render-scene \
  --directory v2/var/p06-claire/publisher-rate-control-matrix-fenced-final-20261028 \
  --report v2/var/p06-claire/publisher-rate-control-matrix-fenced-final-20261028.json
```

The earlier three-pair run used the default `--repeats 3`; its runner predates
the explicit SIGTERM handler, and no interruption occurred. Current batches
delegate failure cleanup to the separately tested owned-process-group runner.

Final offline validation passed **372 core tests**, **46 API tests**, both
five-test Rust selections and the native ABI contract. The new worker with
diagnostic hooks **disabled** also passed the existing **19 GPU/SFU cases**,
receiving **84/84** recorded source frames with the original SDK. This last
regression does **not** qualify lifecycle behavior of the opt-in SDK itself.

[Sanitized evidence](../benchmarks/reports/p06-publisher-stage-rate-control.json)
retains source/runtime hashes, build and software-provisioning receipts,
cohort revisions and missing indices. No V1, model/voice/avatar asset, global
setting or cloud deployment was changed. Before enabling this correction in
the research preview, qualify the patched SDK's full lifecycle, rate-pressure
behavior, actor configuration and browser cancellation/recovery, then pin an
explicit serving policy. Physical A/V, human quality, capacity and release
criteria remain open.

The [2026-10-29 lifecycle and fixed-ceiling follow-up](p06-publisher-lifecycle-rate.md)
now exercises the patched SDK with hooks disabled and records payload overshoot
at four ceilings. Controlled in-turn/network response and serving/browser
qualification remain pending.
