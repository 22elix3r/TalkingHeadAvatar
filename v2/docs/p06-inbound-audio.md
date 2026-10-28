# Bounded native microphone input

The opt-in inbound runtime closes the native audio stream on queue overflow
instead of evicting older speech. Its private descriptor binds the source
revision, exact patch, Cargo lock, library and actual process-exit qualification.
The application rejects a different loaded library or Python SDK version.
`tha local doctor` verifies the descriptor without loading the SDK or models.

The selected path uses **25 native decoded frames**, nominally **250 ms**, and
**12 Python output frames at 20 ms**, or **240 ms**. These bounds cover those
two receiver queues. Network, codec and browser buffering remain separate;
there is no physical presentation or end-to-end latency claim.

Native overflow records its reason before closing and clearing the queue. The
patched FFI sends an EOF with `native_queue_overflow=true` in optional field 1.
The pinned Python SDK retains that extension as unknown protobuf bytes, which
the qualification verifies. The session actor treats unexpected microphone EOF
and the expected user's microphone-unpublished event as
`microphone_discontinuity`: it fences the response, closes live sources and
persists session failure. Normal shutdown and foreign/non-microphone track
events do not trigger that handler.

## Offline build

Use a separate worktree of the already cached Rust revision
`bae4df2f3d6c8f0f6ce81a7370c445779f1948d2`, with cached protocol revision
`9a7c5cf96bbbcbc30a8ce76bca16815ba5b7bb33` and libyuv revision
`917276084a49be726c90292ff0a6b0a3d571a6af`. Apply
[`livekit-native-audio-fail-closed.patch`](../patches/livekit-native-audio-fail-closed.patch)
there, preserving the original source tree. The observed build used copied
project-local Cargo caches and target artifacts, an existing Rust 1.99.0
sysroot, cached WebRTC and the previously built protoc 25.3.0. It did not invoke
Rustup installation or fetch model artifacts.

The actual build command, from the repository root, was:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/.venv/bin/python \
  v2/scripts/build_inbound_audio_ffi.py \
  --source v2/var/sources/livekit-ffi-audio-fenced-20261028 \
  --target v2/var/build/livekit-ffi-audio-fenced-20261028 \
  --cargo-home v2/var/p06-claire/cargo-audio-fenced \
  --webrtc v2/var/sources/livekit-cpp-v1.11.0/client-sdk-rust/target/release/build/scratch-7323677a5bf9c8a5/out/livekit_webrtc/livekit/linux-x64-release-webrtc-89d790b/linux-x64-release \
  --protoc v2/var/build/livekit-cpp/bin/protoc-25.3.0 \
  --rust-sysroot /home/elix3r/.rustup/toolchains/stable-x86_64-unknown-linux-gnu \
  --report v2/var/p06-claire/audio-ffi-build-reason-v1.json
```

The helper checks the complete HEAD diff/lock, rejects extra staged or
untracked source changes and dirty pinned submodules before and after building,
selects four actual native queue
tests and builds with `--locked --offline`. Existing output is retained. The
observed library SHA-256 is
`16eb6e043651eb1c509bf515f512abac2c0e079035cc24928f72d3daa9866d03`.

## Actual queue and lifecycle qualification

`qualify_native_audio_queue.py` sends synthetic tone PCM through real local
LiveKit rooms. Its overload case deliberately stalls one receiver FFI callback
for 500 ms while the source continues; this is a disposable test process, not a
serving hook. A normal case checks received nonzero audio, and the stock-library
control demonstrates that merely setting a finite queue can evict and resume.

`qualify_inbound_audio_matrix.py` runs five normal and five overflow processes
plus one stock control, checking their entire exits and panic logs. Its final
matrix passed all **11** processes with exit code 0. Every patched overflow case
observed the explicit native-overflow EOF; no resumed audio was consumed.
These tests qualify queue and lifetime behavior, not ASR quality or lossless
network audio.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/runtimes/session-api/.venv/bin/python \
  v2/scripts/qualify_inbound_audio_matrix.py \
  --ffi v2/var/build/livekit-ffi-audio-fenced-20261028/release/liblivekit_ffi.so \
  --stock-ffi v2/runtimes/session-api/.venv/lib64/python3.12/site-packages/livekit/rtc/resources/liblivekit_ffi.so \
  --config v2/var/p04/test-runtime.json \
  --directory v2/var/p06-claire/audio-queue-reason-matrix-1 \
  --report v2/var/p06-claire/audio-queue-reason-matrix-1.json
```

Use new output paths for a repeat. A private runtime descriptor is created only
after that evidence exists:

```sh
v2/runtimes/session-api/.venv/bin/python v2/scripts/describe_inbound_audio.py \
  --library v2/var/build/livekit-ffi-audio-fenced-20261028/release/liblivekit_ffi.so \
  --build v2/var/p06-claire/audio-ffi-build-reason-v1.json \
  --matrix v2/var/p06-claire/audio-queue-reason-matrix-1.json \
  --output v2/var/p06-claire/inbound-runtime-reason-v1.json
```

An earlier prototype aborted during interpreter teardown after printing its
stream result. The qualification now disposes its process-owned SDK while
Python callbacks and the event loop remain alive, and the parent verifies the
actual exit. An earlier repeated case also lacked the periodic log warning;
the explicit EOF reason replaces that log-only cause assertion. Failed logs,
receipts and the initial library remain private and are not passing evidence.

## Local profile

Add an explicit `inbound_audio_runtime` descriptor path to an operator-owned
source config before `tha local init`. Startup selects that library in the
owned API process; it does not replace an installed Python package or mutate
the parent environment. The descriptor snapshot is private and immutable by
convention. The qualified helper/entrypoints dispose their owned SDK explicitly.

The isolated profile `p06-input-bounded` runs on `http://127.0.0.1:18760`:

```sh
v2/.venv/bin/tha local doctor --root v2 --name p06-input-bounded
v2/.venv/bin/tha local up --root v2 --name p06-input-bounded
cd v2/web
THA_LOCAL_PROFILE=p06-input-bounded THA_INBOUND_COHORT=chromium-native25-final \
  ./node_modules/.bin/playwright test -c playwright.inbound.config.ts
```

The final Chromium cohort passed **3** checks in **30.11 s** (one Firefox-only
case skipped): real typed
LLM/Qwen/receiver speech with interrupt/resume; synthetic microphone through
actual ASR/LLM/Qwen playback; microphone unpublication fails the real session
and a fresh admitted session reaches readiness. The fault probe uses an actual
served local document: a route-fulfilled document had triggered Chromium Local
Network Access denial. Browser security remains enabled.

The final Firefox cohort passed **2** checks in **19.44 s**: typed
interrupt/resume and synthetic microphone VAD barge-in followed by microphone
mute and a fresh decoded answer. Firefox's continuous fake microphone first
interrupted the answer through the real VAD path; it is not a human ASR-quality
or long-idle recovery qualification. Both browsers used cached real providers.

The saved final commands, from `v2/web`, were:

```sh
THA_LOCAL_PROFILE=p06-input-bounded THA_INBOUND_COHORT=chromium-native25-qualified \
  ./node_modules/.bin/playwright test -c playwright.inbound.config.ts --reporter=json
THA_LOCAL_PROFILE=p06-input-bounded THA_INBOUND_COHORT=firefox-native25-qualified \
  ./node_modules/.bin/playwright test -c playwright.inbound.firefox.config.ts --reporter=json
```

Final offline regression: **351 core** and **39 API** tests passed; the web
TypeScript/production build and **2** web unit tests passed. The existing
585.66 kB LiveKit chunk warning remains. See the sanitized
[checkpoint](../benchmarks/reports/p06-inbound-audio.json),
[native build](../benchmarks/reports/p06-inbound-audio-build.json) and
[11-process matrix](../benchmarks/reports/p06-inbound-audio-matrix.json).
Private browser receipts retain observations; public summaries omit track and
session identifiers, tokens and raw media.

One retry mistakenly invoked `npx` from the repository root, fetching
Playwright 1.64.1 before failing for a missing config. The final cohorts used
the already-installed project executable shown above. No model weights or
browser binaries were downloaded for this checkpoint.

The initial qualification left ordinary previews unchanged. Their subsequent
[idle upgrade/rollback checkpoint](p06-inbound-upgrade.md) migrated ports
18700 and 18730 to this runtime and exercised browser recovery. Physical A/V, independent voice/avatar
quality, complete receiver frame delivery and the remaining P03–P08 gates stay
open.
