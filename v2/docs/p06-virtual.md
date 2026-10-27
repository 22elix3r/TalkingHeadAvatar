# P06 virtual-device reference

The CPU-copy reference feeds the existing project **Avatar Camera `/dev/video10`** and an ephemeral PulseAudio/PipeWire virtual microphone. Actual Firefox capture publishes both through a private local LiveKit room to a separate receiver. It checks EOF, cooperative cancellation, received speech and a black camera-clear acknowledgment. It is separate from the CUDA/NVENC path and does not yet consume live Qwen/session/mesh output.

This is a local browser qualification room, not a Zoom/Teams/Meet release pass. P06 remains **in_progress**. Native product integration, hard-kill camera privacy, arbitrary conference-client audio-buffer flush, physical A/V alignment, live-Qwen virtual output and longer stability remain open.

## Observed host and ownership

The existing v4l2loopback module is **0.15.3**; its kernel-reported driver version is 7.2.5. The project camera already used **640×480 BGR4**, 2,560 bytes/row, 1,228,800 bytes/frame. The adapter validates that exact card/format and refuses other devices/formats. It never calls format-setting ioctls, opens the OBS camera at `/dev/video0`, reloads a kernel module or changes driver settings. An advisory exclusive lock prevents concurrent project producers; the harness also refuses an already-used camera before starting. Other applications must leave this shared project camera idle.

PulseAudio client libraries are **17.0 / LGPL-2.1-or-later**, against the observed PipeWire **1.4.11** compatibility server (MIT package metadata). No third-party implementation is vendored. PulseAudio, driver/kernel and browser licenses remain separate from first-party code. The adapter creates uniquely named `tha_v2_probe_*` null-sink/remapped-source modules with low session priority, explicit 48 kHz mono channel maps and exact sink routing. Both normal EOF and cancellation retire only those owned modules. Host default source/sink and camera format are compared before/after. No system defaults, global profile or driver are configured.

One native owner keeps a BGRA frame buffer and PulseAudio connection. Integer 48 kHz samples independently schedule 10 ms audio blocks and 30 Hz video. The first full audio block starts the PulseAudio playback clock; opening an empty zero-prebuffer stream during browser preparation caused silence despite generated sample counts. A 960-byte / 10 ms prebuffer fixed that observed failure. This explanation is consistent with [PulseAudio's prebuffer/read-index documentation](https://www.freedesktop.org/software/pulseaudio/doxygen/structpa__buffer__attr.html); no server-internal read-index trace was collected.

At EOF/cancel the producer writes black, flushes/drains its own audio stream as appropriate, and keeps feeding neutral camera frames until a bounded consumer `finish` handshake. These extra clear frames are counted separately from valid response frames. A single final black write was missed in one receiver run; refreshing black while awaiting acknowledgment corrected the observed failure. The test verifies actual decoded receiver pixels before sending `finish`. It does not infer that another application's audio or video queues were flushed. A SIGKILL/device/server failure still needs a separate fail-stop privacy/recovery gate.

## Browser boundary

Chromium sees the virtual microphone but does not enumerate this camera's current combined CAPTURE/OUTPUT capabilities. [The upstream driver README](https://github.com/v4l2loopback/v4l2loopback#options) describes exclusive-capability mode for such clients. That would require a host driver/device configuration decision; this implementation preserves the existing configuration and records Chromium as unqualified for this virtual-camera profile.

Firefox **157** captures the current device configuration. Its test uses a newly created disposable profile with localhost camera/microphone permissions and legacy device enumeration so exact virtual IDs can be selected before capture. Firefox exposes labels differently from Chromium, as [Mozilla explains](https://bugzilla.mozilla.org/show_bug.cgi?id=1935371#c3). Only that new profile's permission database is seeded; it is removed afterward. Real device capture is enabled, synthetic browser devices are disabled, and normal loopback-ICE prohibition stays enabled. The test never opens a default/physical microphone to obtain permission or labels. This is not a test of a user's permission prompt or their installed Firefox 149 profile.

The browser requests exact project device IDs, 640×480 camera capture and mono audio, with microphone AEC/noise suppression/AGC disabled for this known input. WebRTC may downscale its encoded video independently; observed receiver dimensions are reported separately. The receiver audio graph has a zero-gain output solely to keep Firefox's analyzer processing; generated speech goes to the virtual sink, not a physical speaker. No human camera or microphone recording is part of this cohort.

## Reproduction

Prerequisites are the existing idle project camera/format, PulseAudio-compatible user session, cached Playwright Firefox, existing local P04 services and already installed development headers. No installer or driver modification is implied.

From the repository root:

```sh
cmake -S v2/native/virtual -B v2/var/build/virtual-reference -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++-14
cmake --build v2/var/build/virtual-reference --parallel 2
v2/.venv/bin/python v2/scripts/create_p04_microphone_fixture.py \
  --directory v2/var/p06/virtual-input
cd v2/web
npx playwright test --config playwright.virtual.firefox.config.ts \
  --reporter=json > ../var/p06/playwright-virtual-firefox-checkpoint.json
```

Fixture generation refuses overwriting existing inputs. It uses installed eSpeak only for the declared synthetic microphone phrase **“What is two plus two?”**, not as the serving speech provider. The test verifies its manifest/hash and takes exactly eight seconds of 48 kHz mono PCM16, including the declared initial silence. Qwen remains the selected product TTS; no model download, voice enrollment, checkpoint change or voice-polish task occurs here.

The native reference can also be driven directly after explicitly creating the project-owned sink: `v2/var/build/virtual-reference/tha_virtual_reference /dev/video10 tha_v2_probe_<unique> private-48k-s16le.pcm`. It waits up to 20 seconds for `start`, emits a sender-complete event, then refreshes black for up to five seconds while awaiting `finish`. Use the harness for owned module creation, actual receiver acknowledgment and cleanup. Merely launching the executable is not a conference delivery pass.

From the repository root, the explicit received-speech/export command uses the existing separate CPU ASR environment:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  v2/runtimes/session-api/.venv/bin/python v2/scripts/qualify_virtual_devices.py \
  --browser-report v2/var/p06/playwright-virtual-firefox-checkpoint.json \
  --output /tmp/p06-virtual-evidence.json
```

The exporter refuses failed/incomplete browser checks, changed source/binary fingerprints, an incorrect fixture, missing clear/ownership evidence or leftover temporary modules. It decodes the private receiver recording with existing FFmpeg and checks the known phrase using cached CPU-int8 Faster Whisper, `local_files_only=True`. This is a functional content check, not WER, independent human quality or live-Qwen qualification. Only synthetic text/hashes/aggregate observations enter the [public report](../benchmarks/reports/p06-virtual-devices.json); recordings, device IDs, raw frame/color traces and permission databases stay private.

The declared copies are CPU BGRA → V4L2 write, browser capture/conversion/VP8 encoding, and PCM → PulseAudio/PipeWire. They are outside the isolated native encoder trace's no-observed-raw-readback boundary. No zero-copy, physical A/V skew, latency/capacity SLO, hard-kill recovery or third-party conferencing claim follows from these short reference checks. Earlier enumeration, analyzer sampling, channel-map, empty-stream clock and single-clear-frame failures are retained in private reports rather than counted as final-protocol passes.

Actual final cohort: **2 passed in 20.14 s**, zero retries, Firefox 157. EOF sent 384,000 valid samples / 240 pattern frames, followed by 17 separately counted neutral frames; the receiver presented 215 nonblack frames and acknowledged black. Cancellation stopped at 179,040 valid samples / 112 pattern frames, followed by four neutral frames, with receiver black acknowledged. Captured input was 640×480; final coded receiver dimensions were 480×360 and 320×240 respectively. These frame observations do not establish a 30 Hz presentation SLO or exact frame conservation. The recorded receiver phrase decoded as **“What is 2 plus 2?”** with the cached CPU ASR. No temporary modules or camera producer remained; normal preview readiness was preserved. GCC 14 Release build, the explicit competing-owner rejection check and `npx tsc --noEmit` passed. The unchanged core previously passed 227 checks; no additional core dependency was introduced.
