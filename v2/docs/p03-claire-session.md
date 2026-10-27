# Claire native session sender and receiver

Initial pre-preroll observation at `82c6e83`, 2026-10-27. The [later preroll checkpoint](p03-claire-preroll.md) reaches 84/84 recorded and 324/324 live source frames in its final cohorts; the initial measurements below are preserved as history. [Structured evidence](../benchmarks/reports/p03-claire-session-worker-20261027.json) qualifies a separate research worker, not the running API/browser backend. The normal preview remains the audio/static reference. [Bridge evidence](p03-claire-bridge.md) covers SDK/GPU geometry ownership and image parity; physical A/V and quality remain open.

`tha_claire_session_media` links the ABI-1 LiveKit C++ runtime through the ABI-0 Claire C bridge. One GPU thread owns Claire, pose storage, raster and NVENC; the CPU audio clock captures 480 samples every 10 ms independently. Paired input contains 960 playback samples at 48 kHz plus 320 motion samples at 16 kHz per normal packet. A 33-byte header carries kind, epoch, sequence, audio start and both counts. Atomic EOF carries the exact shorter pair, including an empty final packet after an aligned input. Empty complete turns are rejected.

`ClaireInputMux` preserves separate FIR histories from the same native 24 kHz Qwen stream, accepts at most 2,400 samples/100 ms per event, emits paired 20 ms packets, and retains only a partial quantum. EOF conserves `2*N` playback and `ceil(2*N/3)` motion samples. Stale generations cannot advance either filter; cancellation discards both histories without flushing. The async caller runs the reference FIR off the event loop and awaits sink backpressure before the next event.

The host audio queue is bounded to 48,000 samples/one second; the host motion queue to 50 packets/one second; GPU input lead to 500 ms relative to valid audio consumed, including Claire's 260 ms lookahead. The bridge separately bounds owned poses to 32. Startup requires 300–500 ms configured audio or actual EOF plus an available pose. Active audio or motion starvation fails the epoch; speech is never deleted to reduce queues. SIGTERM bypasses a blocked producer, and each worker creates fresh publication IDs. Receiver buffer flushing is not inferred from child exit.

Transport feedback retains keyframe demand until a packet is available and applies bounded bitrate/cadence hints without changing the audio/motion grid. At motion EOF the worker holds its last owned pose and keeps publishing explicitly tagged idle frames at later presentation times. These do not add source audio or neural frames. The initial cohort stopped video at EOF and missed the final decoded source frame; the continued idle stream fixes that observed tail. It does not resolve startup presentation.

Actual commands from the repository root:

```sh
cmake --build v2/var/build/native-media --parallel 4 \
  --target tha_claire_session_media tha_claire_bridge_probe
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/runtimes/session-api/.venv/bin/python \
  v2/scripts/qualify_claire_session_worker.py \
  --config v2/var/p04/test-runtime.json \
  --engine v2/var/p03/claire-engine-fp32-n1-final \
  --scene v2/var/p03/claire-render-scene \
  --directory v2/var/p03/claire-session-paired-qualified \
  --report v2/benchmarks/reports/p03-claire-session-worker.json
cd v2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q \
  --junitxml=var/p03/claire-session-core-tests.xml
```

Warnings-as-errors native build passed. **308 CPU tests passed in 5.65 s**, including 21 paired-stream boundary cases. **13 actual GPU/SFU cases passed**: recorded synthetic-reference Qwen, 1/479/480/481/960/961-sample tails, empty-turn rejection, wrong epoch, source gap, sequence error, mismatched motion quantum and priority cancellation under backpressure. Admission was reacquired after all exits; publication pairs were distinct. Bootstrap tokens are purged, and raw biometric media is not put in Git.

The complete recorded-Qwen case was fed incrementally through the actual mux in 997-sample source chunks. It conserves **133,290 valid 48 kHz samples + 1,110 codec padding samples**. The sender published **84 source frames, zero source-video skips**, with host audio peak **48,000 samples**, host motion peak **25 packets**, and GPU pose peak **16 frames**. It observed **one keyframe request / twelve rate updates**. The receiver decoded **81 source frames**, including the final source frame, plus separately marked post-EOF idle frames, and nonsilent audio. Source frame indices **0, 1, 2** were missing at receiver startup. No whole-utterance receiver-delivery or acoustic lip-sync pass is claimed. Source metadata does not measure a speaker's audio presentation time.

The blocked producer's worker retired in **361.77 ms**; this is a measured compute/owner retirement cost, not the ≤200 ms stale-audio suppression gate or a receiver-safe resume proof. The earlier synchronous-FIR cohort overflowed the checked receiver queue; moving FIR work off the event loop fixed that local failure. The initial EOF-tail and helper-fixture failures remain private. The SDK history, owned GPU pose storage, host transport queues and native codec padding are separately accounted boundaries.

No new model/runtime download, training, global dependency/driver change or cloud deployment occurred. The input is an existing Qwen recording with a synthetic voice reference, not live TTS in this cohort. Native LiveKit reception is not a browser/physical playback result. Memory peaks, joint Qwen residency, human identity/consent enrollment, Gaussian retarget, independent quality and release remain unqualified. Next: real cached Qwen incremental output into the paired worker, joint resource measurement, then opt-in actor/browser integration with explicit vendor research asset identity and receiver startup/cancel gates.
