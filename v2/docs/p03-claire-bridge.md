# Claire C ABI and bounded GPU timeline

Local observation, 2026-10-27. This extends the [full-size renderer](p03-claire-render.md) toward live transport. It does not change the running browser backend or close P03/P06. [Structured evidence](../benchmarks/reports/p03-claire-bridge.json) binds source, binary, shared library, PTX, engine, SDK, scene and synthetic Qwen input hashes.

The pinned Claire SDK requires libstdc++ ABI 0; the existing LiveKit runtime uses ABI 1. `native/motion/claire_bridge.h` exposes only five C functions and fixed C structures. A version script hides all C++ symbols. The actual ABI-1 probe links and calls the ABI-0 SDK/encoder library; no LiveKit rebuild, global dependency change or additional model download occurred.

One owner thread/epoch holds the SDK primary context, borrowed-context encoder and 32 preallocated GPU pose slots. Complete skin/tongue poses use **25,774,848 bytes** of slot storage. Each SDK output stream records a dependency before the owned D2D copy; that copy completes before callback return/source reuse. No host geometry stage is introduced. The model remains 60 Hz and the render grid 30 Hz. Render requests use integer 48 kHz source positions in steps of 1,600; the SDK's integer 16 kHz timestamps differ from that nominal grid by at most two 48 kHz samples. The presentation origin cannot change within an epoch.

Future render requests return unavailable without consuming poses. A later audio cursor may discard stale video, counted explicitly. Overflow fails the epoch at 32 slots (**1,066.67 ms**); poses are never overwritten silently. Input remains contiguous, finite 16 kHz PCM with at most 320 samples per push and a 120-second generation bound. SDK EOF context padding is not playable audio. Cancellation retires the entire owner; pooled/reused model residency is not implemented by this bridge.

Actual commands from the repository root:

```sh
cmake --build v2/var/build/native-media --parallel 4 --target tha_claire_bridge_probe
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/.venv/bin/python \
  v2/scripts/qualify_claire_bridge.py \
  --engine v2/var/p03/claire-engine-fp32-n1-final \
  --scene v2/var/p03/claire-render-scene \
  --input-directory v2/var/p03/claire-qwen-input \
  --reference-directory v2/var/p03/claire-direct-render-final \
  --directory v2/var/p03/claire-bridge-final \
  --report v2/benchmarks/reports/p03-claire-bridge.json
cd v2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q \
  --junitxml=var/p03/claire-bridge-core-tests.xml
```

Warnings-as-errors native build passed; offline CPU **287 passed in 5.32 s**. Nine actual GPU cases passed:

| Case | Local evidence |
|---|---|
| Normal | 44,430 source samples → 167 neural / 84 encoded frames; all decoded frames equal the previous direct GPU reference |
| Paced | Real-time 20 ms input, independent 30 Hz deadlines; 84/84 decoded frames equal; queue peak 11; maximum observed schedule lateness 37.25 ms |
| Irregular | 1–320-sample pushes; 84/84 decoded frames equal |
| Short EOF | 177 samples, shorter than lookahead; exactly one neural/encoded frame, with no invented playable tail |
| Late video | Ten stale poses discarded; 74 retained frames, exact neural/render/discard accounting |
| Cancel then fresh | Retired an owner with actual pending GPU poses; fresh owner reproduces all 84 decoded reference frames |
| Overflow | Exactly 32 retained poses; next pose fails without overwrite/discard |
| Invalid input/owner | Foreign thread and nonfinite PCM rejected; failed epoch cannot resume |
| Ordering | Six separately retired owners reject PCM gap, excessive quantum, empty EOF, off-grid render, changed presentation origin and duplicate render |

Normal/paced input first produces a pose after 4,160 samples (**260 ms**). Peak retained SDK audio is 24,320 samples; the serving queue and SDK history are different bounds. The paced statistic is a local scheduler observation, not an acoustic/browser skew, latency percentile or SLO pass. Only decoded image parity is claimed; no new bridge Nsight transfer trace or memory peak is inferred from the prior renderer trace. The original first cohort and a probe timestamp-fixture failure remain in private directories rather than being deleted. Device admission was reacquired after all child exits.

Next: integrate the C bridge with the native session's separate audio clock and fresh publications, jointly bound its audio/motion lead and qualify actual Qwen streaming, EOF and receiver cancellation. Physical A/V, co-loaded memory, independent quality, calibrated Gaussian retarget and human enrollment/release gates remain open. V1 is preserved on `v1/legacy`.
