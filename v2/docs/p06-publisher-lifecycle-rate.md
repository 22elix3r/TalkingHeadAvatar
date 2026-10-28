# P06 publisher lifecycle and fixed-ceiling observations

Local observation, 2026-10-29. The candidate SDK passed the complete recorded
EOF/fault/cancel cohort with publish tracing disabled, before and after adding
an isolated rate probe. Four live Qwen turns conserved every source frame at
different fixed ceilings. This advances the [publisher investigation](p06-publisher-stage.md);
the candidate remains outside serving. Constrained-network response, browser
rollout, physical presentation and quality acceptance are open.

## Lifecycle checks

The qualifier accepts mutually exclusive `--publisher-runtime` (no stage hooks)
and `--publisher-trace-runtime` (live stage/source tracing). Both consume the
same explicitly pinned SDK/worker inventory. Trusted external rate control
requires a descriptor binding the qualified opt-in SDK overlay. Inherited
diagnostic flags are cleared before launching the worker.

Both actual 19-case cohorts passed: recorded Qwen, segmented input with a gap,
one-sample/partial/quantum/grid EOF boundaries, empty input, wrong epoch, sample
gap, wrong sequence, wrong motion and cancellation under backpressure. The
recorded case received **84/84** source frames. Malformed cases exited **1** as
required; successful EOF cases exited **0**. Cancellation stopped its owned
worker and released admission. The parent completed with exit 0/no panic;
publication IDs were fresh. These are SDK/Python-receiver observations, not
browser-buffer flushing or audible cancellation measurements.

The first cohort used descriptor
`e0726d09c2dc243a75fa63645bd18c9db1d465f93bda4799c47c9e64480b3968`.
The final cohort used the rebuilt worker and descriptor
`ef8ad888c6e8638bfc4d1c73e06ecce4da7234cb087290e521872499e9b0366e`,
with probe/stage hooks disabled. Both select FFI
`084e4f30b6748b3c2434eafedbef142856b56bbb8cc0d97c5353992dcb1e1188`.
Distinct worker/source/receipt hashes remain in the evidence.

## Fixed-ceiling study

The rate probe requires both exact operator flags, `THA_CLAIRE_RATE_PROBE=1`
and one of four allowed `THA_CLAIRE_PROBE_BITRATE_BPS` values. It sets the
publish ceiling and initializes the external encoder before preroll. Subsequent
feedback retains the existing encoder bounds, with an additional clamp to the
experimental ceiling. Serving strips both flags. The default worker still
publishes at 1.2 Mbps with its existing feedback policy.

At most **256** feedback records are retained; overflow rejects the observation.
Rows separate the SDK's requested rate from the rate accepted by the external
encoder API. Their clock is the publisher's 48 kHz stream sample counter,
including preroll. They do not acknowledge hardware execution or delivery.
Per-frame payload sizes are measured independently at source capture. The
complete final partial second remains explicit.

Actual order was 300 kbps, 1.2 Mbps, 600 kbps, 2.4 Mbps. Every process exited 0
without panic, receiving **324/324** source frames with none missing at upload,
encoder output, packetizer or receiver boundaries.

| Fixed ceiling | Source payload average | Full one-second payload range | Largest access unit |
|---|---:|---:|---:|
| 300 kbps | 332.09 kbps | 246.55–411.58 kbps | 10,596 bytes |
| 600 kbps | 604.91 kbps | 482.54–783.34 kbps | 18,699 bytes |
| 1.2 Mbps | 1.1963 Mbps | 0.9960–1.5228 Mbps | 33,126 bytes |
| 2.4 Mbps | 2.4881 Mbps | 2.1738–2.9720 Mbps | 51,445 bytes |

Overshoot and peaks are retained. Payload rates exclude RTP overhead. Each
turn has a separate fixed configuration on one local SFU; there is no in-turn
bandwidth transition or network shaper. The results show a rate response and
frame conservation for this fixture, not strict rate-cap, burst, pacer-delay,
constrained-network or quality compliance.

## Reproduction and validation

Existing SDK, Claire/model/scene artifacts were reused. From the repository root:

```sh
cmake --build v2/var/build/native-media \
  --target tha_claire_session_media tha_probe_rate_contract --parallel 2
v2/var/build/native-media/tha_probe_rate_contract
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/.venv/bin/python \
  v2/scripts/describe_publish_trace.py \
  --library v2/var/build/livekit-ffi-publish-rate-control-20261028/release/liblivekit_ffi.so \
  --build v2/var/p06-claire/publish-rate-control-ffi-build-1.json \
  --worker v2/var/build/native-media/tha_claire_session_media \
  --output v2/var/p06-claire/publisher-rate-pressure-20261029/runtime.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 v2/runtimes/session-api/.venv/bin/python \
  v2/scripts/qualify_claire_rate_matrix.py \
  --publisher-runtime v2/var/p06-claire/publisher-rate-pressure-20261029/runtime.json \
  --receiver-runtime v2/var/p06-claire/inbound-runtime-reason-v1.json \
  --config v2/var/p04/test-runtime.json \
  --engine v2/var/p03/claire-engine-fp32-n1-final \
  --scene v2/var/p03/claire-render-scene \
  --directory v2/var/p06-claire/publisher-rate-pressure-matrix-20261029 \
  --report v2/var/p06-claire/publisher-rate-pressure-matrix-20261029.json
```

Preserve old outputs; use new paths for reruns. The matrix owns child process
groups through the separately tested runner, including SIGTERM cleanup, and
checks source/runtime identities at both ends. Lifecycle checks used that same
runner with a 360-second timeout around the qualifier, passing
`--publisher-runtime <descriptor> --trust-encoded-rate-control`, the same
receiver/config/engine/scene options and new directory/report paths. Omitting
`--live-qwen` and every trace/probe option selects all 19 cases. The final
private lifecycle owner receipt records exit 0/no panic.

Final offline command, from `v2/`:
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q --junitxml=var/p06-claire/publisher-lifecycle-rate-core-tests.xml`
— **386 passed in 7.94 s**. From `v2/runtimes/session-api/`, the equivalent
command with `--junitxml=../../var/p06-claire/publisher-lifecycle-rate-api-tests.xml`
passed **46 tests in 5.41 s**. New checks reject missing, malformed, truncated,
overflowing, inconsistent and non-finite feedback and cover serving environment
isolation. The warnings-as-errors native build and CPU rate guard passed.
Review found no critical/important issue and independently passed 35 CPU/parser/
process checks plus the serving-isolation check.

[Sanitized evidence](../benchmarks/reports/p06-publisher-lifecycle-rate.json)
binds both lifecycle cohorts, all four rate turns, source/runtime/model/scene
identities and JUnit receipts. Logs, media, credentials and publication IDs stay
private. All three previews remained ready. No software/model download,
training, V1 mutation, global change or cloud deployment occurred.

Next: controlled in-turn rate reduction and recovery, then an explicit actor/
serving policy with browser cancel/recovery and idle rollback evidence.
P02–P06 stay in progress; no presentation, quality, capacity or release gate is
closed by this study.
