---
name: avatar-realtime-benchmark
description: Design, run or review TalkingHeadAvatar latency, quality and concurrency benchmarks with explicit presentation boundaries and admission gates.
---

# Avatar real-time benchmarking

Use [VALIDATION_PROTOCOL.md](../../../docs/v2/VALIDATION_PROTOCOL.md) for fixtures, event records, cohorts and statistical treatment. Read the applicable budget/gate in the [main plan](../../../V2_RESEARCH_AND_BUILD_PLAN.md).

For model choice, run [Q0](../../../docs/v2/MODEL_QUALITY_BENCHMARKS.md#6-staged-quality-qualification-q0): matched inputs, blinded independent raters, separate quality axes, failures and clustered uncertainty. Include NC candidates equally. Keep author benchmarks and practitioner anecdotes separate from independently measured preference. Compare the exact streaming/runtime configuration that will serve users, then co-load the finalist; a batch checkpoint's quality does not establish streaming parity. Inconclusive preference stays unresolved.

Qwen3-TTS is already user-selected. Its initial gate is official-reference versus actual streaming-runtime parity plus latency/cancel/memory checks, not a mandatory multi-model tournament. Pending human ratings must not be reported as passed, but need not block independent functional phases in the [implementation guide](../../../docs/v2/IMPLEMENTATION_PHASES.md).

Define the tested claim before choosing the benchmark. End-to-end response starts at the annotated client speech endpoint and ends at meaningful audio plus aligned video presentation. Include endpoint patience, first motion-eligible PCM, lookahead, queues and receiver buffering. Report complete-answer duration separately.

For a microbenchmark, disclose isolation, synthetic inputs, synchronization, warmup, precision and sample count. The existing [audit probe](../../../docs/v2/benchmarks/audit_probe.py) is offline and diagnostic; it cannot prove conversational quality or production p99.

For system claims, preserve raw events/configuration and clock uncertainty. Count failed/timeout turns as SLO violations. Use the 1/2/4/8/16 sweep, open-loop offered load, unique/shared assets, burst cases and soak specified by the protocol. Report false cutoffs, quality, fairness and resource limits alongside latency.

Do not add stage p95 values to infer joint p95 or divide peak FPS by target FPS to claim concurrency. Compare matching profiles and run quality noninferiority checks after precision/compilation changes. State observed capacity separately from projections and cost assumptions.

Release validation also needs the real user journeys in [IMPLEMENTATION_BACKLOG.md](../../../docs/v2/IMPLEMENTATION_BACKLOG.md), including external-network/forced-TURN operation, receiver-safe interruption-resume, enrollment/revocation, worker failure and rollback/restore. Benchmark mocks only establish their declared contracts, not these journeys.

Finish with the measured gate decision, reproducibility paths and blockers. Avoid installing dependencies or changing global drivers merely to make a baseline appear runnable.
