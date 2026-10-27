---
name: avatar-gpu-media
description: Profile and improve avatar CUDA scheduling, residency, tensor/surface ownership and render-to-encoder transfers without sacrificing tail latency.
---

# Avatar GPU and media analysis

Read the GPU findings in [V1_AUDIT.md](../../../docs/v2/V1_AUDIT.md), main-plan sections 11–14, and [benchmark protocol](../../../docs/v2/VALIDATION_PROTOCOL.md).

Map every owner and lifetime: shared model, per-avatar geometry, per-session state, tensor, render surface, converter and encoder. Identify host/device copies and synchronization boundaries with traces rather than API names. A Python frame publisher is not inherently zero-copy.

Measure completed operation time, device service, queue wait and total device memory under the actual co-loaded profile. Include non-PyTorch allocations, desktop reserve, encoder surfaces and loading peaks. Do not infer fit from parameter counts or rely on stream priority as hard preemption.

Optimize the largest observed bottleneck first. Retain canonical assets and model state, preallocate surfaces, quantize/color-convert on GPU before unavoidable CPU output, and use explicit stream events. Keep surfaces alive until the final encoder/transport consumer releases them.

Compilation/graphs require validated shapes, addresses and engine compatibility; check ONNX provider placement and quality parity. Batching must have bounded wait. Trace LLM prefill bursts before admitting shared-GPU sessions. Validate the actual GPU SKU/driver encoder capabilities; do not assume datacenter GPUs have NVENC.

The native publishing candidate and fallback are specified in [E2E_IMPLEMENTATION_SPEC.md](../../../docs/v2/E2E_IMPLEMENTATION_SPEC.md). For pre-encoded LiveKit input, serialize capture, supply complete access units, honor keyframe/rate feedback and qualify one coherent A/V clock. Count small encoded-payload copies separately from raw-frame readback; do not claim zero-copy from an encoded-source API name.

A claimed copy removal needs profiler evidence and lifetime/cancellation tests. If interop is unstable or slower, retain the explicit-copy reference and document the result. Report performance and quality at admitted load, not only isolated FPS.
