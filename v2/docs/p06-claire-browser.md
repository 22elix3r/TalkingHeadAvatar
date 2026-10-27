# Claire vendor research browser profile

Local functional observation, 2026-10-28. The opt-in `native_claire_research` profile connects the pinned Claire motion/render/NVENC worker to the conversation actor and the local LiveKit browser reference. It uses the vendor's source-matched sample and a synthetic voice; this is not human enrollment or a retargeted identity. The normal audio/static profile remains separate. [Sanitized browser evidence](../benchmarks/reports/p06-claire-browser.json) and the current [native worker](../benchmarks/reports/p03-claire-session-worker.json), [bridge](../benchmarks/reports/p03-claire-bridge.json) and [live receiver](../benchmarks/reports/p03-claire-live-preroll-20261028-integration.json) reports record the observed boundaries.

The research API exposes only the builtin `claire-vendor-research` asset and denies asset imports or enrollment mutations. It pins the Qwen and Claire revisions together for session admission, checks the model/scene/topology/runtime inventory and GPU identity, and uses separate profile database, credentials and cookies. Imported identities are not silently applied to the vendor mesh. This reference's vendor assertion is not evidence of consent for a human subject.

Claire's 260 ms future context and variable live synthesis exposed a motion underrun in the initial typed browser attempt. The worker now waits for 500 ms of eligible motion, or the available duration at EOF, before speech starts and bounds motion lead to 800 ms; backpressure remains bounded and does not discard speech. `THA_CLAIRE_MEDIA_2` binds the exact 48 kHz audio EOF separately from the rounded 16 kHz motion input and counts any extra rounded EOF pose. The existing 32-slot GPU pose bound remains enforced. These settings are functional safeguards, not a latency qualification.

The existing operator-owned profile on this host can be managed without fetching models:

```sh
v2/.venv/bin/tha local doctor --root v2 --name p06-claire-v2
v2/.venv/bin/tha local up --root v2 --name p06-claire-v2
# Open http://127.0.0.1:18730/ on this computer.
v2/.venv/bin/tha local down --root v2 --name p06-claire-v2
```

This profile depends on the already provisioned and qualified private Claire descriptor, cached Qwen/ASR/LLM, native binaries, isolated TensorRT and local service images. It is not a general fresh-install recipe. Local profile setup can explicitly reuse the checkout's locked API environment using `tha local provision --borrow-runtime v2/runtimes/session-api/.venv`; the provisioner audits installed file hashes and an offline lock dry run. Profile metadata and credentials remain private under `v2/var/`.

Browser qualification used the real loopback API, database, SFU and cached providers:

```sh
cd v2/web
VITE_REFERENCE_PROFILE=claire_vendor_research npm run build -- --outDir ../var/p06-claire/web-research
THA_CLAIRE_COHORT=chromium-avatar-large npx playwright test -c playwright.claire.config.ts
THA_CLAIRE_COHORT=firefox-avatar-large npx playwright test -c playwright.claire.firefox.config.ts
```

Chromium **2 passed**: typed interaction with interruption/fresh audio and video tracks, and a synthetic microphone through CPU ASR, LLM, Qwen and the browser receiver. Firefox **1 passed**: typed interaction with interruption and fresh tracks, with loopback ICE disabled. Each observes decoded audio and a video presentation callback for the final epoch. Sender EOF conserves audio and bounds queues; it explicitly does not acknowledge physical playback. The research video is displayed at 280 pixels in the conversation card. The ordinary audio-reference Firefox test also passed. The Firefox research run used the immediately preceding UI build; the subsequent change restored the static profile's thumbnail while retaining the same research layout.

The native worker's 19 GPU/SFU cases cover recorded input, exact/partial EOF boundaries, protocol faults and blocked-producer cancellation. In one live-Qwen receiver cohort, the sender produced **324 source frames** and the receiver observed **312**, including the first and final source frames; twelve missing intermediate indices remain a finding. A separate earlier zero-gap cohort is preserved as [historical evidence](../benchmarks/reports/p03-claire-live-preroll-20261027.json), not a repeatability claim. The research descriptor requires explicit opt-in when receiver gaps remain. Neither cohort proves complete frame delivery, physical A/V alignment, low latency, visual quality or concurrency capacity.

The CPU core suite passed **335** tests and the isolated API/PostgreSQL suite **29**; the web unit suite passed **2**, and TypeScript/Vite builds passed. The model and runtime were already present; this checkpoint downloaded no weights and did not alter V1. Human consent/enrollment, avatar retargeting, independent visual quality, receiver suppression and physical A/V, deployment and release remain open.
