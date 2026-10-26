---
name: avatar-e2e-delivery
description: Implement or review complete TalkingHeadAvatar product flows across browser, API, enrollment, inference and playback using the V2 work packages and acceptance journeys.
---

# Avatar end-to-end delivery

Read the relevant package and dependencies in [IMPLEMENTATION_BACKLOG.md](../../../docs/v2/IMPLEMENTATION_BACKLOG.md), then the matching contract in [E2E_IMPLEMENTATION_SPEC.md](../../../docs/v2/E2E_IMPLEMENTATION_SPEC.md). The [main plan](../../../V2_RESEARCH_AND_BUILD_PLAN.md) remains the architecture authority.

For implementation sessions, begin with [phase guide](../../../docs/v2/IMPLEMENTATION_PHASES.md) and [current status](../../../docs/v2/IMPLEMENTATION_STATUS.md). Execute the named/current phase, validate it and persist commands, evidence, missing acceptance and next action. P00 needs no model assets. Qwen3-TTS is user-selected; do not reopen a TTS tournament or add alternative families before its vertical slice. Distinguish functional progress from pending runtime/performance/human-quality qualification; unavailable raters need not block independent functional work.

Start from an observable user journey, identify every required boundary and implement a runnable path through it. Preserve the small first slice, but do not label it complete product delivery. Real-model integration, enrollment/consent, browser playback and recovery cannot be replaced by mocks in final acceptance.

Keep session policy owned by one actor, inference weights pooled, and mutable state isolated. Product lifecycle belongs in authenticated APIs and durable metadata; PCM/render frames remain on bounded real-time paths. Use idempotency, fenced leases and immutable asset revisions. Revocation must win races against job completion and new session attachment.

For browser work exercise permission/autoplay failure, device selection, captions, stop/mute, reconnect and receiver-safe cancel-resume. Do not expose debug implementation details or provider secrets in normal flows. For enrollment, exercise a real authorized capture through evaluation, review, publication and use; a progress screen alone does not satisfy the journey.

Close a work package with code, configuration, meaningful tests, a runnable documented command/flow and stored evidence. Mark unavailable assets/providers or failed gates explicitly. Do not claim E2E success from only unit tests, mocked speech or offline MP4 output. Preserve V1 and unrelated user state; update ADRs when measured evidence changes the implementation choice.
