# Implementation backlog and completion matrix

Revision 2026-10-26. This is the complete delivery backlog for the [main plan](../../V2_RESEARCH_AND_BUILD_PLAN.md), with contracts in [E2E_IMPLEMENTATION_SPEC.md](E2E_IMPLEMENTATION_SPEC.md) and release criteria in [DEPLOYMENT_AND_RELEASE.md](DEPLOYMENT_AND_RELEASE.md). All items are **planned**, except the existing audit/research documents. Proposed paths below do not yet exist.

**Execute [P00–P09](IMPLEMENTATION_PHASES.md) in order and maintain [implementation status](IMPLEMENTATION_STATUS.md).** The E packages below define coverage, not a requirement to scaffold the entire stack before the first runnable phase. Qwen3-TTS is the selected initial provider; alternative TTS integrations are deferred. P00 starts without weights or assets.

## Execution rules

Implement the smallest vertical slice first, but do not mistake it for the final deliverable. Each task closes with code, configuration, meaningful validation, a runnable operator/developer path and documented limitations. An unimplemented fallback, mocked provider or placeholder UI is not complete. Record model/asset changes in the evidence ledger and release manifest. Future work may parallelize independent tasks only when authorized; dependencies below do not themselves request agent delegation.

Research-first quality evaluation follows [independent evidence/Q0](MODEL_QUALITY_BENCHMARKS.md). NC models may be primary components. Commercial clearance is later work. The user has selected Qwen: Q0 now qualifies runtime parity and output quality rather than requiring a TTS tournament. Visual Q0 precedes expensive training and final enrollment recipe lock; pending independent ratings do not block functional P04 integration. Reuse research implementations before funding clean replacements.

## Work packages

| ID / package | Concrete paths or outputs | Dependencies | Acceptance evidence |
|---|---|---|---|
| E00: baseline/rights | `v2/docs/baseline.md`, model/asset manifest, legacy import inventory | Existing audit | Reproduce actual load failures, identify lawful fixtures, no V1 mutation; no assumed consent |
| E01: project foundation | `v2/pyproject.toml`, lock, `web/package.json` + lock, native CMake, CI skeleton, config schema | E00 | Fresh isolated CPU install, offline imports/tests, browser build; actual compatible versions pinned |
| E02: contracts | `src/talkinghead_v2/contracts/`, protobuf/schema fixtures | E01 | Audio spans, EOF, epoch, rig/basis, provider capabilities and version negotiation reject invalid inputs |
| E03: sample clock and cancellation | `audio/`, `session/epochs.py`, deterministic replay | E02 | Irregular chunks/rates/tails, drift, stale callbacks and all-stage cancellation pass; no word deletion |
| E04: local media sink | reference renderer/synthetic rig, PCM output and copy-based browser publisher | E03 | Actual playback, skew and interruption measured over 30 min; CPU-only visual claim clearly bounded |
| E05: selected Qwen adapter | Base 1.7B official reference and one actual PCM streaming backend, voice registry/cache | E02–E03; audio sink can be built here; artifacts/reference | P02 streaming/horizon, pronunciation, EOF/cancel, reference parity and memory evidence; alternative TTS families deferred |
| E06: articulation/rig spike | A2F retarget first, mouth tests and consented clips; StreamingTalker if baseline fails | E03–E05; research-eligible asset | Context, peak memory and bilabial/teeth quality measured; no author-only winner; one challenger per failed gate |
| E07: Gaussian rendering baseline | canonical asset loader, reusable research renderer, native gsplat migration target, mesh fallback | E06 | Orientation/covariance and oral geometry correct; dependency provenance; matched-view quality/timing before clean rewrite |
| Q0: staged quality qualification | Qwen runtime/reference parity, visual clips, blinded ratings, result manifests and ADRs | Actual E05–E07 outputs; raters for subjective gate | Preserve selected Qwen; qualify quality/co-load; subjective pending stays pending while functional work proceeds; resolve before E13 recipe lock/release |
| E08: native media transport | native CUDA/NVENC + LiveKit encoded publisher, Python bindings | E04; can precede final E07 | Synthetic surface then real render; one A/V clock, keyframe/rate feedback, no raw readback claim without trace, safe release/churn |
| E09: real conversation | streaming ASR, VAD/turn policy, llama.cpp/provider adapter, stable text segments and session actor | E05–E07; E08 optional until optimized profile | Real user → useful aligned answer; pauses/corrections/barge-in; p95/quality jointly measured |
| E10: application persistence/API | API, database migrations, tenant policies, idempotency, outbox and job schemas | E02 | API contract tests with PostgreSQL, tenant denial, duplicate requests and crash/retry behavior |
| E11: product browser | `web/` device preflight, avatar/voice select, captions, status, stop/mute, accessible errors | E04, E10; E09 for final | Actual user flow and receiver-safe cancel/resume; no secret in bundle; autoplay/device/TURN/reconnect tested |
| E12: asset lifecycle | uploads, validation, immutable revision store, consent registry and scoped grants | E10 | Size/hash/decoder checks, unauthorized access blocked, revocation wins races, client-distribution flag enforced |
| E13: enrollment | stage worker, research-eligible VHAP/LAM or selected fit/export, held-out QC, review UI, cancellation | E06–E07, E12, Q0 recipe decision | Real capture→approved asset, retry without duplicate publication, ≥3 identities pass quality; commercial conversion deferred |
| E14: deployment CLI/images | `deployment/`, `cli/doctor`, explicit provisioning, local/hosted profiles | E01, E08–E12 | Fresh host smoke from docs; hashes/terms checked; no downloads at startup; Fedora and container matrix verified |
| E15: admission and resource pools | reservations, fencing, per-device owner, shared weights, bounded queues | E09–E10, E14 | Concurrent admission cannot oversubscribe; stale owners cannot publish; load sweep determines actual capacity |
| E16: observability/operations | trace/metric schema, dashboards, redacted diagnostics, fault/runbook tests | E09–E15 | Queue/service/presentation trace chain, failure accounting, no biometric defaults, actionable alerts |
| E17: security and revocation | OIDC, roles, token grants, service identity, deletion receipts | E10–E16 | Cross-tenant tests, active asset revoke, still-valid token/rejoin behavior, bounded purge and restored tombstones |
| E18: release engineering | signed manifests/SBOM, migration compatibility, canary/drain, rollback/restore | E13–E17 | Real staging E2E, eight-hour soak, release compatibility and backup recovery; no unresolved critical gate |
| E19: virtual devices | host adapter and provisioning/runbook, paired virtual audio | E03, E09, E14 | Actual selected conference application, pacing/AEC/no-feedback, start/stop/device loss |
| E20: client-rendered tier | motion transport, mesh then Gaussian browser parity, asset export | E07, E11–E13, stable E02 | Local playhead sync and target-device thermal/quality/privacy gates; separately qualified optional profile |
| R01: challengers | LAM-A2E, EmoTaG, AVTR-1/Avatar Forcing, FlashHead, LiveAvatar reference, Drama speech reference; causal student only if needed | Q0 evidence gap or measured deficiency; resources for profile | Controlled independent ablation including NC models; no critical-path dependency on unavailable artifacts or unmeasured hardware |

## Milestone mapping and dependency gates

- **M0:** E00–E02. Create reproducible inputs and contracts; existing research is not an environment lock.
- **M1A:** E03–E04. Timing fixture and playback proof.
- **M1B/G1:** E05–E07 plus the E08 transport feasibility spike. Decide whether the visual/articulation path meets the target before training/enrollment investment.
- **M2:** E09 and a minimal E10/E11 session flow. Real one-user conversation, not an offline audio file demo.
- **M3:** finish E10–E13 and E19. Complete creation-to-use lifecycle and local virtual-device path.
- **M4:** E08 production path, E14–E16. Reproducible deployment and measured capacity with failures.
- **M5:** E20 is optional and separately gated; it does not delay a complete server-rendered product.
- **M6:** E17–E18. Security, data lifecycle, canary/drain/restore and release evidence. These acceptance criteria are mandatory even if security controls were implemented earlier.

The earlier phase effort estimates cover core research integration, not all production delivery. Revised rough planning range: **12–24 engineer-weeks** for one experienced engineer with graphics/ML/security review, assuming usable pretrained components and cleared capture assets; an independent high-quality enrollment pipeline can add substantial time. No delivery promise is made before G1 and environment integration. R tracks and broad device/language coverage are additional scope.

## End-to-end acceptance journeys

| ID | Journey | Pass artifact |
|---|---|---|
| J01 | Fresh checkout → explicit provision → local startup → browser conversation | Exact release manifest, install log, real-model trace, protected clip and documented host |
| J02 | Consent + capture/upload → preprocess/fit → evaluate/review → select avatar → converse | Immutable lineage, held-out report, UI/job transitions and actual output |
| J03 | Pause/hesitate → complete turn → useful response → intentional interrupt → corrected response | Annotated endpoint/interrupt records, no stale resumed speech, cutoff and latency statistics |
| J04 | Deny microphone/autoplay; change device; reconnect after network loss | Accessible user recovery, bounded retained resources, no stale replay |
| J05 | External browser with forced TURN and degraded bandwidth | ICE/relay evidence, quality mode transitions and measured latency; no localhost-only pass |
| J06 | N=1/2/4/8/16 offered load, burst/cancel/cache misses | Actual admitted capacity, failures/rejections/fairness and eight-hour stability |
| J07 | Wrong tenant/expired grant/revoked asset; cancel enrollment during publication | Access-denial records, source shutdown, no unauthorized derivative publication |
| J08 | Worker death/database loss/model failure | Fencing, explicit degraded/error behavior, no dual output or unbounded retry |
| J09 | Rolling upgrade → drain → rollback; restore backup in isolation | Compatibility/migration evidence, preserved active sessions or declared downtime, revoked assets remain revoked |
| J10 | Virtual camera + microphone into a target conference app | Actual app loopback clip and sync/feedback measurements |
| J11 | Optional client rendering on qualified desktop/mobile | Identity/motion parity, own-playhead timing, thermal and distribution-rights report |

J01–J10 are required for the stated complete scope; the server-only release may explicitly omit J10 only by declaring virtual-device support pending, not claiming the entire brief complete. J11 applies only to the optional client tier. A release checklist must point to stored evidence for each journey, not just a green checkbox.

## First implementation batch

Start **P00**: minimal E00/E01 core environment, CLI, read-only diagnostics and CPU validation. P01 builds contracts/replay; P02 delivers selected Qwen speech; P03 adds the avatar; P04 delivers browser conversation. Browser/native/database portions of E01 are implemented when needed, rather than as empty scaffolding. Use the [phase prompts](IMPLEMENTATION_PHASES.md) and [status ledger](IMPLEMENTATION_STATUS.md). The completion objective remains E18 plus required journeys; the first slice is not the whole product.
