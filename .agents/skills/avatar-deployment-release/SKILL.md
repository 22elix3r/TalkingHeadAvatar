---
name: avatar-deployment-release
description: Prepare or verify TalkingHeadAvatar local and hosted deployment, readiness, networking, admission, revocation, drain, rollback and release evidence.
---

# Avatar deployment and release

Use [DEPLOYMENT_AND_RELEASE.md](../../../docs/v2/DEPLOYMENT_AND_RELEASE.md) and journeys J01–J10 in [IMPLEMENTATION_BACKLOG.md](../../../docs/v2/IMPLEMENTATION_BACKLOG.md). Determine the exact target profile before selecting commands; planned CLI/config names are not evidence that those tools already exist.

The current target is research deployment. NC models may be primary dependencies; assess their actual research access conditions without requiring commercial clearance. Keep engineering qualification and commercial-distribution status separate. Use [independent quality/Q0](../../../docs/v2/MODEL_QUALITY_BENCHMARKS.md) for selection; commercial terms are not a quality score. Report deferred commercial rights without treating them as a failed research benchmark.

Validate a fresh environment using pinned images, dependencies, model/asset hashes and rights records. Separate explicit provisioning from service startup. Preserve host drivers, SELinux and global configuration unless the task authorizes their changes. Record local versus remote providers and the actual hardware/runtime matrix.

Check external-network WebRTC, forced TURN, TLS and token scope; an HTTPS health response is not a media test. LiveKit self-hosted participant removal does not invalidate all existing tokens: stop the source, close the room generation, deny grants and test rejoin behavior. Never publish sensitive media into a revoked room.

Distinguish liveness from readiness/admission. Warm profiles and encoder surfaces before new sessions. Test lease fencing, bounded queues, GPU/CPU/encoder limits, worker loss and cancellation under admitted load. Do not use average utilization or a successful N=1 demo as production capacity.

Verify upgrade/drain/rollback and isolated backup restore, including revocation tombstones. Match termination grace to the session policy; declare downtime where no spare capacity exists. Review the real-model/browser evidence, quality gates, SBOM/licenses and operator runbooks before calling a release deployable.

Report the qualified profile, evidence paths and unresolved blockers. Execute external deployment or destructive operations only within the current user's authorization; this procedure does not grant access or permission by itself.
