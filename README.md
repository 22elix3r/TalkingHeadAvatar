# TalkingHeadAvatar V2

The V2 research implementation lives in [`v2/`](v2/). `main` is the default V2 development branch. The V1 runtime and its original history are preserved on [`v1/legacy`](https://github.com/22elix3r/TalkingHeadAvatar/tree/v1/legacy).

Start with [implementation status](docs/v2/IMPLEMENTATION_STATUS.md), [phase requirements](docs/v2/IMPLEMENTATION_PHASES.md), and the [V2 build plan](V2_RESEARCH_AND_BUILD_PLAN.md). [V2 setup](v2/README.md) describes the isolated Python/runtime layout; [local browser reference](v2/docs/p04.md) and [local profiles](v2/docs/p06-local.md) document runnable commands.

Current evidence includes real local Qwen streaming/browser audio, durable static-asset import and revocation, native LiveKit/CUDA/NVENC qualification, and direct full-size Claire GPU rendering. Live speaking-avatar integration, physical A/V and latency, independent quality, human enrollment and release gates remain in progress. Voice polish is deferred.

The core contract suite runs offline without models:

```sh
cd v2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest -q
```

Provisioning model weights is always explicit. The authorized Claire package/runtime exception and the remaining download constraints are recorded in implementation status. Keep private artifacts, source assets and runtime environments out of Git. Existing root `data/` and `output/` placeholders/local caches remain reserved; V2 generated artifacts live under ignored `v2/var/`.

[History repair and branch layout](docs/v2/REPOSITORY_HISTORY.md) explains the scoped attribution/signature correction and maps previous V2 commit IDs to their replacements. It preserves separate phase commits and legacy history.
