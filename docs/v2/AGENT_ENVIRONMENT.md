# Repository-local agent environment

Reviewed 2026-10-26. The [root instructions](../../AGENTS.md) are the short persistent context. The [main plan](../../V2_RESEARCH_AND_BUILD_PLAN.md) is the architecture authority; detailed procedures live in the [validation protocol](VALIDATION_PROTOCOL.md) and focused skills.

The research, benchmarking and deployment skills now enforce [independent quality evidence/Q0](MODEL_QUALITY_BENCHMARKS.md): independent user/reproduction results precede author claims, incompatible score protocols stay separate, and NC models remain eligible. Reuse these focused skills; no additional generic model-ranking skill is necessary. Commercial clearance is a later distribution decision, separate from research deployment qualification.

For coding sessions, use the [P00–P09 task guide](IMPLEMENTATION_PHASES.md) and [persistent status](IMPLEMENTATION_STATUS.md). The delivery skill points to both. Qwen3-TTS is the accepted user choice; do not reopen the research survey as a prerequisite to coding. Keep phase results, commands and outstanding acceptance evidence in the status file, with protected media elsewhere. No extra Codex plugin or global configuration is needed to start P00.

## Configuration choices

Current official Codex documentation uses `.agents/skills/<name>/SKILL.md` with YAML `name` and `description`. These project skills need no global installation or additional MCP server. Root `AGENTS.md` supplies conventions within the normal ancestor/directory instruction hierarchy; keep it below the documented default 32 KiB project instruction budget.

Sources, initially accessed 2026-10-25: [Codex skills](https://developers.openai.com/codex/skills), [AGENTS.md guidance](https://learn.chatgpt.com/docs/agent-configuration/agents-md). Recheck official documentation before changing the format; local historical skills may describe older Cursor conventions.

Installed `openai-docs`, `skill-creator` and `create-rule` guidance was consulted. The older create-rule advice about `.cursor/rules` was not used to replace the user-requested AGENTS/Codex configuration. No unrelated global skills were copied. Existing ignored `.agent/skills/` phase prompts remain intact; they are historical V1 context, not the V2 authority.

The existing `.gitignore` ignored all `.agents/`. Its narrow exception now exposes only these six named skill directories and their SKILL.md files; other agent state remains ignored. No global settings, credentials, plugin registrations or MCP configuration were changed. No root project license was added.

## Skills

| Skill | Use |
|---|---|
| [avatar-research-evidence](../../.agents/skills/avatar-research-evidence/SKILL.md) | Verify model releases, implementation, licenses and benchmark boundaries before changing a selection |
| [avatar-realtime-benchmark](../../.agents/skills/avatar-realtime-benchmark/SKILL.md) | Design or assess latency/concurrency/quality measurements and promotion gates |
| [avatar-av-sync](../../.agents/skills/avatar-av-sync/SKILL.md) | Implement or debug sample accounting, clock mapping, EOF, drift and interruption |
| [avatar-gpu-media](../../.agents/skills/avatar-gpu-media/SKILL.md) | Analyze GPU ownership, memory, copies, compilation and render/encode scheduling |
| [avatar-e2e-delivery](../../.agents/skills/avatar-e2e-delivery/SKILL.md) | Implement complete product journeys and close work packages with real integration evidence |
| [avatar-deployment-release](../../.agents/skills/avatar-deployment-release/SKILL.md) | Qualify deployment, admission, revocation, drain, rollback and restore |

These are concise procedures referencing maintained project documents. They do not imply authorization for deployment, installation, model downloads or changing external services.

## Tools and environments

Existing shell, Python, git/rg, NVIDIA tooling, web search and GitHub connector access were sufficient for this assignment. Prefer primary papers, source/config/license files and official runtime documentation; inspect issues/reproductions as lower-ranked evidence with exact conditions. GitHub MCP is useful for repository files/history/issues, but plain read-only HTTPS/git can substitute. No mandatory proprietary MCP dependency is introduced.

Future implementation should provision separate locked environments for the Python application, legacy V1, A2F native/build tooling and enrollment. The full delivery scope is in [E2E_IMPLEMENTATION_SPEC.md](E2E_IMPLEMENTATION_SPEC.md), [IMPLEMENTATION_BACKLOG.md](IMPLEMENTATION_BACKLOG.md) and [DEPLOYMENT_AND_RELEASE.md](DEPLOYMENT_AND_RELEASE.md). Native development may need CMake, a compatible C++/CUDA toolchain, TensorRT/ONNX Runtime, gsplat, LiveKit C++/FFI, NVENC and WebRTC dependencies; GStreamer is the qualified fallback rather than a second mandatory stack. Install only what the current milestone requires. Nsight Systems/Compute, NVML and browser WebRTC statistics are useful optional profiling tools, not prerequisites for CPU contract tests.

Host baseline and limitations are recorded in [V1_AUDIT.md](V1_AUDIT.md). The audit's Python 3.14 and CUDA packages are observed state, not a supported lock. The proposed Python 3.12 application and CUDA 12.8 media build need validation during M0/M1. A2F auxiliary Python requirements belong in their own environment. New NeMo CUDA 13.2 benchmark binaries cannot be assumed compatible with the current driver.

Use environment variables or external secret management for credentials; do not commit secrets or full personal configuration. Default tests should be offline. Provision model assets deliberately with hashes and rights records; protected fixture access is separate from public source CI.

## Maintenance and review

For an architecture change, update the relevant ADR, evidence row and acceptance gate together. For a performance change, retain baseline and new raw records under declared conditions. For a new model/asset, update provenance and license records before promotion. For a skill change, validate frontmatter, links, scope and actionable behavior; do not accumulate generic instructions.

The installed skill creator's `scripts/quick_validate.py` can validate each skill when available. It is a development convenience, not a repository runtime dependency. Markdown links and Python syntax can be checked without running GPU probes. V1's full test collection blockers remain documented rather than concealed by configuration changes.

## October 26 scope revision validation

All six project skill definitions passed the installed skill-creator validator. Local Markdown links, fence/table structure, JSON artifacts and the existing probe syntax passed static checks; `git diff --check` passed. No V2 runtime tests, model executions or deployments were performed for this documentation-only revision. The earlier V1 measurements/test outcomes remain unchanged.
