# Local session runtime

This separately locked Python 3.12 environment runs P04 FastAPI/PostgreSQL, CPU ASR/LLM adapters and the LiveKit reference. It does not alter the dependency-free core or P02 official-reference lock. See [P04 runbook and measured limits](../../docs/p04.md).

Model startup uses verified cached assets only. The initial Faster-Whisper dependency wheel bundled an unused VAD ONNX weight; its presence and the original no-download constraint deviation are recorded in the runbook/source lock. This is a loopback-only development profile with a synthetic voice and static reference, not a qualified speaking-avatar or hosted release.
