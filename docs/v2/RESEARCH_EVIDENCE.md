# Research evidence and candidate decisions

**Original cutoff and access date: 2026-10-25; deployment addendum and LAM license correction: 2026-10-26.** This is a deployment-oriented survey, not a claim to have reproduced every paper.
[Main plan](../../V2_RESEARCH_AND_BUILD_PLAN.md) · [V1 audit](V1_AUDIT.md) · [repository snapshots](repository-snapshots.json)

**Research-first revision, October 26:** [independent quality evidence and Q0](MODEL_QUALITY_BENCHMARKS.md) governs model priorities. NC models remain eligible for primary research use; commercial clearance is deferred. Rights notes below describe terms, not a quality ranking. Independent human/reproduction evidence takes precedence over author claims when protocols match; missing independent evidence leaves a choice provisional.

**Implementation choice:** Qwen3-TTS is selected by the user. Execute [P00–P09](IMPLEMENTATION_PHASES.md); other speech families remain deferred references. Model selection is distinct from runtime/quality acceptance.

## 1. Method and evidence ranking

Searches covered papers/project pages from CVPR, ICCV, ECCV, SIGGRAPH, AAAI, ICLR, arXiv and associated repositories; related-work trails from the mandatory candidates; Hugging Face weights/cards/discussions; source implementations, licenses, repository histories, issues and practitioner reports; speech and multimedia runtime documentation. Searches were extended into September/October 2026, rather than stopping at 2025 systems or the January 2026 release wave.

Evidence labels:

- **L — local observation:** a reproducible probe in this repository. Only V1 isolated operations reached this level.
- **I — implementation inspected:** source/API/config/license files support a specific capability or limitation. This does not establish speed or quality.
- **A — author report:** paper, official benchmark or model-card claim; useful, but not independent verification.
- **P — practitioner report:** first-hand implementation/issue with setup details. Not a controlled cross-model reproduction.
- **H — independent human evaluation:** third-party listening/viewing study or arena; disclose participants, task, uncertainty and version. Author-run user studies remain A.
- **R — independent reproduction:** third-party measured run with inspectable protocol/artifacts; disclose submitted-row provenance and limitations.
- **U — unresolved:** missing artifacts, unclear rights, incompatible timing boundaries or unverified assertion.

“NR” below means not reported or not established in the inspected sources, **not zero**. Unless a row explicitly says otherwise, total resident VRAM, p95/p99 first-output latency, production concurrency, training GPU-hours and exact full-pipeline parameter count are NR. Disk size and parameter count do not establish peak runtime memory. All candidates lack independently reproduced full conversational p95 measurements on the target 4070 Ti SUPER.

Release dates distinguish paper submission, explicit code/weight announcements, and current artifact availability. A repository creation date or last commit is not a release date. Snapshot SHAs are supplied for major repositories; they are evidence anchors, not a frozen dependency lock for an unimplemented V2.

## 2. Mandatory candidates

### 2.1 Instant Expressive Gaussian Head Avatars

- **Identity:** “Instant Expressive Gaussian Head Avatar via 3D-Aware Expression Distillation,” submitted **2025-12-18**, project labeled ECCV 2026.
- **Architecture:** image encoder creates an animatable Gaussian head; image-driven expression distillation avoids a fixed expression basis. This is an avatar reconstruction/reenactment component, not an audio-driven conversation stack.
- **Artifacts/rights:** paper, project and demonstrations available; no usable code/weight release or model license was established from the official project page as inspected.
- **Performance A:** ~20 ms image encoding and 107.31 FPS animation on RTX 6000 Ada. These are component measurements, not user endpoint-to-playback latency; resident VRAM and independent reproduction NR.
- **Decision:** research watchlist for rapid enrollment and expressive representation. Do not block V2 on an unreleased dependency; audio-to-latent driving and commercial rights remain separate work.

Sources: [paper](https://arxiv.org/abs/2512.16893), [official project](https://research.nvidia.com/labs/amri/projects/instant4d/).

### 2.2 AVTR-1

- **Release:** repository/model license identifies **2026-05-20**. Current source/weights and interactive demo exist; technical report and production backend remain unchecked in the repository roadmap.
- **Architecture I/A:** autoregressive flow-matching motion using two audio streams, with LivePortrait-derived rendering; 5-frame/200 ms chunks at 25 FPS, TensorRT/CUDA deployment.
- **Performance A:** chunk time 84 ms L40, 91 ms A100, 166 ms 4060 Ti, 202 ms L4, 232 ms 4060. Normalize as compute/audio-duration: 0.42, 0.455, 0.83, 1.01, 1.16 respectively. The repository's speed ratio uses the reciprocal convention. These do not include ASR/LLM/TTS/network.
- **Rights I:** community model license includes a USD10M revenue threshold **and a separate competing-product restriction**. Renderer and streamer are PolyForm Noncommercial with additional notices/patent provisions; InsightFace assets add restrictions. Being below the revenue threshold does not clear the whole stack.
- **Decision:** eligible primary research challenger for interactive behavior. Validate actual runtime, two-stream behavior and quality; record component-specific research access conditions. Commercial distribution is a later separate decision.

Sources: [code and benchmark](https://github.com/avaturn-live/avtr-1), [weights](https://huggingface.co/avaturn-live/avtr-1), [model license](https://github.com/avaturn-live/avtr-1/blob/main/LICENSE-MODEL.md), [renderer license](https://github.com/avaturn-live/avtr-1/blob/main/LICENSE-RENDERER.md), [streamer license](https://github.com/avaturn-live/avtr-1/blob/main/LICENSE-STREAMER.md).

### 2.3 EmoTaG

- **Publication:** **2026-03-22**, CVPR 2026. Source and pretrained-checkpoint download instructions are available by the audit date; first public artifact release date not established.
- **Architecture:** FLAME-Gaussian representation, gated residual motion network separates phonetic/emotional information; identity-conditioned adaptation and geometric regularization. Head pose/upper-face cues supplement what audio cannot determine.
- **Performance A:** RTX A6000; ~25 ms audio encoding **once per sequence**, ~6 ms motion plus ~7 ms rendering per frame. 60K initial Gaussians; pretraining 250K iterations, adaptation 20K. Five seconds means enrollment footage, not training time.
- **Maturity/rights I:** runnable research structure and checkpoint link, but no top-level LICENSE found; README limits use to research and imports GaussianAvatars/InsTaG/VHAP-derived components. Full streaming audio/first-frame delay and 16 GB training fit not established.
- **Decision:** strong architectural evidence for structured articulation plus expressive residuals; research comparator, not a directly shippable production foundation.

Sources: [paper and implementation details](https://arxiv.org/html/2603.21332v1), [code](https://github.com/jamesdemon923/EmoTaG), [project](https://emotag26.github.io/).

### 2.4 GaussianHeadTalk

- **Publication:** WACV 2026; repository commit **2025-12-10** is a documentation snapshot, not released inference.
- **Architecture A:** temporal audio modeling drives a parametric head/Gaussian renderer.
- **Artifacts I:** inspected repository contains README and .gitignore only. Training/inference code, pretrained weights and a usable distribution license were not established.
- **Performance:** no independently reproducible timing/VRAM/full-pipeline result established. Do not turn a demonstration into a deployable baseline.
- **Decision:** paper-level comparison only until artifacts, rights and exact causal context are available.

Sources: [official project](https://madhav1ag.github.io/gaussianheadtalk/), [repository](https://github.com/madhav1ag/GaussianHeadTalk).

### 2.5 Avatar Forcing — Ki et al. (CVPR 2026)

- **Publication:** arXiv **2026-01-02**, “Real-Time Interactive Head Avatar Generation for Natural Conversation.”
- **Architecture:** motion-latent diffusion forcing with user's audio/video and avatar audio; preference optimization targets responsive talking/listening behavior.
- **Performance A:** reported ~500 ms interaction latency / 6.8× speed improvement; not independently reproduced here.
- **Artifacts I:** source, motion autoencoder/flow model weights and minimal PyTorch inference supplied. README explicitly excludes the optimized real-time conversational system from this release.
- **Rights:** current LICENSE.md is **CC BY-NC 4.0**. Precise total model size, production VRAM and released-code reproduction of the accelerated demo are unresolved.
- **Decision:** eligible behavior-model contender, including for the primary research profile. Independent matched user-quality evidence remains unresolved; compare against bounded behavior and AVTR-1 before adoption.

Sources: [paper](https://arxiv.org/abs/2601.00664), [source and release scope](https://github.com/TaekyungKi/AvatarForcing), [license](https://github.com/TaekyungKi/AvatarForcing/blob/main/LICENSE.md).

### 2.6 AvatarForcing — Cui et al. (distinct work)

- **Publication:** **2026-03-15**, “One-Step Streaming Talking Avatars via Local-Future Sliding-Window Denoising.”
- **Architecture:** 1.3B Wan-based student, distribution-matching distillation, sliding local-future blocks and dual KV anchors.
- **Performance A:** 34 ms/output frame at 25 FPS, 832×480; default four blocks × four frames requires **480 ms future audio**. Paper reports ~510 ms audio-embedding-to-video delay; this starts after speech encoding availability and excludes conversation. Hardware type was not clearly established in the inspected timing text.
- **Artifacts I:** code and two training-stage checkpoints exist. First release date NR.
- **Rights I:** repository LICENSE.txt carries RollingForcing academic-only/no-production terms; HF card does not independently resolve derivative weight rights. Do not infer Apache rights from Wan alone.
- **Decision:** eligible bounded-lookahead research contender; its future-audio requirement must pass the selected latency profile. No NC exclusion; independent quality evidence remains unresolved.

Sources: [paper, Table 2](https://arxiv.org/html/2603.14331v1), [repository](https://github.com/KlingAIResearch/AvatarForcing), [license](https://github.com/KlingAIResearch/AvatarForcing/blob/main/LICENSE.txt), [weights](https://huggingface.co/lycui/AvatarForcing).

### 2.7 StreamAvatar

- **Publication:** **2025-12-26**, CVPR 2026.
- **Architecture:** autoregressive distilled video generation, reference sink, region-aware policy and consistency refinement; talking/listening and body motion.
- **Performance A:** 928×704 at 25 FPS on **two H800s**. First DiT ~330 ms, VAE ~390 ms, audio chunk 480 ms: **1.20 s avatar startup**, before ASR/LLM/TTS/network. Stage real-time factors are separate throughput measures.
- **Artifacts/rights:** official project gives paper/demos; usable code/weight release and license not established. Model size and production memory NR.
- **Decision:** fails the primary first-response budget on its own published boundary. Watchlist for higher-cost video quality mode; not a local default.

Sources: [project performance breakdown](https://streamavatar.github.io/), [paper](https://arxiv.org/abs/2512.22065).

### 2.8 Live Avatar — Alibaba Quark

- **Publication:** **2025-12-04**; README announces inference/weights **2025-12-08**, single-GPU route December 12, v1.1 January 20, 2026; ECCV 2026 Spotlight.
- **Architecture:** Wan2.2-S2V 14B with four-step acceleration and pipelined generation; long video consistency.
- **Performance A/I:** ~45 FPS is a multi-H800 result, typically five-GPU configuration. Single-GPU mode requires 80 GB; FP8 documentation lowers this to 48 GB. This is not a 16 GB deployment.
- **Artifacts:** source/weights exist; training and full TTS/interactive integration are not equivalent to complete released production serving.
- **Rights:** Apache-2.0 code/base claims still require exact weight/component audit before redistribution.
- **Decision:** first large-server visual-quality reference after the independent practitioner comparison in MODEL_QUALITY_BENCHMARKS.md. Keep outside the initial 16 GB profile; the reviewed large-GPU configuration does not establish local feasibility.

This is different from proprietary HeyGen **LiveAvatar**, which is a hosted product rather than the open model evaluated here.

Sources: [repository, hardware/release notes](https://github.com/Alibaba-Quark/LiveAvatar), [paper](https://arxiv.org/abs/2512.04677), [weights](https://huggingface.co/Quark-Vision/Live-Avatar).

### 2.9 NVIDIA Audio2Face-3D

- **Release:** model cards identify **2025-09-24** for Claire v2.3.1 and v3.0. SDK source and training framework available.
- **Architecture:** regression Claire **39.8M** parameters versus diffusion v3.0 **180M**, producing geometry/rig animation; not photorealistic pixels.
- **Causality I:** Claire network_info.json specifies buffer_len 8320, offset 4160, sample rate16000. SDK sets startOffset=-targetOffset: **520 ms centered window, 260 ms lookahead**. “Interactive” does not mean zero lookahead.
- **Deployability:** SDK supports CUDA streams and multiple tracks. Current build README needs CUDA12.8–12.x, TensorRT10.13–10.x; auxiliary Python tooling ≤3.10. Keep this isolated from the application environment.
- **Rights I:** SDK MIT, training framework Apache; weights NVIDIA Open Model License. A2Emotion and NIM have separate terms; neither is required in the minimal path.
- **Decision:** first pretrained articulation experiment, with explicit retargeting and first-future-audio budget. No local throughput or peak VRAM result claimed.

Sources: [SDK](https://github.com/NVIDIA/Audio2Face-3D-SDK), [Claire card](https://huggingface.co/nvidia/Audio2Face-3D-v2.3.1-Claire), [network configuration](https://huggingface.co/nvidia/Audio2Face-3D-v2.3.1-Claire/blob/main/network_info.json), [offset implementation](https://github.com/NVIDIA/Audio2Face-3D-SDK/blob/main/audio2face-sdk/source/audio2face-core/executor_regression_core.cpp), [v3 card](https://huggingface.co/nvidia/Audio2Face-3D-v3.0).

### 2.10 StreamingTalker

- **Publication:** November 2025 preprint, AAAI2026 oral; source/ModelScope weights available by audit; exact first artifact release date NR.
- **Architecture:** VQ motion latent, autoregressive condition transformer and iterative diffusion head. VOCASET/BIWI vertex outputs require retargeting; not FLAME coefficients.
- **Performance A:** paper reports 25 ms first model output and 20 ms subsequent motion-frame compute; training ~12 h on one4090. This excludes conversational ASR/LLM/TTS; paper demo generates response audio before animation. README GPU guidance is stricter than paper's3090 demo statement. Inspect HuBERT preprocessing rather than assuming causal features.
- **Rights I:** **current LICENSE is Project Registration License v1.0**, requiring organizational project registration, including development/evaluation, and granting no patent license. Do not label it Apache based on older metadata or third-party summaries. Dataset/weights terms need separate review.
- **Decision:** compare directly with A2F on a matched rig in Q0. Registration is a concrete access requirement, not a commercial-license exclusion. Author geometry metrics do not establish superiority in rendered mouth quality; record actual research access steps before execution.

Sources: [paper appendix](https://arxiv.org/html/2511.14223v2), [code](https://github.com/zju3dv/StreamingTalker), [actual license](https://github.com/zju3dv/StreamingTalker/blob/main/LICENSE).

## 3. Strong alternatives and related work

| Candidate / sources | Contribution and availability | Deployment conclusion |
|---|---|---|
| [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead), [paper](https://arxiv.org/abs/2602.07449), [weight terms](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B) | Paper Feb7, code/weights Feb12 2026. 1.3B video model; cached audio context and oracle distillation. Author: Lite96FPS / three25FPS streams on4090; Pro10.8FPS4090 or real-time two5090s. Code/DiT/WanVAE Apache; Lite LTX-VAE **Open-RAIL-M**. | **Local feasibility challenger with reported mouth-quality risk**, not a visual-quality winner; see independent evidence in MODEL_QUALITY_BENCHMARKS.md. Actual config33frames25FPS and demo groups three chunks into MP4 segments. Neither96FPS nor a “streaming” Gradio label proves first audiovisual latency. Peak16GB fit, block fill, cancellation and long-run quality require measurement. |
| [LAM](https://github.com/aigc3d/LAM), [paper](https://arxiv.org/abs/2502.17796), [Web renderer](https://github.com/aigc3d/LAM_WebRender) | SIGGRAPH2025 one-shot animatable3DGS; released LAM20K weights and April/May2025 Audio2Expression/browser integrations. Author table1.4s enrollment,562.9FPS and mobile110+FPS under own configurations. | **Eligible primary research enrollment contender.** October 26 correction: [LICENSE_WEIGHT](https://github.com/aigc3d/LAM/blob/339573649dd93df4cba8093a964e85a80d1b61f3/LICENSE_WEIGHT), added June 10, is CC-BY-NC-4.0. Apache code does not clear these weights, FLAME/data or the imported Inria rasterizer. Do not exclude from research; commercial distribution is a separate later decision. |
| [MeshLAM](https://arxiv.org/abs/2604.22865), [project](https://meshlam.github.io/) | April23 2026/CVPR2026; textured mesh,8Kvertices, sub-second reconstruction claim. Project “Code” resolves to LAM repository, not a separately verified MeshLAM release. | Promising client-rendered alternative. Do not assume MeshLAM weights are supplied by the older LAM release. Verify artifacts before adoption. |
| [PanoLAM](https://arxiv.org/abs/2509.07552) | Sep2025; v3 Oct10 2026, full-head single-image Gaussian reconstruction using synthetic priors. Latest abstract describes **static** reconstruction. | Useful enrollment prior, not evidence of animatable head motion by itself. |
| [InsTaG](https://github.com/Fictionarry/InsTaG), [paper](https://arxiv.org/abs/2502.20387) | CVPR2025 person-independent motion priors plus few-second personalization. Code exists; face/mouth separation informs residual design. | Five-second footage is not five-second compute. Compare held-out quality and audio context in research; preserve data/renderer terms for later distribution review. |
| [GaussianSpeech](https://arxiv.org/abs/2411.18675) | ICCV2025 personalized Gaussian speech animation; rich multiview training/supervision and expressive details. | Quality reference, not minimal-footage enrollment or proven low-cost conversational serving. Training capture and exact assets must be budgeted. |
| [Beyond Monologue](https://arxiv.org/abs/2604.10367), [project](https://warmcongee.github.io/beyond-monologue/) | Apr11 2026; different temporal receptive fields for speech articulation and listening reactions, dual-stream audio, VoxHear. | Supports separating fast phonetic alignment from slower context-dependent behavior. “Gaussian kernels” here describe temporal weighting, not necessarily3DGS representation. Artifact/production rights unresolved. |
| [Omni-LiveAvatar](https://arxiv.org/abs/2608.13602), [repo](https://github.com/Aoko955/Omni-LiveAvatar) | Aug2026 joint audio/video distillation from LTX-2; author21.99FPS singleH200 and33×teacher speed. Repository explicitly says code/checkpoints not released. | Research watchlist. Relative speedup is not first-response latency, and joint generation compromises independent TTS replacement. |
| [Vorch-Streamer](https://arxiv.org/abs/2608.05663), [project](https://vorch-project.github.io/Vorch-Streamer-project/) | Aug2026 long-form text-to-audio/video,22.8B and speech-planning tokens. | Interesting cross-modal conditioning; too large/insufficient deployment evidence for local primary. No small-model capacity assumed. |
| [MuseTalk](https://github.com/TMElyralab/MuseTalk), [paper](https://arxiv.org/abs/2410.10122) | Released2024;1.5Mar28 2025; single-pass latent mouth inpainting, precomputed avatar imagery,30+FPSV100 author claim. Code MIT, dependency/weight licenses separate. | Useful low-cost **2D lip-sync fallback experiment**, not full novel head/gaze generation. Inspect Whisper feature buffering and full-face compositing. Cannot infer3050 or4070 performance fromV100. |
| [LivePortrait](https://github.com/KwaiVGI/LivePortrait) | Efficient learned motion/appearance reenactment; underlies several newer interactive models. | An image/video driver is still needed. Code license does not license InsightFace pretrained detector assets. Replacing detection alone does not prove all rights clear. |
| [gsplat](https://github.com/nerfstudio-project/gsplat), [paper](https://arxiv.org/abs/2409.06765) | Maintained Apache2 CUDA renderer,3DGS/2DGS and batching-oriented APIs. Source exposes legacy external renderer wrappers too. | **Portable rendering target**, specifically native3DGS. Research vendor backends are eligible with explicit provenance; native migration follows quality validation. |
| [TileGS](https://arxiv.org/abs/2609.03613) | Sep3 2026 tile-local depth binning. Author raster kernel1.44× but complete frame only1.069×4090 /1.094×1000Ada across9scenes. | Optimization watchlist after profiling. Excellent warning against extrapolating kernel speedups to conversation gains. No avatar benchmark. |
| [Spark](https://sparkjs.dev/docs/splat-mesh/), [PlayCanvas](https://developer.playcanvas.com/user-manual/graphics/) | Spark exposes dual-quaternion splat skinning and shader modifiers. PlayCanvas supports WebGPU/WebGL fallback and active GS tooling. | Viable browser targets; neither automatically reproduces FLAME expression deformation, covariance updates or sorting. Start renderer parity prototype, not a new browser engine. |
| [WebSplatter](https://arxiv.org/abs/2602.03207), [Visionary](https://arxiv.org/abs/2512.08478) | Cross-device WebGPU splatting and a per-frame Gaussian generator interface. | Additional portability/system references. Static scene throughput cannot establish face-deformation quality or mobile thermal stability. |

### Direct-video reproduction evidence

[FlashHead issue18](https://github.com/Soul-AILab/SoulX-FlashHead/issues/18) contains a4090 user's ~61.64s pipeline time for52s output including face handling, super-resolution and compositing. Maintainers point to compilation/warmup and recommended environment; one commenter misstates the published multi-GPU requirement. This is **P**, not a refutation or reproduction of the isolated96FPS result. The boundaries differ. [Issue21](https://github.com/Soul-AILab/SoulX-FlashHead/issues/21) reports identity-dependent mouth/teeth artifacts with config/videos. It motivates held-out face diversity and mouth-crop review, not an asserted failure rate.

The inspected [Gradio streaming source](https://github.com/Soul-AILab/SoulX-FlashHead/blob/main/gradio_app_streaming.py) loads audio, processes chunks, copies frames to CPU and groups three chunks for MP4 delivery. Replace that delivery path for any live experiment and **measure** first usable output with incrementally available audio; do not subtract chunk-fill time.

## 4. Speech and turn-taking choices

| Component | Evidence, rights and current limitations | Decision |
|---|---|---|
| [sherpa-onnx](https://k2-fsa.github.io/sherpa/), [English Zipformer checkpoint](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26) | Mature stateful streaming API; specific English checkpoint card Apache2. Older model is a reproducibility baseline, not current accuracy winner. Chunk/context is model-specific. | First CPU ASR fixture. Select the lowest tested latency configuration meeting accuracy; compare newer Nemotron before production lock. |
| [Nemotron3.5 streaming0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | ReleasedJune4 2026; cache-aware FastConformerRNNT,80–1120ms configurable chunks, OpenMDW1.1. Card distinguishes32 usable locales from8 needing adaptation. Author H100 concurrency is ASR-only. | Strong server/multilingual candidate. English-specific predecessor also compare locally. Do not use1120ms mode in sub-second profile. |
| [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), [report](https://arxiv.org/abs/2601.21337) | Jan29 2026,0.6B/1.7B Apache2, streaming toolkit/forced aligner. | Multilingual alternative; measure actual finalization and resource use, not just offline WER. Forced alignment is offline evaluation, not a free online phoneme clock. |
| [Silero VAD](https://github.com/snakers4/silero-vad), [Smart Turn v3](https://huggingface.co/pipecat-ai/smart-turn-v3) | Speech activity plus acoustic turn completion. Smart Turn8M,WhisperTiny-derived, int8ONNX8MB, BSD2. Needs pause context; advertised inference time excludes silence waiting. | CPU VAD + semantic endpoint experiment, calibrated for pauses/hesitations. Endpoint timeout cannot be eliminated by a fast classifier. |
| [LiveKit turn tuning](https://docs.livekit.io/agents/logic/turns/tuning/) | Documents speculation, false-interruption recovery and adaptive turn policies. Some enhanced features are cloud-specific. | Use concepts and existing transport; implement explicit local fallback. Do not silently require paid/cloud interruption inference. |
| [Pocket TTS](https://github.com/kyutai-labs/pocket-tts), [weights](https://huggingface.co/kyutai/pocket-tts) | Jan2026 original100M, CPU streaming and voice state reuse; author~200msTTFA. MIT code; CC-BY4 weights plus gated use/consent conditions; voices have their own rights. | **Deferred CPU comparison candidate**; Qwen3-TTS is selected for immediate implementation. Do not add this adapter to the first slice. |
| [Pocket small models](https://kyutai.org/blog/2026-10-20-pocket-tts-small/) | Oct20 2026 announcement; repo advertises18.2M/6.9M/1.9M models and up-to25×CPU speed. | New low-resource challenger; only five days old at cutoff. No automatic quality equivalence or production promotion. |
| [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), [API source](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py) | Jan22 2026;0.6B/1.7B,Apache2; author97ms architecture claim. API docstrings say non_streaming_mode=False **simulates** text streaming, not true streaming input/output generation. | **Selected initial TTS:** Base1.7B, X-Vector first; qualify a streaming engine and official-reference parity in P02. ICL is an explicit option. Reject whole-waveform wrapper as low-latency adapter. |
| [Qwen3TTS-Streaming](https://github.com/X-Square-Robot/Qwen3TTS-Streaming), [native C++ alternative](https://github.com/offgridai/qwen3-tts-cpp-streaming) | Independent implementations expose incremental callbacks; reported results depend on prefix hits, text batching, precision, packet size and hardware. | Prototype one maintained/pinned engine after API/license audit. Do not average their~24ms5090 cache-hit figure with another engine's hundreds-of-ms result. |
| [Magpie357M](https://huggingface.co/nvidia/magpie_tts_multilingual_357m), [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp), [benchmark protocol](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/BENCHMARK.md) | v2607 July21 2026. NVIDIA Open Model License; not a zero-shot cloning model. Native runtime author benchmark Q8 gives8.6ms average/9.4msp99TTFA4090 versus203.4/238.2msCPU,186msPCM chunks. | **Deferred fixed-voice latency comparator**, not an initial implementation dependency. Small20-request protocol andCUDA13.2/driver595.84 differ from localCUDA12.8/driver580; do not promise those numbers or upgrade host automatically. |
| [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | Streaming family and multilingual cloning; model-specific tokenizer/vocoder/license and text-input buffering matter. | Secondary challenger if Pocket/Qwen/Magpie cannot meet language/prosody needs; no throughput assumed. |
| [Fish Speech](https://github.com/fishaudio/fish-speech/blob/main/LICENSE), [S2 report](https://arxiv.org/abs/2603.08823) | Strong author latency/expressivity claims; October13 2026 community source license imposes revenue and hosted-service conditions; S2-Pro weights separately restricted. | Eligible research quality candidate. Prioritize Drama 3 on a larger-GPU profile and S2-Pro as reference using independent evidence; current documented requirements are 48/24 GB respectively. Historical license descriptions still need a pinned revision. |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Apache2 small-model candidate with multiple runtimes; hybrid/multimodal implementation differs from old Gemma. | Local text-only4-bit streaming trial, bounded context and non-reasoning conversational mode. Exact quantization/provider quality and cancellation support are gates. |
| [PersonaPlex7B](https://huggingface.co/nvidia/personaplex-7b-v1), [Moshi](https://github.com/kyutai-labs/moshi) | PersonaPlexJan15 2026, dual-stream24kHz speech understanding/generation, NVIDIA Open Model License;7B weights alone~14GBbf16 before codecs/state. | Experimental full-duplex backend on separate/larger GPU. Not combined with renderer on16GB by assumption; tool control, transcript fidelity and independent TTS replacement differ from cascade. |

### Practitioner evidence on speech

[Qwen model discussion4](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/discussions/4) reports inability to access advertised local streaming. It agrees with the inspected API limitation, without establishing every later fork's behavior. [Offgrid's July27 first-hand release](https://www.reddit.com/r/LLMDevs/comments/1v78zbz/qwen3tts_native_c_streaming_and_voice_cloning/) and linked code distinguish inference callbacks from playback buffering. [FluidAudio benchmark methodology](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/TTS/Benchmarks.md) explicitly labels batch TTS TTFT as full-waveform completion; its Apple hardware is not comparable to this CUDA machine.

No community comment, GitHub star count, demo video or self-promotional post is used as proof of system capacity. No compatible independent head-to-head full conversational avatar benchmark was found in the inspected evidence. That gap is a reason to build the validation harness first.

## 5. Representation comparison and selected direction

| Architecture | Identity/control | Latency and scheduling | Enrollment | Scaling and risks | Role |
|---|---|---|---|---|---|
| FLAME/rig-driven3DGS | Explicit pose/eyes/mouth, stable identity, weak unseen oral detail if enrollment poor | Small motion packets; independent25/50Hz motion and30/60Hz render; bounded per-frame work | Calibrated multi-minute route; one-shot priors optional | Per-avatar assets, raster/deformation bandwidth; license and rig quality | **Integration baseline**, with research-eligible rig/renderer; final model chosen by independent Q0 quality and profile feasibility |
| Modern latent/deformableGS | More expressive/hair/skin detail; latent controls less interpretable | Fast renderer but driver may need future audio or video | Strong one-shot potential | Unreleased weights, hidden context, bespoke training/caches | Experimental enrollment/driver |
| Direct diffusion/flow video | Strong portrait realism/body coverage; identity drift and mouth artifacts possible | Chunk acquisition+denoising+VAE; reset/cancel boundaries matter | Single-image easy | Large per-session state, poor multi-tenant tail isolation; output-only video | FlashHead challenger; expensive mode |
| Mesh+learned residual | Explicit oral/eye geometry plus wrinkles/appearance | Deterministic fallback, residual optional per frame | More calibration or feedforward initialization | Residual can add flicker/latency; easier client tier | Primary extension after baseline |
| Client rendering from motion | Lowest server pixel cost, camera freedom | Audio playout must drive browser motion clock | Download/compress assets once | Client GPU/thermal variability; biometric asset exposure; sorting parity | **Scale-oriented optional tier**, mesh first thenGS |
| Pure textured mesh | Strong controls, conventional tooling | Predictable cheap render | Can start from artist/consented rig | Lower photorealism/hair quality than strong neural renderers | Functional fallback and timing fixture |
| Joint speech/video foundation model | Natural audiovisual coupling | Fewer explicit stages but large joint generator and chunk delays | Often single-image prompt | Less component replaceability; safety/tool/control/rights maturity | Research-only backend |

The recommendation is **not** that FLAME/3DGS is already the visual winner. It offers bounded scheduling, interruption recovery and explicit controls. Use independent/blinded Q0 comparisons to establish mouth/identity quality. If a direct-video challenger wins within the selected profile's latency/capacity gates, change the ADR. Research eligibility includes NC components; native gsplat portability does not justify discarding a better research model.

## 6. Licensing conclusions

A production-capable open-source **application** can support separately licensed model plugins. It must not describe all bundled weights as OSI open source.

- [FLAME model terms](https://flame.is.tue.mpg.de/modellicense.html) distinguish old2017–2023 academic models from **FLAME2023Open**. The [official news](https://flame.is.tue.mpg.de/) records the Open release in November2025 and conversion tools. Commercial use is granted for Open, with attribution and additional use language. Review exact downloaded terms; old checkpoints and texture/data rights do not automatically change.
- V1 GaussianAvatars CC-BY-NC-SA and Inria research rasterizer are incompatible with an assumed unrestricted commercial bundle. VHAP and pretrained tracking assets need their own clearance.
- Top-level Apache/MIT is insufficient for LAM, LivePortrait-derived stacks or a GS wrapper exposing a differently licensed backend.
- Dataset download permission, model-weight license, consent/likeness rights, derivative code, voice-reference license and output usage are separate records.
- No new broad root LICENSE is added in this planning assignment: ownership of existing project modifications has not been established. Proposed new independently authored core: Apache2, subject to owner selection and provenance review. Preserve upstream notices.

## 7. Infrastructure evidence used in the design

| Source | Supported engineering conclusion |
|---|---|
| [TensorRT optimization guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html) | Graph/stream optimizations require compatible memory lifetimes/shapes and can be broken by implicit synchronization. Benchmark resource-sharing configuration, not only isolated engines. |
| [ONNX Runtime CUDA provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html), [I/O binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html) | Bind device buffers and explicitly validate provider placement; graph capture is not a blanket speed flag. |
| [vLLM prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | Reuse shared prompt prefixes; this reduces prefill, not arbitrary token-generation time. Bound KV/state pools and isolate tenant cache keys. |
| [GStreamer CUDA conversion](https://gstreamer.freedesktop.org/documentation/nvcodec/cudaconvert.html), [nvcodec elements](https://gstreamer.freedesktop.org/documentation/nvcodec/index.html) | CUDA-memory conversion/encoding paths exist. A Python RGB frame passed to a CPU-facing SDK does not become zero-copy automatically. |
| [NVIDIA encode support matrix](https://developer.nvidia.com/video-encode-decode-support-matrix), [NVENC note](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvenc-application-note/index.html) | Current matrix lists4070TiSuper two encoders and12sessions; installed driver/runtime must be queried/tested. Encoder count and session cap are distinct from sustainable throughput. Never assume A100/H100 inference nodes have usable NVENC. |
| [RFC8834](https://www.rfc-editor.org/rfc/rfc8834.html) | RTP media clocks require RTCP synchronization context; timestamps across audio/video are not directly comparable without mapping. |
| [Codex skills](https://developers.openai.com/codex/skills), [AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md) | Current repo-local skill path is .agents/skills with SKILL.md name/description metadata; project instructions use AGENTS.md. No global settings needed. |

## 8. Uncertainty register

1. No shortlisted new avatar model was executed locally; no new weights were downloaded.
2. No consent-cleared production avatar dataset is established in this workspace.
3. A2F retargeting quality, first260ms-future-audio availability, and multitrack memory remain unmeasured.
4. No generally compatible phoneme/viseme timestamps are guaranteed by the selected neural TTS APIs.
5. Server GPU throughput/cost and browser mobile quality are projections until the shared harness runs.
6. Weight availability can be gated or moved; pin revisions/hashes and archive license text at acquisition.
7. Missing/contradictory terms must be recorded for the specific access/distribution action. Unresolved commercial clearance does not block research selection; public access does not imply unrestricted rights.
8. Benchmark claims use different resolution, scene complexity, precision, chunk lengths, batch sizes and warming. There is no defensible global FPS leaderboard across this table.
9. Speech systems released within days of cutoff are challengers, not mature defaults on novelty alone.
10. Website statements, repository defaults and release histories can differ. Integration must review the exact pinned code and dependencies, and update this ledger with new evidence.


## 9. Deployment scope follow-up

The [deployment research addendum](DEPLOYMENT_RESEARCH.md) verifies native pre-encoded LiveKit publishing, GStreamer alternatives, SDK queue/timestamp limitations, OpenAvatarChat and LAM-A2E implementation boundaries, current LAM weight restrictions, infrastructure and framework tradeoffs. The [E2E specification](E2E_IMPLEMENTATION_SPEC.md) and [release specification](DEPLOYMENT_AND_RELEASE.md) translate those findings into complete product and operational contracts. These are new source findings, not new measured model performance.
