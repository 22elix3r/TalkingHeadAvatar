# Model quality: independent evidence and selection decisions

Revision/access date: **2026-10-26**. This extends the original October 25 survey with live leaderboard observations; it is **not** a reconstruction of those boards on October 25. No model-quality experiment was run locally in this documentation revision. The V1 microbenchmarks measure execution, not quality.

This document governs model promotion alongside the [main plan](../../V2_RESEARCH_AND_BUILD_PLAN.md) and [validation protocol](VALIDATION_PROTOCOL.md). The project is **research-first**: noncommercial models are eligible on the same quality and engineering criteria as permissive models. Record terms and provenance; defer commercial clearance to a later commercial-distribution decision. Actual access/registration requirements and consent still apply to experiments. A missing commercial grant is not a quality penalty.

**Implementation update:** the user has selected Qwen3-TTS now. Follow [P00–P09](IMPLEMENTATION_PHASES.md); do not require a TTS tournament before coding. Scores below remain research evidence, not a claim that the chosen runtime has passed. Q0 is now staged quality qualification, with other TTS families deferred.

## 1. What counts as independent evidence

| Class | Evidence | Use in selection |
|---|---|---|
| Independent human comparison | Evaluator independent of the model authors; blinded votes or documented listening/viewing study | Strongest external quality signal when the task, version and cohort match |
| Independent reproducible evaluation | Third-party runner, identified inputs/configuration and inspectable outputs/results | Objective accuracy/quality evidence; not automatically human preference |
| Practitioner observation | User's own execution, examples/configuration, experience report | Useful failure cases and feasibility; never manufacture a population score from it |
| Author evaluation | Paper tables, author-run user studies, model cards, vendor demos | Hypothesis and experiment prioritization only; cannot establish an independent winner |
| Aggregation/marketing | Stars, downloads, copied paper tables, unexecuted comparison articles | Discovery only; no quality score |

Independence concerns who ran the evaluation, not where it is hosted. A model author's MOS study remains author evidence even with external participants. A leaderboard can include submitted results: check row provenance before calling it independently reproduced. A hosted endpoint and similarly named downloadable checkpoint are not necessarily identical. Keep exact checkpoint/runtime, sampling, precision, stream mode, language, voice and hardware in every local result.

**Selection rule:** choose the strongest supported quality candidate **among models meeting the target profile's measured timing, memory and reliability limits**. Keep the highest-quality larger model as a separate reference. Do not average FID, WER, Elo, FPS and anecdotes into a made-up overall score. A lack of independent evidence means *unresolved*, not *bad* or *best*.

## 2. TTS: independent listener evidence changes the defaults

### Provider-voice preference

Selected observations from [Artificial Analysis' open-weights Provider Voice Arena](https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice/open-weights), accessed October 26. Higher Elo is preferred; intervals shown are the board's 95% intervals. “Samples” is the board field, not a verified count of unique listeners.

| Exact board label | Elo | 95% interval | Samples |
|---|---:|---:|---:|
| Fish Audio Drama 3 Preview | 1246 | 1232–1260 | 2,256 |
| Breeze TTS 2 | 1230 | 1217–1243 | 2,407 |
| Ming-3.0-tts | 1189 | 1175–1203 | 2,212 |
| Fish Audio S2 Pro | 1109 | 1097–1121 | 2,764 |
| Kokoro 82M v1.0 | 1063 | 1052–1074 | 5,461 |
| Chatterbox | 1026 | 1015–1037 | 4,735 |

This board compares each provider's selected voices; voice appeal confounds model quality. Drama and Breeze intervals overlap, so do not declare a decisive universal winner. Coverage is incomplete. These are endpoint observations, not measurements of our quantized streaming implementation.

### Cloning: use the controlled-voice board

Selected [Controlled Voice Arena](https://artificialanalysis.ai/text-to-speech/leaderboard/controlled-voice) observations, English, the same eight cloned voices (four US/four UK), accessed October 26:

| Exact board label | Elo | 95% interval | Samples |
|---|---:|---:|---:|
| Fish Audio Drama 3 Preview | 1057 | 1043–1071 | 2,331 |
| Breeze TTS 2 | 1021 | 1008–1034 | 2,787 |
| Voxtral TTS | 1005 | 992–1018 | 3,099 |
| Fish Audio S2 Pro | 1000 | 988–1012 | 4,686 |
| Chatterbox | 933 | 919–947 | 2,841 |

Drama has stronger evidence here than the older Fish checkpoint and Chatterbox. Breeze/Voxtral/S2 intervals overlap; their point estimates do not establish a decisive ordering. **Do not compare Elo values between the two boards:** anchors, voters and comparison graphs differ. Do not transfer Qwen-Audio hosted-service scores to Qwen3-TTS weights. [Evaluator methodology](https://artificialanalysis.ai/methodology/text-to-speech).

### Independent Qwen3-TTS study

[Baroudi et al., October 16, 2026, Table 1](https://arxiv.org/html/2610.23678v1) evaluates public Qwen3-TTS-12Hz-1.7B-Base, Chatterbox, XTTS-v2 and F5-TTS. It is separate from Qwen's model report. On AMI, the fraction of trials where listeners selected the synthetic clip as real was:

| System | Human realism/fooling rate ↑ |
|---|---:|
| Qwen3-TTS X-Vector | 35.6% |
| Qwen3-TTS ICL | 27.0% |
| Chatterbox | 9.0% |
| XTTS-v2 | 6.0% |
| F5-TTS | 1.0% |

The study uses only **three annotators** and 100 sampled AMI pairs per system; this is not broad population preference or a conversational deployment test. Other domain columns use an LLM judge, not humans. Qwen's realism advantage supports a local quality trial, while reported prosody limitations and reference-transcript sensitivity require targeted tests. Fooling rate measures realism, not permission to conceal synthetic speech. There is no matched comparison against Drama, PocketTTS or StreamingTalker in this study.

### Independent objective evidence and resource feasibility

[Hugging Face's Open TTS evaluation announcement/source](https://github.com/huggingface/blog/blob/main/open-tts-leaderboard.md) describes macro-average English WER on Seed TTS Eval and CV3 Eval, Qwen3-ASR scoring, WavLM speaker similarity and output listening. It identifies Kokoro, Supertonic 3 and Fish S2-Pro as strong English WER models. This is complementary intelligibility evidence, **not a published human preference score**. No precise numeric rank is inferred from its plots here; contribution provenance must be checked per run. Use the [live evaluation](https://huggingface.co/spaces/hf-audio/open_tts_leaderboard) to select future fixtures and inspect outputs.

The [Fish installation requirements](https://speech.fish.audio/install/) currently specify at least **48 GB for Drama 3 FP8** and **24 GB for S2-Pro**. These are vendor requirements, not independently measured peaks. Neither belongs in the 16 GB co-resident default without a separately tested smaller/quantized implementation. Preview endpoint-to-weight equivalence also needs verification. Research-license eligibility does not solve memory pressure.

**Current decision:** implement Qwen3-TTS-12Hz-1.7B-Base now, X-Vector first and ICL as an explicit option. Qualify official-reference quality and the actual streaming backend in P02; do not integrate other TTS families first. Chatterbox, Pocket, Kokoro, Drama, S2-Pro, Breeze and Ming remain deferred comparison candidates. This user-selected implementation path does not assert universal superiority or erase the independent evidence above.

## 3. Avatars: actual usage contradicts a speed-only preference

[Tight Studio's August 23, 2026 execution report](https://tight.studio/blog/open-source-talking-avatar-models/) uses one portrait and a 3.63-second narration, with output videos and configuration descriptions. Its reviewers favor LiveAvatar's sharpness/mouth quality, while FlashHead Lite has weaker identity and mouth detail.

| Tested configuration | Reported peak VRAM | Evidence scope |
|---|---:|---|
| LiveAvatar, H200 | 61,401 MiB | Strongest large-model visual result in this review |
| EchoMimicV3-Flash, L40S | 33,726 MiB | Favorable practical visual result at that configuration |
| SoulX-FlashHead Lite, L40S | 5,763 MiB | Small footprint; weakest visual result in this review |

This is a practitioner's own comparison, **not blinded MOS**, and uses different resolutions/settings. Timings include startup through MP4 return, so they do not refute or verify warm streaming p95. One face and short clip cannot establish generality. No numeric user quality score is available. The reviewed configurations do not establish 4070 Ti SUPER performance.

Independent [FlashHead user issue #21](https://github.com/Soul-AILab/SoulX-FlashHead/issues/21), opened March 31, includes configuration/examples and reports blurry mouth detail, disappearing/distorted teeth and face-dependent artifacts. It is an actionable regression case, not an estimated failure frequency. Neither a complaint count nor a positive anecdote substitutes for blinded multi-identity testing.

**Decision:** LiveAvatar becomes the first **server visual-quality reference**, EchoMimicV3-Flash the medium/large-GPU comparator. FlashHead Lite remains a **local feasibility challenger with an explicit mouth-quality risk**, not the preferred visual-quality model. No independent matched evidence found here establishes a quality winner among A2F, StreamingTalker, EmoTaG, GaussianAvatars, LAM, AVTR-1 or the two Avatar Forcing systems. Keep their comparisons open; do not use paper scores to override this gap.

## 4. Author benchmarks retained for experiment design, not promotion

These selected values answer “what benchmarks exist?” They **do not satisfy the requirement for independent user evidence**. Do not blend them with the results above.

### EmoTaG: reconstruction and emotion cues

[EmoTaG v1, Table 1](https://arxiv.org/html/2603.21332v1), neutral self-reconstruction after five seconds of adaptation:

| Model | PSNR ↑ | LPIPS ↓ | Sync-C ↑ |
|---|---:|---:|---:|
| InsTaG | 28.92 | 0.029 | 5.329 |
| EmoTaG | 30.02 | 0.019 | 6.212 |
| Real3DPortrait | 25.41 | 0.053 | 6.719 |

EmoTaG improves reconstruction here but does not lead every lip metric. The evaluation uses auxiliary pose/expression cues; a reconstruction result is not evidence for unconstrained audio-only emotional behavior. Eligible research contender; no independent winner designation.

### StreamingTalker: geometry, not final rendered realism

[StreamingTalker v2, Table 1](https://arxiv.org/html/2511.14223v2):

| Model | VOCASET LVE ↓ | BIWI LVE ↓ | BIWI mouth-opening difference ↓ |
|---|---:|---:|---:|
| DiffSpeaker | 3.1478 | 4.2829 | 8.4091 |
| StreamingTalker | 2.7206 | 4.2504 | 8.5208 |

Values retain the paper's scaling; its captions use unusual scaled-mm notation. **Do not convert these to physical millimeters without checking the evaluator.** LVE is lip vertex error. Geometry accuracy does not measure identity texture, teeth realism or conversational listening. A2F is absent from this table: no A2F superiority/inferiority conclusion follows.

### FlashHead: stream mode changes quality

[FlashHead v1, HDTF Table 3](https://arxiv.org/html/2602.07449v1):

| Configuration | FID ↓ | FVD ↓ | Sync-C ↑ |
|---|---:|---:|---:|
| Lite streaming | 11.37 | 126.52 | 4.21 |
| Lite non-streaming | 10.78 | 115.94 | 5.12 |
| Base streaming | 9.97 | 111.38 | 5.73 |

The streaming tradeoff is material. This paper version calls the larger variant **Base**; newer repository names use **Pro**. Do not silently equate checkpoints or attach a newer FPS claim to these quality values. FID/FVD compare distributions and depend on clip count/preprocessing; they are not per-user satisfaction scores.

### Interactive behavior: Ki Avatar Forcing

[Ki et al. v1, RealTalk Table 1](https://arxiv.org/html/2601.00664v1):

| Model | Expression reactivity error ↓ | FVD ↓ | CSIM ↑ |
|---|---:|---:|---:|
| INFP* | 0.035 | 159.000 | 0.867 |
| Avatar Forcing | 0.003 | 170.874 | 0.833 |

The star marks the paper's INFP reproduction. Better reactivity does not imply better identity or video distribution. The author's human study remains author-run evidence. This model is eligible for the listening-behavior comparison regardless of noncommercial status; the optimized released runtime must still be qualified.

## 5. Coverage gaps and consequences

| Component | Independent evidence established in this revision | Action |
|---|---|---|
| ASR | [HF Open ASR results](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-results) exposes Qwen3-ASR and other rows; direct dataset fetch failed in this tool session, and per-row execution provenance was not established | Do not convert search-result WER into a verified independent ranking. Keep streaming Zipformer resource baseline; run Qwen/Nemotron on the same noise/endpoint fixtures |
| LLM | No matched independent result established for exact local Qwen3.5-4B/Gemma quantization, conversational prompt and context budget | Keep provisional backend; evaluate useful first response, instruction correctness, spoken brevity and interruptions. General coding/math leaderboards cannot settle this choice |
| Motion and Gaussian representation | Paper comparisons available; no independent matched user score located for the shortlisted 2026 systems | A2F is an integration baseline; StreamingTalker/EmoTaG/LAM/AVTR-1/Avatar Forcing stay eligible. Run controlled comparisons before locking the visual model |
| Enrollment | No independent common-input identity/novel-view ranking established for LAM versus Instant Expressive | Reuse research-eligible LAM/VHAP/GaussianAvatars for validation where useful; do not build a clean enrollment stack merely to avoid NC before proving quality |
| Full conversation | No independent sub-second p95 plus visual-quality result established for this exact stack | Existing end-to-end measurement and concurrency gates remain mandatory |

Search scope included the independent evaluators linked above, primary research reproductions, official issue reports and searches for EmoTaG/StreamingTalker reproductions. “Not located” is a bounded search finding, not proof that no such evidence exists. Review new independent results at each model promotion.

## 6. Staged quality qualification: Q0

Apply **Q0** before expensive training, final renderer/enrollment commitment and release-quality claims. Qwen family selection is already made; begin implementation without a competing-TTS experiment. The P02 runtime/reference gate and P03 visual clips come first, with independent ratings before quality acceptance. Pending raters do not block unrelated functional P04 work. All paths below are proposed, not existing results.

1. Freeze `v2/evals/model-selection.yaml`: target profile, candidate checkpoint/runtime hashes, stream mode, voices, languages, precision, paired inputs, predeclared quality margins and timing/resource gates. Capture external evaluator date/version and conflicts of interest. Separate provider API and self-hosted runs.
2. TTS first: compare the selected Qwen streaming runtime with the official offline reference on matched voices/text. Evaluate X-Vector versus ICL only where needed. Defer cross-family comparisons unless measured Qwen deficiencies or a later user request justify them. Measure sequential configurations before co-loading selected Qwen with the actual stack.
3. Visual first: use identical PCM and consented identities. Compare the viable rig/motion baseline with FlashHead Lite; include LiveAvatar reference outputs on separately available hardware. Test A2F versus StreamingTalker on the same retargeted rig when assets/runtime are available. Test EmoTaG both with its declared cues and with the deployable cue source; never hide privileged input. Select at most two challengers per round.
4. Pilot: at least three identities/voices, 30 short/long, emotional/neutral, number/name-heavy clips, and at least 12 independent listeners/viewers, randomized/blinded order. Obtain at least five judgments per item; pilot size is a proposed starting point, not statistical proof. Use consented fixtures and preserve failed outputs. Follow with a powered held-out study based on the pilot variance before a release-quality claim.
5. Score separate axes: speech intelligibility, voice match, naturalness/prosody, lip contact/sync, teeth, identity, eyes/head behavior, temporal stability and listening appropriateness. Include 60-second outputs and live interruption sessions; a three-second showcase cannot pass. Keep WER/CER, speaker embedding, SyncNet/landmark errors and anatomy checks diagnostic alongside human judgments.
6. Compute paired preference with ties and clustered bootstrap uncertainty across identities/clips and raters; avoid treating every frame/vote as independent. Predeclare a promotion threshold: proposed lower 95% bound above 50% preference versus incumbent **and** no important quality-axis regression, subject to the validation protocol's noninferiority and latency gates. Treat a small/inconclusive pilot as unresolved, not a victory. For many candidates, validate the selected finalist on fresh items.
7. Publish `v2/evals/results/<run-id>/` manifests, protected outputs, anonymized ratings, failures, raw timing and confidence intervals; update ADR/model registry with `selected`, `conditional`, `rejected-for-profile` or `unresolved`. A higher-quality model that misses local latency remains eligible for a separately labeled high-quality profile.

These experiments are future implementation work. Qwen is selected for implementation, while runtime/quality acceptance remains unproven. No local model has passed Q0.
