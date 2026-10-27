---
name: avatar-research-evidence
description: Verify avatar and speech model evidence, releases, licenses and benchmark boundaries before selecting or updating V2 components.
---

# Avatar research verification

Read the relevant candidate in [RESEARCH_EVIDENCE.md](../../../docs/v2/RESEARCH_EVIDENCE.md) and its ADR in the [main plan](../../../V2_RESEARCH_AND_BUILD_PLAN.md).

For model selection, start with [independent quality evidence/Q0](../../../docs/v2/MODEL_QUALITY_BENCHMARKS.md). This is research-first: noncommercial models are eligible for the primary stack. Record terms now; defer commercial clearance to a later distribution profile. Actual research access conditions and consent still apply.

Honor the accepted Qwen3-TTS implementation decision. Do not use missing comparative evidence to reopen model selection or block P00–P04 functional work. Qualify the selected model/runtime honestly; defer cross-family comparisons unless a measured failure or new user request justifies them. [Phase guide](../../../docs/v2/IMPLEMENTATION_PHASES.md).

- Resolve exact model identity; similar paper names and product names are not interchangeable. Record paper date, actual code/weight release, inspected revision and access date separately.
- Inspect the implementation/configuration behind streaming claims: input accumulation, lookahead, output yield point, decoder/compositor and demo transport. Distinguish incremental generation from chunked delivery after batch inference.
- Inspect code, weights, base models, face trackers, rig assets and datasets separately for rights. Missing LICENSE is unresolved. New upstream terms do not relicense old derivatives.
- Normalize only compatible boundaries: media duration, resolution, full crop/composition/encode path, hardware, precision, batch and cache state. FPS and first-useful-output are different measurements.
- Label local observation, source inspection, author report, practitioner report and unresolved claims. Search actual issues/reproductions and newer alternatives; retain contradictory evidence with conditions.
- Establish who performed each evaluation. Author-run human studies remain author evidence; a leaderboard row may be submitted by a provider. Prefer independent blinded user ratings and inspectable third-party runs. Stars, downloads, complaint counts and marketing comparisons are not quality scores.
- Keep raw scores with metric direction, evaluator/model/runtime revision, inputs, stream mode, sample/rater counts, uncertainty and access date. Separate endpoint from downloadable weights, provider-voice from controlled-voice arenas, and paired reconstruction from unconstrained generation. An inaccessible table or unverified row stays unresolved.
- Change selection only for evidence relevant to the target profile. Preserve higher-quality large models as separate references; never combine their quality with a smaller model's timing. If independent evidence is missing, schedule the Q0 blinded comparison and keep the model provisional rather than substituting author claims.
- Write direct primary links and decision consequences into the ledger. Do not promote a model without code/weight availability, resource feasibility and the applicable quality/timing gate. Do not download large weights merely to establish that a release exists.

For deployment/framework changes also read [DEPLOYMENT_RESEARCH.md](../../../docs/v2/DEPLOYMENT_RESEARCH.md). Verify the tagged release API, transitive binary/runtime compatibility and separate weight licenses; LAM and LiveKit illustrate why root README labels and helper names are insufficient. Record live-document access separately from a historical cutoff.

Return the evidence update, remaining uncertainty and ADR/gate impact. Preserve previous measurements and their original conditions.
