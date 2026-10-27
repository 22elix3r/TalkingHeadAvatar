---
name: avatar-av-sync
description: Implement or diagnose avatar audio/video clocks, sample accounting, EOF, drift, queue policies and interruption cancellation.
---

# Avatar synchronization

Read main-plan sections 6–9 in [the specification](../../../V2_RESEARCH_AND_BUILD_PLAN.md) and the sample/EOF/epoch fixtures in [the protocol](../../../docs/v2/VALIDATION_PROTOCOL.md).

Trace one generation from native TTS samples through stateful resampling, motion context, render PTS, transport and actual audio playout. Identify clock domains and uncertainty before subtracting timestamps. Audio presentation is the master; generation completion and packet arrival are not presentation times.

Preserve exact sample ranges and resampler history, with explicit EOF and discontinuity. Padding needed by a model is not playable audio. Queue-empty is not EOF. Motion lookahead and interpolation delay must be included in startup.

Cancellation advances an epoch, rejects late provider callbacks and flushes queued audio/video; account for already committed device buffers. Keep RTP clocks continuous across epochs and prohibit interpolation across reset boundaries. Conversation history retains only presented speech.

Exercise short tails, irregular chunks/rates, delayed packets, drift, reconnect and cancel at each stage. Validate with actual playback or loopback; matching server timestamps alone cannot prove lip sync. Under overload drop stale video, apply backpressure or explicitly fail; never silently delete words.

For browser/media integration read the cancel-resume and native transport contracts in [E2E_IMPLEMENTATION_SPEC.md](../../../docs/v2/E2E_IMPLEMENTATION_SPEC.md). Sender queue estimates/AVSynchronizer logs are not receiver presentation. Clearing a sender queue does not flush a browser decoder. Keep local output muted until a proven safe boundary; qualify track replacement if needed and measure its resume delay. Video frame metadata does not supply exact audio-sample acknowledgment.

Report the faulty boundary, invariant restored and tests/measurement uncertainty. Keep clock logic independent of the selected neural model.
