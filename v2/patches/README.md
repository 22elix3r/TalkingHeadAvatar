# Pinned SDK patches

`livekit-native-audio-fail-closed.patch` applies to LiveKit Rust SDK revision
`bae4df2f3d6c8f0f6ce81a7370c445779f1948d2`. Its SHA-256 is bound by
`server/inbound_config.py`. The original cached checkout remains unchanged;
build and test the patch in a separate local worktree.

The patch closes a bounded native audio stream on overflow, clears queued
audio and marks the EOF event with optional boolean field 1,
`native_queue_overflow`. This avoids continuing after silently evicting speech.
The qualified application path uses unfiltered, explicit microphone tracks.
Other SDK consumers and audio-filter paths are not qualified by this patch.

The modified upstream code is Apache-2.0, with its upstream notices preserved.
The built FFI also includes separately licensed WebRTC/libyuv and other
dependencies; this patch does not relicense them or authorize binary
redistribution. No compiled libraries are checked in.

See [the build and qualification runbook](../docs/p06-inbound-audio.md).
