#pragma once
// C ABI: LiveKit uses the current C++ ABI; the pinned Claire SDK uses ABI=0.
// One thread owns a handle and its SDK/context/encoder for one generation.
#include <stddef.h>
#include <stdint.h>
#if defined(__cplusplus)
extern "C" {
#endif
typedef struct tha_claire tha_claire;
typedef struct {
  const uint8_t* data; size_t size;
  int64_t pts_us; uint64_t source_sample_48k, neural_sample_16k;
  int keyframe;
} tha_claire_packet;
typedef struct {
  uint64_t offered_samples, neural_frames, rendered_frames, discarded_video_frames;
  uint64_t first_result_after_samples, retained_audio_peak_samples, queue_peak_frames;
  uint64_t queued_frames, eof;
} tha_claire_metrics;
// Paths must already be hash/provenance checked by the operator. The caller
// must hold the project device lease. No download or engine build occurs here.
int tha_claire_open(const char* ptx, const char* scene, const char* model,
                    tha_claire** result, char* error, size_t error_size);
// Contiguous finite 16 kHz float PCM, <=20 ms per push. EOF closes actual input;
// SDK context padding never becomes playable audio. Overflow fails the epoch.
int tha_claire_push(tha_claire*, const float*, size_t count, uint64_t start_sample,
                    int eof, char* error, size_t error_size);
// Audio-relative 48 kHz/30 Hz grid. Returns 1 for a packet, 0 when that motion
// is not available, -1 on failure. Future geometry is never presented early.
// Packet storage belongs to the handle until the next operation. Later targets
// may discard stale VIDEO only. No interpolation crosses handle/epoch lifetime.
int tha_claire_render(tha_claire*, uint64_t source_sample_48k, int64_t pts_us,
                      int keyframe, tha_claire_packet*, char* error, size_t error_size);
int tha_claire_get_metrics(tha_claire*, tha_claire_metrics*, char* error, size_t error_size);
int tha_claire_close(tha_claire*, char* error, size_t error_size);
#if defined(__cplusplus)
}
#endif
