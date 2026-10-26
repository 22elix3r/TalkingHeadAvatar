// CPU-only bridge fault fixture. No model or synthesis implementation.
#pragma once
#include <cstdio>
#include <functional>
#include <string>
#include <vector>
namespace qwen3_tts {
struct tts_params {
    bool play_streaming, streaming_generate, print_progress, print_timing;
    long seed;
    int max_audio_tokens;
    float max_audio_frames_per_text_token;
    bool paced_audio_delivery, async_streaming_decode;
    std::function<bool(const float*, int, int, bool)> audio_chunk_callback;
};
struct tts_result {
    bool success=true, reached_eos=true, hit_token_limit=false, hit_dynamic_token_limit=false;
    std::string error_msg;
};
struct tts_model_capabilities {
    std::string model_type="base";
    bool supports_voice_clone=true;
    int speaker_embedding_dim=4;
};
class Qwen3TTS {
public:
    bool load_models(const std::string&, const std::string&) { std::printf("upstream load stdout diagnostic\n"); return true; }
    std::string get_error() { return "fixture"; }
    tts_model_capabilities get_model_capabilities() { return {}; }
    tts_result synthesize_with_speaker_embedding(const std::string&, const std::vector<float>&, const tts_params& params) {
        std::printf("upstream request stdout diagnostic\n");
        const float samples[2]={.1f,.2f};
        params.audio_chunk_callback(samples,2,24000,true);
        std::printf("upstream completion stdout diagnostic\n");
        return {};
    }
};
inline bool load_speaker_embedding_file(const std::string&, std::vector<float>& values) {
    values.assign(4,.1f); return true;
}
}
