// Local pipe protocol, version 1. Linux/x86_64 float32 LE; never a network API.
#include "qwen3_tts.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
std::mutex output_mutex;
void le(uint64_t value, int bytes) {
    for (int i=0; i<bytes; ++i) std::cout.put(static_cast<char>((value >> (8*i)) & 255));
}
void frame(char kind, uint32_t rate, uint32_t count, uint64_t ns, const char* data=nullptr, size_t bytes=0) {
    std::lock_guard<std::mutex> lock(output_mutex);
    std::cout.put(kind); le(rate, 4); le(count, 4); le(ns, 8);
    if (bytes) std::cout.write(data, static_cast<std::streamsize>(bytes));
    std::cout.flush();
}
void error(const std::string& message) { frame('E', 0, message.size(), 0, message.data(), message.size()); }
uint64_t elapsed(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();
}
bool read_size(uint32_t& value) {
    value=0;
    for (int i=0; i<4; ++i) {
        int byte=std::cin.get();
        if (byte==EOF) return false;
        value |= static_cast<uint32_t>(static_cast<unsigned char>(byte)) << (8*i);
    }
    return true;
}
}

int main(int argc, char** argv) {
    if (argc != 3) { error("expected model directory and approved speaker embedding"); return 2; }
    qwen3_tts::Qwen3TTS engine;
    const auto load_start=Clock::now();
    if (!engine.load_models(argv[1], "qwen3-tts-1.7b-base-f16.gguf")) { error(engine.get_error()); return 1; }
    auto caps=engine.get_model_capabilities();
    if (caps.model_type != "base" || !caps.supports_voice_clone) { error("selected Base clone route unavailable"); return 1; }
    std::vector<float> embedding;
    if (!qwen3_tts::load_speaker_embedding_file(argv[2], embedding) ||
        embedding.size() != static_cast<size_t>(caps.speaker_embedding_dim)) {
        error("speaker embedding missing or incompatible; no fallback voice permitted"); return 1;
    }
    frame('R', 24000, 0, elapsed(load_start));
    std::mutex mutex;
    std::condition_variable changed;
    std::string pending;
    bool has_pending=false, closing=false, active=false;
    std::atomic<bool> cancelled{false};
    std::thread control([&] {
        while (true) {
            int command=std::cin.get(); uint32_t size=0;
            if (command==EOF || !read_size(size) || size>65536) break;
            std::string payload(size, '\0');
            if (size && !std::cin.read(payload.data(), size)) break;
            std::lock_guard<std::mutex> lock(mutex);
            if (command=='C') { cancelled.store(true); changed.notify_all(); }
            else if (command=='G' && size && !active && !has_pending) {
                cancelled.store(false); pending=std::move(payload); has_pending=true; changed.notify_all();
            } else if (command=='X') break;
            else { error("invalid/overlapping request"); }
        }
        { std::lock_guard<std::mutex> lock(mutex); closing=true; cancelled.store(true); }
        changed.notify_all();
    });
    while (true) {
        std::string text;
        { std::unique_lock<std::mutex> lock(mutex);
          changed.wait(lock, [&] { return has_pending || closing; });
          if (closing) break;
          text=std::move(pending); has_pending=false; active=true; }
        const auto started=Clock::now();
        qwen3_tts::tts_params params;
        params.play_streaming=false; params.streaming_generate=true;
        params.print_progress=false; params.print_timing=false;
        params.seed=17; params.max_audio_tokens=1024;
        params.max_audio_frames_per_text_token=0; // A safety cutoff is reported as failure, never EOF.
        params.paced_audio_delivery=false; params.async_streaming_decode=true;
        params.audio_chunk_callback=[&](const float* samples, int32_t count, int32_t rate, bool) {
            if (cancelled.load()) return false;
            if (count<0 || count>48000 || rate!=24000) return false;
            if (count) frame('A', rate, count, elapsed(started), reinterpret_cast<const char*>(samples), count*sizeof(float));
            return !cancelled.load() && std::cout.good();
        };
        auto result=engine.synthesize_with_speaker_embedding(text, embedding, params);
        { std::lock_guard<std::mutex> lock(mutex); active=false; }
        if (cancelled.load()) frame('C', 24000, 0, elapsed(started));
        else if (!result.success || !result.reached_eos || result.hit_token_limit || result.hit_dynamic_token_limit)
            error("generation failed or ended at safety limit; native EOF not established: "+result.error_msg);
        else frame('D', 24000, 0, elapsed(started));
    }
    if (control.joinable()) control.join();
    return 0;
}
