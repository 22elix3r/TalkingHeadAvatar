// Local pipe protocol, version 1. Linux/x86_64 float32 LE; never a network API.
#include "qwen3_tts.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <cerrno>
#include <csignal>
#include <unistd.h>

namespace {
using Clock = std::chrono::steady_clock;
std::mutex output_mutex;
std::atomic<uint64_t> request_id{0};
int protocol_fd=-1;
std::atomic<bool> protocol_ok{true};
void le(std::string& header, uint64_t value, int bytes) {
    for (int i=0; i<bytes; ++i) header.push_back(static_cast<char>((value >> (8*i)) & 255));
}
bool write_all(const char* data, size_t bytes) {
    while (bytes) {
        const auto written=::write(protocol_fd, data, bytes);
        if (written<0 && errno==EINTR) continue;
        if (written<=0) { protocol_ok.store(false); return false; }
        data+=written; bytes-=written;
    }
    return true;
}
void frame(char kind, uint32_t rate, uint32_t count, uint64_t ns, const char* data=nullptr, size_t bytes=0) {
    std::lock_guard<std::mutex> lock(output_mutex);
    std::string header(1, kind); le(header, rate, 4); le(header, count, 4); le(header, ns, 8);
    le(header, request_id.load(), 8);
    if (write_all(header.data(), header.size()) && bytes) write_all(data, bytes);
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
    // Keep a dedicated framed-protocol descriptor. All library/CUDA stdout is
    // redirected to diagnostics, so an upstream printf cannot corrupt PCM.
    protocol_fd=::dup(STDOUT_FILENO);
    if (protocol_fd<0 || ::dup2(STDERR_FILENO, STDOUT_FILENO)<0) return 2;
    std::signal(SIGPIPE, SIG_IGN);
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
            else if (command=='G' && size>8 && !active && !has_pending) {
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
        uint64_t serial=0;
        for (int i=0; i<8; ++i) serial |= static_cast<uint64_t>(static_cast<unsigned char>(text[i])) << (8*i);
        request_id.store(serial); text.erase(0, 8);
        const auto started=Clock::now();
        qwen3_tts::tts_params params;
        params.play_streaming=false; params.streaming_generate=true;
        params.print_progress=false; params.print_timing=false;
        params.seed=17; params.max_audio_tokens=1024;
        params.max_audio_frames_per_text_token=0; // A safety cutoff is reported as failure, never EOF.
        params.paced_audio_delivery=false;
        const char* async_decode=std::getenv("THA_QWEN_ASYNC_DECODE");
        params.async_streaming_decode=async_decode && std::string(async_decode)=="1";
        params.audio_chunk_callback=[&](const float* samples, int32_t count, int32_t rate, bool) {
            if (cancelled.load()) return false;
            if (count<0 || count>48000 || rate!=24000) return false;
            if (count) frame('A', rate, count, elapsed(started), reinterpret_cast<const char*>(samples), count*sizeof(float));
            return !cancelled.load() && protocol_ok.load();
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
