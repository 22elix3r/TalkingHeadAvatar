#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tha {
struct EncodedFrame {
  std::vector<std::uint8_t> data;
  std::int64_t pts_us=0;
  bool keyframe=false;
  float render_gpu_ms=0,convert_gpu_ms=0;
  double ready_wall_ms=0,encode_wall_ms=0;
  int slot=0;
};

// One thread owns the CUDA context, fixed surfaces, conversion streams and NVENC.
// The encoded payload owns its CPU bytes; no raw frame crosses this boundary.
class GpuEncoder {
public:
  GpuEncoder(const std::string& ptx,int width=512,int height=512,int fps=30,int bitrate=1200000);
  ~GpuEncoder();
  GpuEncoder(const GpuEncoder&)=delete;
  GpuEncoder& operator=(const GpuEncoder&)=delete;
  EncodedFrame encode(int pattern,std::int64_t pts_us,bool force_keyframe=false);
  void bitrate(int bits_per_second);
  void framerate(double frames_per_second);
  std::uint64_t coded_bytes() const;
private:
  struct State;
  std::unique_ptr<State> state_;
};
}
