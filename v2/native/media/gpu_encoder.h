#pragma once
#include <cstdint>
#include <array>
#include <cuda.h>
#include <memory>
#include <string>
#include <vector>

namespace tha {
enum class ContextPolicy { Private, BorrowCurrent };
struct MeshCamera {
  std::array<double,9> rotation;
  std::array<double,3> translation_m;
  double units_to_meters;
  double half_extent_m;
};
struct MeshUpdate {
  float gpu_ms=0;
  unsigned int maximum_tile_faces=0;
  unsigned int tile_references=0;
};
struct DeviceMeshPart {
  CUdeviceptr xyz;
  std::size_t vertex_count;
  CUstream producer;
};
struct EncodedFrame {
  std::vector<std::uint8_t> data;
  std::int64_t pts_us=0;
  bool keyframe=false;
  float render_gpu_ms=0,convert_gpu_ms=0;
  double ready_wall_ms=0,encode_wall_ms=0;
  int slot=0;
};

// One thread owns the CUDA context, fixed surfaces, conversion streams and NVENC.
// Publishing owns CPU encoded bytes and retains raw frames on the device.
// The separately named qualification readback is an explicit exception.
class GpuEncoder {
public:
  GpuEncoder(const std::string& ptx,int width=512,int height=512,int fps=30,int bitrate=1200000,
             ContextPolicy context_policy=ContextPolicy::Private);
  ~GpuEncoder();
  GpuEncoder(const GpuEncoder&)=delete;
  GpuEncoder& operator=(const GpuEncoder&)=delete;
  EncodedFrame encode(int pattern,std::int64_t pts_us,bool force_keyframe=false);
  // Immutable projected neutral geometry. Canonical authorization/rig checks
  // remain the server's responsibility; this boundary validates every index.
  void mesh(const std::vector<double>& xyz,const std::vector<std::uint32_t>& triangles);
  // Separate full-size research profile: immutable topology, explicit camera,
  // bounded tile lists and exact dynamic vertex count. No small-profile widening.
  void large_mesh(std::size_t vertex_count,const std::vector<std::uint32_t>& triangles,
                  const std::vector<std::uint8_t>& materials,const MeshCamera& camera);
  // Data is copied into owned geometry before return. The producer's CUDA
  // stream/event dependency protects SDK buffers until this copy completes.
  MeshUpdate update_mesh_device(CUdeviceptr xyz,std::size_t vertex_count,CUstream producer);
  MeshUpdate update_mesh_device(const std::vector<DeviceMeshPart>& parts);
  // Explicit recorded-geometry reference; this uploads vertex data each update.
  MeshUpdate update_mesh_host(const std::vector<float>& xyz);
  // Explicit qualification-only readback, never used by the publishing path.
  std::vector<std::uint8_t> qualification_readback_rgba();
  std::vector<double> qualification_projected_vertices();
  void bitrate(int bits_per_second);
  void framerate(double frames_per_second);
  std::uint64_t coded_bytes() const;
private:
  struct State;
  std::unique_ptr<State> state_;
};
}
