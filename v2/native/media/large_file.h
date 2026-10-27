#pragma once
#include "gpu_encoder.h"
#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <sys/stat.h>

namespace tha {
struct LargeGeometry {
  std::vector<float> vertices;
  std::vector<std::uint32_t> faces;
  std::vector<std::uint8_t> materials;
  MeshCamera camera;
};
inline LargeGeometry read_large_geometry(const char* path) {
  struct stat info{};
  if(lstat(path,&info)!=0 || !S_ISREG(info.st_mode) || info.st_size<128 || info.st_size>8*1024*1024)
    throw std::runtime_error("bounded regular full-size research geometry required");
  std::ifstream stream(path,std::ios::binary);std::array<char,8> magic{};std::array<std::uint32_t,2> shape{};
  std::array<double,14> camera{};
  stream.read(magic.data(),magic.size());stream.read(reinterpret_cast<char*>(shape.data()),sizeof(shape));stream.read(reinterpret_cast<char*>(camera.data()),sizeof(camera));
  if(!stream || std::memcmp(magic.data(),"THALRG1\0",8)!=0 || shape[0]<3 || shape[0]>100000 || shape[1]<1 || shape[1]>200000 ||
     info.st_size!=static_cast<off_t>(128+shape[0]*3*sizeof(float)+shape[1]*3*sizeof(std::uint32_t)+shape[1]))
    throw std::runtime_error("research geometry schema/extent mismatch; no truncation");
  LargeGeometry result;
  std::copy(camera.begin(),camera.begin()+9,result.camera.rotation.begin());
  std::copy(camera.begin()+9,camera.begin()+12,result.camera.translation_m.begin());
  result.camera.units_to_meters=camera[12];result.camera.half_extent_m=camera[13];
  result.vertices.resize(shape[0]*3);result.faces.resize(shape[1]*3);result.materials.resize(shape[1]);
  stream.read(reinterpret_cast<char*>(result.vertices.data()),result.vertices.size()*sizeof(float));
  stream.read(reinterpret_cast<char*>(result.faces.data()),result.faces.size()*sizeof(std::uint32_t));
  stream.read(reinterpret_cast<char*>(result.materials.data()),result.materials.size());
  if(!stream)throw std::runtime_error("truncated research geometry");
  return result;
}
}
