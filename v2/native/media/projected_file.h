#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>

namespace tha {
struct ProjectedGeometry {
  std::uint32_t width=0,height=0;
  std::vector<double> vertices;
  std::vector<std::uint32_t> faces;
};
inline ProjectedGeometry read_projected(const char* path) {
  struct stat info{};
  if(lstat(path,&info)!=0 || !S_ISREG(info.st_mode) || info.st_size<24 || info.st_size>32768)
    throw std::runtime_error("bounded regular projected mesh required");
  std::ifstream input(path,std::ios::binary);std::array<char,8> magic{};std::array<std::uint32_t,4> shape{};
  input.read(magic.data(),magic.size());input.read(reinterpret_cast<char*>(shape.data()),sizeof(shape));
  if(!input || std::memcmp(magic.data(),"THAMESH1",8)!=0 || shape[0]!=512 || shape[1]!=512 || shape[2]<3 || shape[2]>512 || shape[3]<1 || shape[3]>1024 ||
    info.st_size!=static_cast<off_t>(24+shape[2]*3*sizeof(double)+shape[3]*3*sizeof(std::uint32_t)))
    throw std::runtime_error("native serving geometry must match the qualified 512 profile");
  ProjectedGeometry result;result.width=shape[0];result.height=shape[1];result.vertices.resize(shape[2]*3);result.faces.resize(shape[3]*3);
  input.read(reinterpret_cast<char*>(result.vertices.data()),result.vertices.size()*sizeof(double));
  input.read(reinterpret_cast<char*>(result.faces.data()),result.faces.size()*sizeof(std::uint32_t));
  if(!input) throw std::runtime_error("truncated projected geometry");
  return result;
}
}
