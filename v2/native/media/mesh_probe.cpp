// Explicit one-frame raw readback for CPU/GPU parity, outside publishing traces.
#include "gpu_encoder.h"
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace {
void require(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
void write_private(const char* path,const std::uint8_t* bytes,std::size_t length) {
  const int file=open(path,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
  require(file>=0,"qualification output must be new/private");
  std::size_t offset=0;
  while(offset<length) {auto count=write(file,bytes+offset,length-offset);if(count<=0) {close(file);throw std::runtime_error("qualification write failed");}offset+=count;}
  close(file);
}
}

int main(int argc,char** argv) {
  try {
    require(argc==5 || (argc==6 && std::strcmp(argv[5],"--no-readback")==0),"usage: tha_mesh_probe kernels.ptx projected-mesh.bin rgba-output h264-output [--no-readback]");
    const bool readback=argc==5;
    struct stat info{};
    require(lstat(argv[2],&info)==0 && S_ISREG(info.st_mode) && info.st_size<=32768,"bounded regular projected mesh required");
    std::ifstream input(argv[2],std::ios::binary);
    std::array<char,8> magic{};std::array<std::uint32_t,4> shape{};
    input.read(magic.data(),magic.size());input.read(reinterpret_cast<char*>(shape.data()),sizeof(shape));
    require(bool(input) && std::memcmp(magic.data(),"THAMESH1",8)==0,"mesh protocol magic mismatch");
    const auto width=shape[0],height=shape[1],vertices=shape[2],faces=shape[3];
    require(width>=16 && width<=512 && !(width%2) && height>=16 && height<=512 && !(height%2) &&
      vertices>=3 && vertices<=512 && faces>=1 && faces<=1024,"mesh shape is outside qualification bounds");
    require(info.st_size==24+vertices*3*sizeof(double)+faces*3*sizeof(std::uint32_t),"mesh byte count mismatch");
    std::vector<double> xyz(vertices*3);std::vector<std::uint32_t> indices(faces*3);
    input.read(reinterpret_cast<char*>(xyz.data()),xyz.size()*sizeof(double));
    input.read(reinterpret_cast<char*>(indices.data()),indices.size()*sizeof(std::uint32_t));require(bool(input),"truncated projected mesh");
    tha::GpuEncoder encoder(argv[1],width,height);encoder.mesh(xyz,indices);
    std::vector<std::uint8_t> coded;double render_ms=0,convert_ms=0;
    for(int frame=0;frame<60;++frame) {
      auto packet=encoder.encode(frame,1000000+frame*1000000LL/30,frame==17);
      coded.insert(coded.end(),packet.data.begin(),packet.data.end());render_ms+=packet.render_gpu_ms;convert_ms+=packet.convert_gpu_ms;
      if(frame==0 && readback) {auto pixels=encoder.qualification_readback_rgba();write_private(argv[3],pixels.data(),pixels.size());}
    }
    write_private(argv[4],coded.data(),coded.size());
    std::cout<<"{\"mesh_vertices\":"<<vertices<<",\"mesh_triangles\":"<<faces<<",\"frames\":60,\"render_gpu_total_ms\":"<<render_ms
      <<",\"convert_gpu_total_ms\":"<<convert_ms<<",\"qualification_raw_readback_bytes\":"<<(readback?width*height*4:0)
      <<",\"speaking_avatar_qualified\":false}"<<std::endl;
    return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
