// Explicit recorded-neural reference: host geometry uploads and two RGBA readbacks.
#include "gpu_encoder.h"
#include "large_file.h"
#include <array>
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace {
void require(bool good,const char* message) {if(!good)throw std::runtime_error(message);}
void write_all(int output,const void* data,std::size_t count) {
  const char* bytes=static_cast<const char*>(data);
  while(count) {const auto n=write(output,bytes,count);require(n>0,"research probe file write failed");bytes+=n;count-=n;}
}
}
int main(int argc,char** argv) {
  int output=-1;
  try {
    require(argc==6,"usage: tha_large_mesh_probe kernels.ptx research-mesh.bin neural-geometry.bin new-video.h264 snapshot-prefix");
    const char* inherited=std::getenv("THA_DEVICE_LEASE_FD");
    require(inherited && fcntl(std::stoi(inherited),F_GETFD)>=0,"inherited admitted device lease required");
    const auto scene=tha::read_large_geometry(argv[2]);const auto vertices=scene.vertices.size()/3;
    struct stat info{};require(lstat(argv[3],&info)==0 && S_ISREG(info.st_mode) && info.st_size>=24 && info.st_size<=1536LL*1024*1024,"bounded recorded neural geometry required");
    std::ifstream source(argv[3],std::ios::binary);std::array<char,8> magic{};std::array<std::uint32_t,4> counts{};
    source.read(magic.data(),magic.size());source.read(reinterpret_cast<char*>(counts.data()),sizeof(counts));
    require(source && std::memcmp(magic.data(),"THACLR1\0",8)==0 && counts[0]==61520*3 && counts[1]==5602*3 && counts[2]==16 && counts[3]==6 && vertices==61520+5602,"source neural/model topology identity mismatch");
    const std::size_t values=counts[0]+counts[1]+counts[2]+counts[3],record=16+values*sizeof(float);
    require((info.st_size-24)%record==0,"incomplete source neural frame");const auto total=(info.st_size-24)/record;
    require(total>0 && total<=1800,"bounded source frame count required");
    output=open(argv[4],O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);require(output>=0,"new private coded output required");
    tha::GpuEncoder encoder(argv[1]);encoder.large_mesh(vertices,scene.faces,scene.materials,scene.camera);
    std::vector<float> frame(values);double update_total=0,render_total=0;unsigned int peak_tile=0,peak_references=0;int emitted=0;
    const auto began=std::chrono::steady_clock::now();
    for(std::int64_t index=0;index<static_cast<std::int64_t>(total);++index) {
      std::array<std::int64_t,2> stamps{};source.read(reinterpret_cast<char*>(stamps.data()),sizeof(stamps));source.read(reinterpret_cast<char*>(frame.data()),frame.size()*sizeof(float));
      require(bool(source) && stamps[0]==index*16000/60 && stamps[1]==(index+1)*16000/60,"source sample/frame timebase mismatch");
      if(index%2)continue;
      std::vector<float> world(frame.begin(),frame.begin()+counts[0]+counts[1]);
      tha::MeshUpdate update;
      try {update=encoder.update_mesh_host(world);}
      catch(...) {
        const auto projected=encoder.qualification_projected_vertices();
        const auto filename=std::string(argv[5])+"-failed-projection.bin";
        const int diagnostic=open(filename.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
        if(diagnostic>=0) {write_all(diagnostic,projected.data(),projected.size()*sizeof(double));close(diagnostic);}
        throw;
      }
      peak_tile=std::max(peak_tile,update.maximum_tile_faces);
      peak_references=std::max(peak_references,update.tile_references);update_total+=update.gpu_ms;
      const auto coded=encoder.encode(0,1000000+stamps[0]*1000000/16000,emitted==0 || emitted==31);render_total+=coded.render_gpu_ms;
      require((emitted!=0 && emitted!=31) || coded.keyframe,"required independent keyframe missing");
      write_all(output,coded.data.data(),coded.data.size());
      if(index==0 || index==static_cast<std::int64_t>(total/2)/2*2) {
        const auto pixels=encoder.qualification_readback_rgba();
        const auto filename=std::string(argv[5])+"-"+std::to_string(index)+".rgba";
        const int snapshot=open(filename.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);require(snapshot>=0,"new private snapshot required");
        write_all(snapshot,pixels.data(),pixels.size());close(snapshot);
      }
      ++emitted;
    }
    require(fsync(output)==0,"coded output persistence failed");close(output);output=-1;
    std::cout<<"{\"source_frames\":"<<total<<",\"encoded_frames\":"<<emitted<<",\"vertices\":"<<vertices<<",\"triangles\":"<<scene.faces.size()/3
      <<",\"geometry_update_gpu_total_ms\":"<<update_total<<",\"render_gpu_total_ms\":"<<render_total<<",\"maximum_tile_faces\":"<<peak_tile
      <<",\"tile_edge\":8,\"tile_face_limit\":8192,\"maximum_tile_references\":"<<peak_references
      <<",\"tile_reference_capacity\":4194304,\"geometry_host_uploads_per_frame\":1,\"status_readback_bytes_per_frame\":20,\"qualification_rgba_readbacks\":2,\"elapsed_ms\":"
      <<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count()<<"}\n";
    return 0;
  } catch(const std::exception& error) {if(output>=0)close(output);std::cerr<<"research mesh probe: "<<error.what()<<'\n';return 1;}
}
