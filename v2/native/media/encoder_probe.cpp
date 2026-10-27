#include "gpu_encoder.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>

int main(int argc,char** argv) {
  try {
    if(argc!=4) throw std::invalid_argument("usage: tha_encoder_probe kernels.ptx frames output.h264");
    const int frames=std::stoi(argv[2]);
    if(frames<1 || frames>18000) throw std::invalid_argument("frame count outside bounded probe");
    tha::GpuEncoder encoder(argv[1]);
    std::ofstream file(argv[3],std::ios::binary);
    if(!file) throw std::runtime_error("cannot open private synthetic output");
    double render=0,convert=0,encode=0,ready=0,max_ready=0;
    int keys=0;
    auto start=std::chrono::steady_clock::now();
    for(int i=0;i<frames;++i) {
      auto frame=encoder.encode(i,1000000+static_cast<std::int64_t>(i)*1000000/30,i==17);
      file.write(reinterpret_cast<const char*>(frame.data.data()),frame.data.size());
      render+=frame.render_gpu_ms;convert+=frame.convert_gpu_ms;encode+=frame.encode_wall_ms;
      ready+=frame.ready_wall_ms;max_ready=std::max(max_ready,frame.ready_wall_ms);keys+=frame.keyframe;
    }
    file.close();
    std::cout<<"{\"frames\":"<<frames<<",\"keyframes\":"<<keys<<",\"coded_bytes\":"<<encoder.coded_bytes()
      <<",\"render_gpu_total_ms\":"<<render<<",\"convert_gpu_total_ms\":"<<convert
      <<",\"ready_wall_total_ms\":"<<ready<<",\"ready_wall_max_ms\":"<<max_ready
      <<",\"encode_wall_total_ms\":"<<encode<<",\"wall_ms\":"
      <<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()
      <<",\"gpu_surface_slots\":3,\"synthetic_pattern\":true,\"raw_readback_trace_qualified\":false}\n";
    return 0;
  } catch(const std::exception& error) {std::cerr<<"native encoder: "<<error.what()<<'\n';return 1;}
}
