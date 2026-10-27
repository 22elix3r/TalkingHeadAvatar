// Explicit GPU qualification, deliberately separate from CPU contract tests.
#include "gpu_encoder.h"
#include <iostream>
#include <stdexcept>
#include <thread>

template<class F> void rejected(F action) {
  bool caught=false;
  try {action();} catch(const std::exception&) {caught=true;}
  if(!caught) throw std::runtime_error("unsafe operation was accepted");
}

int main(int argc,char** argv) {
  try {
    if(argc!=2) throw std::invalid_argument("usage: tha_encoder_contract kernels.ptx");
    rejected([&]{tha::GpuEncoder invalid(argv[1],511);});
    // Repeated partial initialization exercises cleanup before a CUDA module
    // exists. Subsequent successful contexts must remain usable.
    for(int i=0;i<3;++i) rejected([]{tha::GpuEncoder missing("/nonexistent/tha-qualification.ptx");});
    for(int owner=0;owner<3;++owner) {
      tha::GpuEncoder encoder(argv[1]);
      const auto first=encoder.encode(0,1000000,true);
      if(!first.keyframe || first.data.empty()) throw std::runtime_error("first access unit is not independently decodable");
      rejected([&]{encoder.encode(1,1000000);});
      rejected([&]{encoder.encode(1,999999);});
      rejected([&]{encoder.bitrate(0);});
      rejected([&]{encoder.framerate(0);});
      bool foreign_rejected=false;
      std::thread other([&]{try {encoder.encode(1,1033333);} catch(const std::exception&) {foreign_rejected=true;}});
      other.join();
      if(!foreign_rejected) throw std::runtime_error("foreign thread acquired media ownership");
      for(int frame=1;frame<90;++frame) {
        auto encoded=encoder.encode(frame,1000000+frame*1000000LL/30,frame==17);
        if(encoded.slot!=frame%3 || encoded.data.empty() || (frame==17 && !encoded.keyframe))
          throw std::runtime_error("slot/access-unit/keyframe contract failed");
      }
    }
    std::cout<<"{\"gpu_contract\":\"passed\",\"owner_lifetimes\":3,\"failed_initializations\":3,\"frames\":270,\"physical_av_qualified\":false}\n";
    return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
