// Explicit GPU safety/lifetime cases, separate from model-free CPU tests.
#include "gpu_encoder.h"
#include <cmath>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

template<class F>void rejected(F action) {
  bool caught=false;try {action();}catch(const std::exception&){caught=true;}
  if(!caught)throw std::runtime_error("unsafe research operation accepted");
}
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
void checked(CUresult result) {require(result==CUDA_SUCCESS,"GPU contract CUDA operation failed");}
int main(int argc,char** argv) {
  try {
    require(argc==2,"usage: tha_large_mesh_contract kernels.ptx");
    const auto* inherited=std::getenv("THA_DEVICE_LEASE_FD");require(inherited && fcntl(std::stoi(inherited),F_GETFD)>=0,"inherited device admission required");
    const tha::MeshCamera camera{{1,0,0,0,1,0,0,0,1},{0,0,0},1,1};
    const std::vector<float> xyz{-.5f,-.5f,1.f,.5f,-.5f,1.f,0.f,.5f,1.f};
    int successful_frames=0,owners=0;
    {
      tha::GpuEncoder encoder(argv[1]);++owners;
      rejected([&]{encoder.large_mesh(100001,{0,1,2},{0},camera);});
      rejected([&]{encoder.large_mesh(3,{0,1,3},{0},camera);});
      rejected([&]{encoder.large_mesh(3,{0,0,2},{0},camera);});
      rejected([&]{encoder.large_mesh(3,{0,1,2},{2},camera);});
      auto reflection=camera;reflection.rotation[8]=-1;
      rejected([&]{encoder.large_mesh(3,{0,1,2},{0},reflection);});
      auto invalid_units=camera;invalid_units.units_to_meters=0;
      rejected([&]{encoder.large_mesh(3,{0,1,2},{0},invalid_units);});
      // Equal-depth overlays must select the first face, independent of atomic
      // tile insertion order. Materials deliberately differ.
      encoder.large_mesh(3,{0,1,2,0,1,2},{0,1},camera);
      rejected([&]{encoder.encode(0,1000000);});
      rejected([&]{encoder.large_mesh(3,{0,1,2},{0},camera);});
      rejected([&]{encoder.update_mesh_host({0,0,1});});
      auto invalid=xyz;invalid[0]=std::numeric_limits<float>::quiet_NaN();
      rejected([&]{encoder.update_mesh_host(invalid);});
      rejected([&]{encoder.encode(0,1000000);});
      invalid=xyz;invalid[2]=-1;
      rejected([&]{encoder.update_mesh_host(invalid);});
      for(int frame=0;frame<8;++frame) {
        const auto update=encoder.update_mesh_host(xyz);
        require(update.maximum_tile_faces==2 && update.tile_references>0,"compact tile accounting failed");
        const auto packet=encoder.encode(0,1000000+frame*33333,frame==0);require(!packet.data.empty(),"missing reference access unit");++successful_frames;
        const auto pixels=encoder.qualification_readback_rgba();const int center=(256*512+256)*4;
        require(pixels[0]==18 && pixels[1]==24 && pixels[2]==34 && pixels[center]>pixels[center+1],"reference coverage/material selection failed");
        const double light=.25+.75/std::sqrt(1.185);
        require(pixels[center]==static_cast<unsigned char>(std::floor(188*light+.5)) && pixels[center+1]==static_cast<unsigned char>(std::floor(151*light+.5)),"equal-depth first-face identity changed");
      }
      bool foreign_thread_rejected=false;
      std::thread other([&]{try {encoder.update_mesh_host(xyz);}catch(const std::exception&){foreign_thread_rejected=true;}});other.join();
      require(foreign_thread_rejected,"foreign thread acquired geometry ownership");
      // Two independent producer allocations/stream parts exercise the same
      // exact concatenation contract used by skin and tongue SDK buffers.
      CUdeviceptr first=0,second=0;CUstream producer=nullptr;
      checked(cuMemAlloc(&first,6*sizeof(float)));checked(cuMemAlloc(&second,3*sizeof(float)));checked(cuStreamCreate(&producer,CU_STREAM_NON_BLOCKING));
      checked(cuMemcpyHtoD(first,xyz.data(),6*sizeof(float)));checked(cuMemcpyHtoD(second,xyz.data()+6,3*sizeof(float)));checked(cuCtxSynchronize());
      rejected([&]{encoder.update_mesh_device(first,4,producer);});
      const auto device_update=encoder.update_mesh_device(std::vector<tha::DeviceMeshPart>{{first,2,producer},{second,1,producer}});
      require(device_update.maximum_tile_faces==2,"producer event/part mapping failed");
      encoder.encode(0,1400000,true);++successful_frames;
      checked(cuStreamDestroy(producer));checked(cuMemFree(first));checked(cuMemFree(second));
      CUcontext foreign=nullptr;checked(cuCtxCreate(&foreign,0,0));CUdeviceptr foreign_memory=0;checked(cuMemAlloc(&foreign_memory,xyz.size()*sizeof(float)));
      CUcontext popped=nullptr;checked(cuCtxPopCurrent(&popped));
      rejected([&]{encoder.update_mesh_device(foreign_memory,3,nullptr);});
      checked(cuCtxPushCurrent(foreign));checked(cuMemFree(foreign_memory));checked(cuCtxPopCurrent(&popped));checked(cuCtxDestroy(foreign));
    }
    for(bool global:{false,true}) {
      tha::GpuEncoder encoder(argv[1]);++owners;
      const int count=global?5000:8193;std::vector<std::uint32_t> faces;faces.reserve(count*3);
      for(int face=0;face<count;++face)faces.insert(faces.end(),{0,1,2});
      encoder.large_mesh(3,faces,std::vector<std::uint8_t>(count,0),camera);
      const std::vector<float> points=global?std::vector<float>{-1,-1,1,1,-1,1,0,1,1}:
        std::vector<float>{-0.984f,-0.984f,1,-0.980f,-0.984f,1,-0.984f,-0.980f,1};
      rejected([&]{encoder.update_mesh_host(points);});
      rejected([&]{encoder.encode(0,1000000);});
    }
    CUdevice device=0;checked(cuDeviceGet(&device,0));CUcontext primary=nullptr;checked(cuDevicePrimaryCtxRetain(&primary,device));checked(cuCtxPushCurrent(primary));
    {
      tha::GpuEncoder encoder(argv[1],512,512,30,1200000,tha::ContextPolicy::BorrowCurrent);++owners;
      encoder.large_mesh(3,{0,1,2},{0},camera);encoder.update_mesh_host(xyz);encoder.encode(0,1000000,true);++successful_frames;
    }
    CUcontext after=nullptr;checked(cuCtxGetCurrent(&after));require(after==primary,"borrowed SDK context was not restored");
    CUdeviceptr still_valid=0;checked(cuMemAlloc(&still_valid,32));checked(cuMemFree(still_valid));CUcontext popped=nullptr;checked(cuCtxPopCurrent(&popped));checked(cuDevicePrimaryCtxRelease(device));
    std::cout<<"{\"full_size_gpu_contract\":\"passed\",\"owner_lifetimes\":"<<owners<<",\"successful_frames\":"<<successful_frames
      <<",\"global_reference_overflow_rejected\":true,\"per_tile_overflow_rejected\":true,\"wrong_context_rejected\":true,\"borrowed_context_survives\":true}\n";
    return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
