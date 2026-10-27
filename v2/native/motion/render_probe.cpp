// Direct SDK geometry -> owned GPU mesh -> NV12/NVENC. No host geometry stage.
#include "gpu_encoder.h"
#include "large_file.h"
#include <audio2face/audio2face.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
void check(std::error_code code) {if(code)throw std::runtime_error(code.message());}
void write_all(int output,const void* data,std::size_t bytes) {
  const auto* source=static_cast<const char*>(data);
  while(bytes) {const auto count=write(output,source,bytes);require(count>0,"private render output failed");source+=count;bytes-=count;}
}
void private_file(const std::string& name,const void* data,std::size_t bytes) {
  const int output=open(name.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);require(output>=0,"new private qualification output required");
  try {write_all(output,data,bytes);close(output);} catch(...) {close(output);throw;}
}
struct Destroy {template<class T>void operator()(T* item)const {if(item)item->Destroy();}};
using Bundle=std::unique_ptr<nva2f::IGeometryExecutorBundle,Destroy>;
struct Capture {
  tha::GpuEncoder& encoder;
  int output;
  std::string prefix;
  std::size_t expected;
  bool readback;
  std::size_t frames=0,encoded=0,offered=0,first_after=0,readbacks=0;
  unsigned int maximum_tile_faces=0,maximum_references=0;
  double geometry_gpu_ms=0,render_gpu_ms=0;
  std::string failure;
  static bool callback(void* pointer,const nva2f::IGeometryExecutor::Results& result) {
    auto& state=*static_cast<Capture*>(pointer);
    try {
      require(result.trackIndex==0 && result.timeStampCurrentFrame==static_cast<std::int64_t>(state.frames*16000/60) &&
        result.timeStampNextFrame==static_cast<std::int64_t>((state.frames+1)*16000/60),"neural source timebase changed");
      require(result.skinGeometry.Size()==61520*3 && result.tongueGeometry.Size()==5602*3,"direct neural topology mismatch");
      if(!state.frames)state.first_after=state.offered;
      const auto index=state.frames++;
      if(index%2)return true;
      const auto update=state.encoder.update_mesh_device(std::vector<tha::DeviceMeshPart>{
        {reinterpret_cast<CUdeviceptr>(result.skinGeometry.Data()),61520,result.skinCudaStream},
        {reinterpret_cast<CUdeviceptr>(result.tongueGeometry.Data()),5602,result.tongueCudaStream}});
      state.geometry_gpu_ms+=update.gpu_ms;state.maximum_tile_faces=std::max(state.maximum_tile_faces,update.maximum_tile_faces);
      state.maximum_references=std::max(state.maximum_references,update.tile_references);
      const auto packet=state.encoder.encode(0,1000000+result.timeStampCurrentFrame*1000000/16000,state.encoded==0 || state.encoded==31);
      require((state.encoded!=0 && state.encoded!=31) || packet.keyframe,"required direct-render keyframe missing");
      write_all(state.output,packet.data.data(),packet.data.size());state.render_gpu_ms+=packet.render_gpu_ms;
      if(state.readback && (index==0 || index==state.expected/2/2*2)) {
        const auto image=state.encoder.qualification_readback_rgba();
        private_file(state.prefix+"-"+std::to_string(index)+".rgba",image.data(),image.size());++state.readbacks;
        if(index==0) {
          const auto projected=state.encoder.qualification_projected_vertices();
          private_file(state.prefix+"-projected.bin",projected.data(),projected.size()*sizeof(double));
        }
      }
      ++state.encoded;return true;
    } catch(const std::exception& error) {state.failure=error.what();return false;}
  }
};
}
int main(int argc,char** argv) {
  int output=-1;
  try {
    require(argc==6 || (argc==7 && std::string(argv[6])=="--no-readback"),"usage: tha_claire_render kernels.ptx research-scene.bin model.json source-16k.f32 new-output-prefix [--no-readback]");
    const char* inherited=std::getenv("THA_DEVICE_LEASE_FD");require(inherited && fcntl(std::stoi(inherited),F_GETFD)>=0,"inherited admitted device lease required");
    const auto scene=tha::read_large_geometry(argv[2]);require(scene.vertices.size()==(61520+5602)*3,"matched full-size skin/tongue research profile required");
    struct stat info{};require(lstat(argv[4],&info)==0 && S_ISREG(info.st_mode) && info.st_size>0 && info.st_size<=16000*30*4 && info.st_size%4==0,"bounded 16 kHz source float PCM required");
    std::vector<float> audio(info.st_size/4);std::ifstream source(argv[4],std::ios::binary);source.read(reinterpret_cast<char*>(audio.data()),info.st_size);
    require(source && std::all_of(audio.begin(),audio.end(),[](float value){return std::isfinite(value) && std::abs(value)<=4;}),"finite bounded source PCM required");
    const auto began=std::chrono::steady_clock::now();
    Bundle bundle(nva2f::ReadRegressionGeometryExecutorBundle(1,argv[3],nva2f::IGeometryExecutor::ExecutionOption::All,60,1,nullptr));
    require(bool(bundle),"verified Claire SDK executor unavailable");
    // SDK owns its primary context. FFmpeg borrows it without changing scheduling
    // flags or destroying it; the encoder is destroyed before the SDK bundle.
    tha::GpuEncoder encoder(argv[1],512,512,30,1200000,tha::ContextPolicy::BorrowCurrent);
    encoder.large_mesh(scene.vertices.size()/3,scene.faces,scene.materials,scene.camera);
    const auto filename=std::string(argv[5])+".h264";
    output=open(filename.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);require(output>=0,"new private coded output required");
    Capture capture{encoder,output,argv[5],(audio.size()*60+15999)/16000,argc==6,0,0,0,0,0,0,0,0,0,{}};
    auto& executor=bundle->GetExecutor();require(executor.GetSamplingRate()==16000 && executor.GetNbTracks()==1,"direct SDK source contract mismatch");
    check(executor.SetResultsCallback(Capture::callback,&capture));
    const auto stream=bundle->GetCudaStream().Data();auto& emotions=bundle->GetEmotionAccumulator(0);
    std::vector<float> neutral(emotions.GetEmotionSize(),0.f);
    check(emotions.Accumulate(0,nva2x::HostTensorFloatConstView{neutral.data(),neutral.size()},stream));check(emotions.Close());
    auto& accumulator=bundle->GetAudioAccumulator(0);std::size_t retained_peak=0;
    auto execute=[&] {
      while(nva2x::GetNbReadyTracks(executor)>0) {check(executor.Execute(nullptr));require(capture.failure.empty(),capture.failure.c_str());}
      check(accumulator.DropSamplesBefore(executor.GetNextAudioSampleToRead(0)));
    };
    for(std::size_t offset=0;offset<audio.size();offset+=320) {
      const auto count=std::min<std::size_t>(320,audio.size()-offset);capture.offered=offset+count;
      check(accumulator.Accumulate(nva2x::HostTensorFloatConstView{audio.data()+offset,count},stream));
      retained_peak=std::max(retained_peak,accumulator.NbAccumulatedSamples()-accumulator.NbDroppedSamples());execute();
    }
    const auto pre_eof=capture.frames;check(accumulator.Close());execute();check(bundle->GetCudaStream().Synchronize());
    require(capture.frames==capture.expected && capture.frames==executor.GetTotalNbFrames(0),"direct-render neural EOF frame conservation failed");
    require(fsync(output)==0,"direct-render output persistence failed");close(output);output=-1;
    std::cout<<"{\"source_samples\":"<<audio.size()<<",\"neural_frames\":"<<capture.frames<<",\"frames_before_eof\":"<<pre_eof
      <<",\"encoded_frames\":"<<capture.encoded<<",\"first_result_after_samples\":"<<capture.first_after<<",\"retained_audio_peak_samples\":"<<retained_peak
      <<",\"geometry_gpu_ms\":"<<capture.geometry_gpu_ms<<",\"render_gpu_ms\":"<<capture.render_gpu_ms<<",\"maximum_tile_faces\":"<<capture.maximum_tile_faces
      <<",\"maximum_tile_references\":"<<capture.maximum_references<<",\"status_readback_bytes_per_update\":20,\"qualification_rgba_readbacks\":"<<capture.readbacks
      <<",\"qualification_projection_readbacks\":"<<(argc==6?1:0)<<",\"geometry_host_stage\":false,\"borrowed_sdk_cuda_context\":true,\"elapsed_ms\":"
      <<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count()<<"}\n";
    return 0;
  } catch(const std::exception& error) {if(output>=0)close(output);std::cerr<<"direct Claire render: "<<error.what()<<'\n';return 1;}
}
