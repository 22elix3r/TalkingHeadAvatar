// Offline neural qualification: explicit geometry readback, never a zero-copy claim.
#include <audio2face/audio2face.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {
void require(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
void check(std::error_code result) {if(result) throw std::runtime_error(result.message());}
struct Destroy {template<class T> void operator()(T* item) const {if(item)item->Destroy();}};
using Bundle=std::unique_ptr<nva2f::IGeometryExecutorBundle,Destroy>;
void write_all(int descriptor,const void* data,std::size_t bytes) {
  const auto* cursor=static_cast<const char*>(data);
  while(bytes) {const auto count=write(descriptor,cursor,bytes);require(count>0,"private geometry output failed");cursor+=count;bytes-=count;}
}
struct Capture {
  int output;std::array<std::uint32_t,4> counts;std::size_t frames=0,offered=0,first_after=0;
  std::int64_t last=-1;std::string error;
  static bool callback(void* pointer,const nva2f::IGeometryExecutor::Results& result) {
    auto& state=*static_cast<Capture*>(pointer);
    try {
      require(result.trackIndex==0 && result.timeStampCurrentFrame>state.last && state.frames<1801,"invalid neural track/frame order");
      std::array<nva2x::DeviceTensorFloatConstView,4> views{result.skinGeometry,result.tongueGeometry,result.jawTransform,result.eyesRotation};
      std::array<cudaStream_t,4> streams{result.skinCudaStream,result.tongueCudaStream,result.jawCudaStream,result.eyesCudaStream};
      std::vector<float> values;std::size_t total=0;for(auto count:state.counts) total+=count;values.resize(total);
      std::size_t offset=0;
      for(std::size_t i=0;i<4;++i) {
        require(views[i].Size()==state.counts[i],"neural result topology/pose size mismatch");
        if(state.counts[i]) require(cudaMemcpyAsync(values.data()+offset,views[i].Data(),state.counts[i]*sizeof(float),cudaMemcpyDeviceToHost,streams[i])==cudaSuccess,"qualification geometry readback failed");
        offset+=state.counts[i];
      }
      for(auto stream:streams) require(cudaStreamSynchronize(stream)==cudaSuccess,"qualification result stream failed");
      require(std::all_of(values.begin(),values.end(),[](float value){return std::isfinite(value);}),"nonfinite neural geometry");
      const std::int64_t stamps[]{result.timeStampCurrentFrame,result.timeStampNextFrame};
      write_all(state.output,stamps,sizeof(stamps));write_all(state.output,values.data(),values.size()*sizeof(float));
      if(!state.frames) state.first_after=state.offered;
      state.last=result.timeStampCurrentFrame;++state.frames;return true;
    } catch(const std::exception& error) {state.error=error.what();return false;}
  }
};
}
int main(int argc,char** argv) {
  int output=-1;
  try {
    require(argc==6,"usage: tha_claire_geometry private-model.json source-16k.f32 new-geometry.bin chunk-samples cancel-at-samples");
    const char* inherited=std::getenv("THA_DEVICE_LEASE_FD");require(inherited && fcntl(std::stoi(inherited),F_GETFD)>=0,"inherited admitted device lease required");
    const std::size_t chunk=std::stoull(argv[4]),cancel_at=std::stoull(argv[5]);require(chunk>0 && chunk<=480000,"bounded source quantum required");
    struct stat info{};require(lstat(argv[2],&info)==0 && S_ISREG(info.st_mode) && info.st_size>0 && info.st_size<=16000*30*4 && info.st_size%4==0,"bounded 16 kHz float PCM source required");
    std::vector<float> audio(info.st_size/4);std::ifstream source(argv[2],std::ios::binary);source.read(reinterpret_cast<char*>(audio.data()),info.st_size);
    require(bool(source) && std::all_of(audio.begin(),audio.end(),[](float value){return std::isfinite(value) && std::abs(value)<=4;}),"finite source PCM required");
    const auto started=std::chrono::steady_clock::now();
    Bundle bundle(nva2f::ReadRegressionGeometryExecutorBundle(1,argv[1],nva2f::IGeometryExecutor::ExecutionOption::All,60,1,nullptr));
    require(bool(bundle),"pinned Claire geometry executor unavailable");
    auto& executor=bundle->GetExecutor();require(executor.GetSamplingRate()==16000 && executor.GetNbTracks()==1,"qualified neural source contract mismatch");
    output=open(argv[3],O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);require(output>=0,"new private output required");
    Capture capture{output,{static_cast<std::uint32_t>(executor.GetSkinGeometrySize()),static_cast<std::uint32_t>(executor.GetTongueGeometrySize()),
      static_cast<std::uint32_t>(executor.GetJawTransformSize()),static_cast<std::uint32_t>(executor.GetEyesRotationSize())},0,0,0,-1,{}};
    require(capture.counts[0]==61520*3 && capture.counts[1]==5602*3 && capture.counts[2]<=32 && capture.counts[3]<=8,"unexpected Claire geometry basis");
    write_all(output,"THACLR1\0",8);write_all(output,capture.counts.data(),sizeof(capture.counts));
    check(executor.SetResultsCallback(Capture::callback,&capture));
    auto& accumulator=bundle->GetAudioAccumulator(0);auto& emotions=bundle->GetEmotionAccumulator(0);
    std::vector<float> neutral_emotion(emotions.GetEmotionSize(),0.f);const auto stream=bundle->GetCudaStream().Data();
    check(emotions.Accumulate(0,nva2x::HostTensorFloatConstView{neutral_emotion.data(),neutral_emotion.size()},stream));check(emotions.Close());
    std::size_t peak=0,pre_eof=0;bool cancelled=false;
    auto execute=[&] {
      while(nva2x::GetNbReadyTracks(executor)>0) {check(executor.Execute(nullptr));require(capture.error.empty(),capture.error.c_str());}
      check(accumulator.DropSamplesBefore(executor.GetNextAudioSampleToRead(0)));
    };
    for(std::size_t offset=0;offset<audio.size();offset+=chunk) {
      const auto count=std::min(chunk,audio.size()-offset);capture.offered=offset+count;
      check(accumulator.Accumulate(nva2x::HostTensorFloatConstView{audio.data()+offset,count},stream));
      peak=std::max(peak,accumulator.NbAccumulatedSamples()-accumulator.NbDroppedSamples());execute();
      if(cancel_at && capture.offered>=cancel_at) {cancelled=true;check(executor.Reset(0));check(accumulator.Reset());break;}
    }
    pre_eof=capture.frames;
    if(!cancelled) {check(accumulator.Close());execute();require(capture.frames==executor.GetTotalNbFrames(0),"neural EOF frame conservation failed");}
    check(bundle->GetCudaStream().Synchronize());bundle.reset();require(fsync(output)==0,"geometry persistence failed");close(output);output=-1;
    std::cout<<"{\"frames\":"<<capture.frames<<",\"frames_before_eof\":"<<pre_eof<<",\"first_result_after_samples\":"<<capture.first_after
      <<",\"peak_retained_audio_samples\":"<<peak<<",\"last_frame_sample\":"<<capture.last<<",\"cancelled\":"<<(cancelled?"true":"false")
      <<",\"source_samples\":"<<audio.size()<<",\"offered_samples\":"<<capture.offered<<",\"elapsed_ms\":"
      <<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()<<",\"qualification_geometry_readback\":true}\n";
    return 0;
  } catch(const std::exception& error) {if(output>=0)close(output);std::cerr<<"Claire geometry probe: "<<error.what()<<'\n';return 1;}
}
