#include "claire_bridge.h"
#include "gpu_encoder.h"
#include "large_file.h"
#include <audio2face/audio2face.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <thread>

namespace {
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
void checked(CUresult code) {require(code==CUDA_SUCCESS,"Claire timeline CUDA operation failed");}
void checked(std::error_code code) {if(code)throw std::runtime_error(code.message());}
void message(char* output,size_t size,const char* value) {if(output && size)std::snprintf(output,size,"%s",value);}
struct Destroy {template<class T>void operator()(T* value)const {if(value)value->Destroy();}};
constexpr size_t vertices=61520+5602, slot_count=32;
struct Slot {CUdeviceptr data=0;uint64_t target=0,neural=0;};
}

struct tha_claire {
  // Reverse destruction keeps the SDK primary context alive through NVENC.
  std::unique_ptr<nva2f::IGeometryExecutorBundle,Destroy> bundle;
  std::unique_ptr<tha::GpuEncoder> encoder;
  std::thread::id owner=std::this_thread::get_id();CUcontext context=nullptr;
  CUstream copies=nullptr;CUevent producer=nullptr;
  std::array<Slot,slot_count> slots{};size_t head=0,count=0;
  tha_claire_metrics metrics{};std::string failure;
  uint64_t last_target=0,last_neural=0;int64_t pts_origin=0,last_pts=-1;bool rendered=false,poisoned=false;
  tha::EncodedFrame packet;

  void ownership()const {
    require(owner==std::this_thread::get_id(),"Claire handle belongs to its owner thread");
    CUcontext current=nullptr;checked(cuCtxGetCurrent(&current));require(context==current,"Claire CUDA owner context changed");
  }
  void usable()const {ownership();require(!poisoned,"Claire epoch failed; retire the handle");}
  ~tha_claire() {
    // All public operations fence copies before returning, including callbacks.
    for(auto& slot:slots)if(slot.data)cuMemFree(slot.data);
    if(producer)cuEventDestroy(producer);
    if(copies)cuStreamDestroy(copies);
  }
  static bool callback(void* pointer,const nva2f::IGeometryExecutor::Results& result) {
    auto& state=*static_cast<tha_claire*>(pointer);
    try {
      state.ownership();const auto index=state.metrics.neural_frames;
      require(result.trackIndex==0 && result.timeStampCurrentFrame==static_cast<int64_t>(index*16000/60) &&
        result.timeStampNextFrame==static_cast<int64_t>((index+1)*16000/60),"Claire source timestamps changed");
      require(result.skinGeometry.Size()==61520*3 && result.tongueGeometry.Size()==5602*3,"Claire source topology changed");
      if(!index)state.metrics.first_result_after_samples=state.metrics.offered_samples;
      ++state.metrics.neural_frames;
      if(index%2)return true; // Independent 60 Hz model and 30 Hz presentation.
      require(state.count<slot_count,"Claire motion exceeds 32-frame / 1067-ms queue bound");
      auto& slot=state.slots[(state.head+state.count)%slot_count];
      // Record each source dependency before reading it, then fence the owned
      // copy before the callback returns and the SDK reuses its output buffers.
      checked(cuEventRecord(state.producer,result.skinCudaStream));checked(cuStreamWaitEvent(state.copies,state.producer,0));
      checked(cuMemcpyDtoDAsync(slot.data,reinterpret_cast<CUdeviceptr>(result.skinGeometry.Data()),61520*3*sizeof(float),state.copies));
      checked(cuEventRecord(state.producer,result.tongueCudaStream));checked(cuStreamWaitEvent(state.copies,state.producer,0));
      checked(cuMemcpyDtoDAsync(slot.data+61520*3*sizeof(float),reinterpret_cast<CUdeviceptr>(result.tongueGeometry.Data()),5602*3*sizeof(float),state.copies));
      checked(cuStreamSynchronize(state.copies));
      slot.target=index/2*1600;slot.neural=result.timeStampCurrentFrame;
      ++state.count;state.metrics.queue_peak_frames=std::max<uint64_t>(state.metrics.queue_peak_frames,state.count);
      return true;
    }catch(const std::exception& error){state.failure=error.what();state.poisoned=true;return false;}
  }
};

extern "C" int tha_claire_open(const char* ptx,const char* scene,const char* model,tha_claire** result,char* error,size_t size) {
  if(result)*result=nullptr;
  try {
    require(ptx && scene && model && result,"Claire open paths/result required");
    const char* lease=std::getenv("THA_DEVICE_LEASE_FD");require(lease && fcntl(std::stoi(lease),F_GETFD)>=0,"inherited device lease required");
    auto state=std::make_unique<tha_claire>();const auto geometry=tha::read_large_geometry(scene);
    require(geometry.vertices.size()==vertices*3,"exact matched Claire skin/tongue scene required");
    state->bundle.reset(nva2f::ReadRegressionGeometryExecutorBundle(1,model,nva2f::IGeometryExecutor::ExecutionOption::All,60,1,nullptr));
    require(bool(state->bundle),"verified Claire executor unavailable");checked(cuCtxGetCurrent(&state->context));
    state->encoder=std::make_unique<tha::GpuEncoder>(ptx,512,512,30,1200000,tha::ContextPolicy::BorrowCurrent);
    state->encoder->large_mesh(vertices,geometry.faces,geometry.materials,geometry.camera);
    // One immutable neutral asset upload for decoder preroll. Dynamic SDK poses
    // still use owned device copies; this is not a neural geometry readback.
    state->encoder->update_mesh_host(geometry.vertices);
    checked(cuStreamCreate(&state->copies,CU_STREAM_NON_BLOCKING));checked(cuEventCreate(&state->producer,CU_EVENT_DISABLE_TIMING));
    for(auto& slot:state->slots)checked(cuMemAlloc(&slot.data,vertices*3*sizeof(float)));
    auto& executor=state->bundle->GetExecutor();require(executor.GetSamplingRate()==16000 && executor.GetNbTracks()==1,"Claire mono/16k source required");
    checked(executor.SetResultsCallback(tha_claire::callback,state.get()));
    auto& emotions=state->bundle->GetEmotionAccumulator(0);std::vector<float> neutral(emotions.GetEmotionSize(),0.f);
    checked(emotions.Accumulate(0,nva2x::HostTensorFloatConstView{neutral.data(),neutral.size()},state->bundle->GetCudaStream().Data()));checked(emotions.Close());
    checked(state->bundle->GetCudaStream().Synchronize());*result=state.release();return 0;
  }catch(const std::exception& exception){message(error,size,exception.what());return -1;}
}

extern "C" int tha_claire_push(tha_claire* state,const float* samples,size_t count,uint64_t start,int eof,char* error,size_t size) {
  try {
    require(state,"Claire handle required");state->usable();
    require(!state->metrics.eof && start==state->metrics.offered_samples && count<=320 &&
      (samples || !count) && (count || eof) && (eof==0 || eof==1) && (!eof || start+count>0) &&
      start+count<=16000*120ULL,"Claire PCM gap/overlap/EOF/quantum bound");
    for(size_t i=0;i<count;++i)require(std::isfinite(samples[i]) && std::abs(samples[i])<=4,"finite bounded Claire PCM required");
    auto& accumulator=state->bundle->GetAudioAccumulator(0);auto& executor=state->bundle->GetExecutor();
    if(count)checked(accumulator.Accumulate(nva2x::HostTensorFloatConstView{samples,count},state->bundle->GetCudaStream().Data()));
    state->metrics.offered_samples+=count;
    state->metrics.retained_audio_peak_samples=std::max<uint64_t>(state->metrics.retained_audio_peak_samples,accumulator.NbAccumulatedSamples()-accumulator.NbDroppedSamples());
    if(eof){checked(accumulator.Close());state->metrics.eof=1;}
    while(nva2x::GetNbReadyTracks(executor)>0) {
      const auto executed=executor.Execute(nullptr);
      require(state->failure.empty(),state->failure.c_str());checked(executed);
    }
    checked(accumulator.DropSamplesBefore(executor.GetNextAudioSampleToRead(0)));
    checked(state->bundle->GetCudaStream().Synchronize());return 0;
  }catch(const std::exception& exception){
    if(state && state->owner==std::this_thread::get_id())state->poisoned=true;
    message(error,size,exception.what());return -1;
  }
}

extern "C" int tha_claire_render(tha_claire* state,uint64_t target,int64_t pts,int keyframe,tha_claire_packet* packet,char* error,size_t size) {
  if(packet)*packet={};
  try {
    require(state && packet,"Claire render handle/result required");state->usable();
    require(target%1600==0 && target<48000*120ULL && (!state->rendered || target>state->last_target) && pts>state->last_pts && (keyframe==0 || keyframe==1),"Claire render source grid/order/PTS mismatch");
    const auto relative_pts=static_cast<int64_t>(target*1000000/48000);
    require(pts>=relative_pts && (!state->rendered || pts-relative_pts==state->pts_origin),"Claire presentation timebase changed within epoch");
    // A missing future frame is backpressure, never permission to show it early
    // or hold an older mouth pose over new speech without declaring the fault.
    if(!state->count || state->slots[state->head].target>target ||
      state->slots[(state->head+state->count-1)%slot_count].target<target)return 0;
    while(state->count>1 && state->slots[(state->head+1)%slot_count].target<=target) {
      state->head=(state->head+1)%slot_count;--state->count;++state->metrics.discarded_video_frames;
    }
    auto& slot=state->slots[state->head];require(slot.target==target,"missing Claire pose on render grid");
    state->encoder->update_mesh_device(slot.data,vertices,state->copies);
    state->packet=state->encoder->encode(0,pts,keyframe!=0);
    *packet={state->packet.data.data(),state->packet.data.size(),state->packet.pts_us,slot.target,slot.neural,state->packet.keyframe?1:0};
    state->head=(state->head+1)%slot_count;--state->count;++state->metrics.rendered_frames;
    state->rendered=true;state->last_target=target;state->last_neural=slot.neural;state->pts_origin=pts-relative_pts;state->last_pts=pts;return 1;
  }catch(const std::exception& exception){
    if(state && state->owner==std::this_thread::get_id())state->poisoned=true;
    message(error,size,exception.what());return -1;
  }
}
extern "C" int tha_claire_get_metrics(tha_claire* state,tha_claire_metrics* output,char* error,size_t size) {
  try {require(state && output,"Claire metrics handle/result required");state->ownership();*output=state->metrics;output->queued_frames=state->count;return 0;}
  catch(const std::exception& exception){message(error,size,exception.what());return -1;}
}
extern "C" int tha_claire_close(tha_claire* state,char* error,size_t size) {
  try {require(state,"Claire handle required");state->ownership();delete state;return 0;}
  catch(const std::exception& exception){message(error,size,exception.what());return -1;}
}
extern "C" int tha_claire_rate(tha_claire* state,int bitrate,double fps,char* error,size_t size) {
  try {
    require(state,"Claire rate handle required");state->usable();
    require(bitrate>=100000 && bitrate<=8000000 && std::isfinite(fps) && fps>=1 && fps<=60,"Claire encoder rate bounds");
    state->encoder->bitrate(bitrate);state->encoder->framerate(fps);return 0;
  }catch(const std::exception& exception){message(error,size,exception.what());return -1;}
}
extern "C" int tha_claire_hold(tha_claire* state,int64_t pts,int keyframe,tha_claire_packet* packet,char* error,size_t size) {
  if(packet)*packet={};
  try {
    require(state && packet,"Claire idle handle/result required");state->usable();
    require(state->metrics.eof && state->count==0 && state->rendered && pts>state->last_pts && (keyframe==0 || keyframe==1),"Claire idle requires drained EOF and increasing presentation PTS");
    state->packet=state->encoder->encode(0,pts,keyframe!=0);state->last_pts=pts;
    *packet={state->packet.data.data(),state->packet.data.size(),pts,state->last_target,state->last_neural,state->packet.keyframe?1:0};return 1;
  }catch(const std::exception& exception){message(error,size,exception.what());return -1;}
}
extern "C" int tha_claire_preroll(tha_claire* state,int64_t pts,int keyframe,tha_claire_packet* packet,char* error,size_t size) {
  if(packet)*packet={};
  try {
    require(state && packet,"Claire preroll handle/result required");state->usable();
    require(!state->rendered && pts>state->last_pts && (keyframe==0 || keyframe==1),"Claire preroll requires increasing PTS before speech");
    state->packet=state->encoder->encode(0,pts,keyframe!=0);state->last_pts=pts;
    *packet={state->packet.data.data(),state->packet.data.size(),pts,UINT64_MAX,UINT64_MAX,state->packet.keyframe?1:0};return 1;
  }catch(const std::exception& exception){message(error,size,exception.what());return -1;}
}
