#include "gpu_encoder.h"
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <cuda.h>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>
}

namespace tha {
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point begin) {return std::chrono::duration<double,std::milli>(Clock::now()-begin).count();}
static void cu(CUresult result,const char* operation) {
  if(result!=CUDA_SUCCESS) {const char* name=nullptr;cuGetErrorName(result,&name);throw std::runtime_error(std::string(operation)+": "+(name?name:"CUDA failure"));}
}
static void av(int result,const char* operation) {
  if(result<0) {char text[AV_ERROR_MAX_STRING_SIZE];av_strerror(result,text,sizeof(text));throw std::runtime_error(std::string(operation)+": "+text);}
}

struct GpuEncoder::State {
  std::thread::id owner=std::this_thread::get_id();
  int width,height,fps,next_slot=0;
  std::uint64_t bytes=0;
  AVBufferRef *device=nullptr,*frames=nullptr;
  AVCodecContext* codec=nullptr;
  std::array<AVFrame*,3> slots{};
  AVPacket* packet=nullptr;
  CUcontext context=nullptr;
  CUmodule module=nullptr;
  CUfunction render=nullptr,convert=nullptr;
  CUstream render_stream=nullptr,convert_stream=nullptr;
  std::array<CUevent,4> events{};
  CUdeviceptr rgba=0;
  std::size_t rgb_pitch=0;
  bool pushed=false;
  std::int64_t previous_pts=-1;
  void thread() const {if(owner!=std::this_thread::get_id()) throw std::runtime_error("media owner thread mismatch");}
  void option(const char* name,const char* value) {av(av_opt_set(codec->priv_data,name,value,0),name);}
  ~State() {
    // NVENC must release its frame refs before surfaces/context are retired.
    if(codec) avcodec_free_context(&codec);
    for(auto& frame:slots) av_frame_free(&frame);
    av_packet_free(&packet);
    if(render_stream) cuStreamSynchronize(render_stream);
    if(convert_stream) cuStreamSynchronize(convert_stream);
    for(auto event:events) if(event) cuEventDestroy(event);
    if(rgba) cuMemFree(rgba);
    if(module) cuModuleUnload(module);
    if(convert_stream) cuStreamDestroy(convert_stream);
    if(render_stream) cuStreamDestroy(render_stream);
    av_buffer_unref(&frames);
    if(pushed) {CUcontext popped=nullptr;cuCtxPopCurrent(&popped);}
    av_buffer_unref(&device);
  }
  State(const std::string& ptx,int w,int h,int rate,int bits):width(w),height(h),fps(rate) {
    if(w<16 || h<16 || w>1024 || h>1024 || (w%2) || (h%2) || rate<1 || rate>60 || bits<100000 || bits>8000000)
      throw std::invalid_argument("unsupported bounded media shape/rate");
    try {
      av(av_hwdevice_ctx_create(&device,AV_HWDEVICE_TYPE_CUDA,"0",nullptr,0),"CUDA device");
      auto* device_context=reinterpret_cast<AVHWDeviceContext*>(device->data);
      context=reinterpret_cast<AVCUDADeviceContext*>(device_context->hwctx)->cuda_ctx;
      cu(cuCtxPushCurrent(context),"own CUDA context");pushed=true;
      cu(cuModuleLoad(&module,ptx.c_str()),"load pinned PTX");
      cu(cuModuleGetFunction(&render,module,"render_pattern"),"render entry point");
      cu(cuModuleGetFunction(&convert,module,"rgba_to_nv12"),"conversion entry point");
      cu(cuStreamCreate(&render_stream,CU_STREAM_NON_BLOCKING),"render stream");
      cu(cuStreamCreate(&convert_stream,CU_STREAM_NON_BLOCKING),"convert stream");
      for(auto& event:events) cu(cuEventCreate(&event,CU_EVENT_DEFAULT),"media event");
      cu(cuMemAllocPitch(&rgba,&rgb_pitch,width*4,height,16),"preallocate RGBA surface");
      frames=av_hwframe_ctx_alloc(device);
      if(!frames) throw std::bad_alloc();
      auto* pool=reinterpret_cast<AVHWFramesContext*>(frames->data);
      pool->format=AV_PIX_FMT_CUDA;pool->sw_format=AV_PIX_FMT_NV12;
      pool->width=width;pool->height=height;pool->initial_pool_size=3;
      av(av_hwframe_ctx_init(frames),"preallocate NV12 pool");
      for(auto& frame:slots) {
        frame=av_frame_alloc();if(!frame) throw std::bad_alloc();
        av(av_hwframe_get_buffer(frames,frame,0),"retain NV12 slot");
        frame->color_range=AVCOL_RANGE_MPEG;frame->colorspace=AVCOL_SPC_BT709;
        frame->color_primaries=AVCOL_PRI_BT709;frame->color_trc=AVCOL_TRC_BT709;
      }
      const AVCodec* implementation=avcodec_find_encoder_by_name("h264_nvenc");
      if(!implementation) throw std::runtime_error("h264_nvenc unavailable; no software substitution");
      codec=avcodec_alloc_context3(implementation);if(!codec) throw std::bad_alloc();
      codec->width=width;codec->height=height;codec->pix_fmt=AV_PIX_FMT_CUDA;
      codec->time_base=AVRational{1,1000000};codec->framerate=AVRational{fps,1};
      codec->bit_rate=bits;codec->gop_size=fps;codec->max_b_frames=0;
      codec->color_range=AVCOL_RANGE_MPEG;codec->colorspace=AVCOL_SPC_BT709;
      codec->color_primaries=AVCOL_PRI_BT709;codec->color_trc=AVCOL_TRC_BT709;
      codec->hw_frames_ctx=av_buffer_ref(frames);
      codec->flags|=AV_CODEC_FLAG_LOW_DELAY;
      option("preset","p1");option("tune","ull");option("rc","cbr");
      option("profile","baseline");option("level","3.1");
      option("zerolatency","1");option("rc-lookahead","0");option("delay","0");
      option("surfaces","3");option("forced-idr","1");option("aud","1");
      av(avcodec_open2(codec,implementation,nullptr),"open low-delay NVENC");
      packet=av_packet_alloc();if(!packet) throw std::bad_alloc();
    } catch(...) {
      // unique_ptr's State destructor is not invoked for a failed constructor.
      if(codec) avcodec_free_context(&codec);
      for(auto& frame:slots) av_frame_free(&frame);
      for(auto& event:events) {if(event) cuEventDestroy(event);event=nullptr;}
      if(rgba) {cuMemFree(rgba);rgba=0;}
      if(module) {cuModuleUnload(module);module=nullptr;}
      if(convert_stream) {cuStreamDestroy(convert_stream);convert_stream=nullptr;}
      if(render_stream) {cuStreamDestroy(render_stream);render_stream=nullptr;}
      av_buffer_unref(&frames);
      if(pushed) {CUcontext popped=nullptr;cuCtxPopCurrent(&popped);pushed=false;}
      av_buffer_unref(&device);throw;
    }
  }
};

GpuEncoder::GpuEncoder(const std::string& ptx,int width,int height,int fps,int bitrate)
  :state_(std::make_unique<State>(ptx,width,height,fps,bitrate)) {}
GpuEncoder::~GpuEncoder()=default;
void GpuEncoder::bitrate(int bits) {
  state_->thread();if(bits<100000 || bits>8000000) throw std::invalid_argument("bitrate outside profile");
  state_->codec->bit_rate=bits;
}
void GpuEncoder::framerate(double fps) {
  state_->thread();if(!std::isfinite(fps) || fps<1 || fps>60) throw std::invalid_argument("invalid encoder rate-control framerate");
  state_->codec->framerate=av_d2q(fps,100000);
}
std::uint64_t GpuEncoder::coded_bytes() const {return state_->bytes;}

EncodedFrame GpuEncoder::encode(int pattern,std::int64_t pts,bool force) {
  auto& s=*state_;s.thread();
  if(pts<=s.previous_pts) throw std::invalid_argument("nonmonotonic video PTS");
  s.previous_pts=pts;
  const auto ready_begin=Clock::now();
  const int index=s.next_slot;s.next_slot=(s.next_slot+1)%3;
  AVFrame* frame=s.slots[index];
  if(!av_frame_is_writable(frame)) throw std::runtime_error("encoder still retains a slot; refusing overwrite/allocation");
  cu(cuEventRecord(s.events[0],s.render_stream),"render start");
  void* render_args[]={&s.rgba,&s.rgb_pitch,&s.width,&s.height,&pattern};
  cu(cuLaunchKernel(s.render,(s.width+15)/16,(s.height+15)/16,1,16,16,1,0,s.render_stream,render_args,nullptr),"render pattern");
  cu(cuEventRecord(s.events[1],s.render_stream),"render complete");
  cu(cuStreamWaitEvent(s.convert_stream,s.events[1],0),"conversion waits for renderer");
  cu(cuEventRecord(s.events[2],s.convert_stream),"conversion start");
  auto* y=frame->data[0];auto* uv=frame->data[1];
  std::size_t yp=frame->linesize[0],uvp=frame->linesize[1];
  void* convert_args[]={&s.rgba,&s.rgb_pitch,&y,&yp,&uv,&uvp,&s.width,&s.height};
  cu(cuLaunchKernel(s.convert,(s.width+15)/16,(s.height+15)/16,1,16,16,1,0,s.convert_stream,convert_args,nullptr),"convert NV12");
  cu(cuEventRecord(s.events[3],s.convert_stream),"NV12 complete");
  // FFmpeg/NVENC does not consume our CUDA event. Complete that dependency
  // explicitly before passing the GPU surface to the encoder.
  cu(cuEventSynchronize(s.events[3]),"encoder waits for conversion");
  EncodedFrame result;result.pts_us=pts;result.slot=index;
  cu(cuEventElapsedTime(&result.render_gpu_ms,s.events[0],s.events[1]),"render elapsed");
  cu(cuEventElapsedTime(&result.convert_gpu_ms,s.events[2],s.events[3]),"conversion elapsed");
  result.ready_wall_ms=ms(ready_begin);
  frame->pts=pts;frame->pict_type=force?AV_PICTURE_TYPE_I:AV_PICTURE_TYPE_NONE;
  const auto encode_begin=Clock::now();
  av(avcodec_send_frame(s.codec,frame),"submit GPU surface");
  av(avcodec_receive_packet(s.codec,s.packet),"receive complete low-delay access unit");
  if(s.packet->pts!=pts) {av_packet_unref(s.packet);throw std::runtime_error("unexpected NVENC delay/reordering");}
  result.keyframe=(s.packet->flags&AV_PKT_FLAG_KEY)!=0;
  result.data.assign(s.packet->data,s.packet->data+s.packet->size);
  s.bytes+=s.packet->size;av_packet_unref(s.packet);
  result.encode_wall_ms=ms(encode_begin);
  return result;
}
}
