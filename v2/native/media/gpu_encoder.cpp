#include "gpu_encoder.h"
#include <algorithm>
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
  CUfunction render=nullptr,convert=nullptr,render_mesh=nullptr;
  CUfunction project_large=nullptr,bin_large=nullptr,scan_large=nullptr,fill_large=nullptr,render_large=nullptr;
  CUstream render_stream=nullptr,convert_stream=nullptr;
  std::array<CUevent,4> events{};
  CUdeviceptr rgba=0;
  CUdeviceptr mesh_vertices=0,mesh_faces=0,mesh_bounds=0;
  int mesh_face_count=0;
  int large_vertex_count=0,tile_columns=0,tile_rows=0,tile_edge=8,tile_capacity=8192,reference_capacity=4*1024*1024;
  bool large_ready=false;
  // world, camera, materials, colors, tile counts, compact lists, status,
  // face bounds, tile offsets, tile cursors. Fixed allocation, no per-frame growth.
  std::array<CUdeviceptr,10> large_buffers{};
  std::array<CUevent,2> geometry_events{};
  std::array<CUevent,8> source_events{};
  std::size_t rgb_pitch=0;
  bool pushed=false;
  std::int64_t previous_pts=-1;
  void thread() const {
    if(owner!=std::this_thread::get_id()) throw std::runtime_error("media owner thread mismatch");
    CUcontext active=nullptr;cu(cuCtxGetCurrent(&active),"check media owner context");
    if(active!=context) throw std::runtime_error("media owner CUDA context changed");
  }
  MeshUpdate finish_geometry();
  void option(const char* name,const char* value) {av(av_opt_set(codec->priv_data,name,value,0),name);}
  ~State() {
    // NVENC must release its frame refs before surfaces/context are retired.
    if(codec) avcodec_free_context(&codec);
    for(auto& frame:slots) av_frame_free(&frame);
    av_packet_free(&packet);
    if(render_stream) cuStreamSynchronize(render_stream);
    if(convert_stream) cuStreamSynchronize(convert_stream);
    for(auto event:events) if(event) cuEventDestroy(event);
    for(auto event:geometry_events) if(event) cuEventDestroy(event);
    for(auto event:source_events) if(event) cuEventDestroy(event);
    for(auto pointer:large_buffers) if(pointer) cuMemFree(pointer);
    if(rgba) cuMemFree(rgba);
    if(mesh_vertices) cuMemFree(mesh_vertices);
    if(mesh_faces) cuMemFree(mesh_faces);
    if(mesh_bounds) cuMemFree(mesh_bounds);
    if(module) cuModuleUnload(module);
    if(convert_stream) cuStreamDestroy(convert_stream);
    if(render_stream) cuStreamDestroy(render_stream);
    av_buffer_unref(&frames);
    if(pushed) {CUcontext popped=nullptr;cuCtxPopCurrent(&popped);}
    av_buffer_unref(&device);
  }
  State(const std::string& ptx,int w,int h,int rate,int bits,ContextPolicy policy):width(w),height(h),fps(rate) {
    if(w<16 || h<16 || w>1024 || h>1024 || (w%2) || (h%2) || rate<1 || rate>60 || bits<100000 || bits>8000000)
      throw std::invalid_argument("unsupported bounded media shape/rate");
    try {
      int flags=0;
      if(policy==ContextPolicy::BorrowCurrent) {
        CUcontext active=nullptr;cu(cuCtxGetCurrent(&active),"borrow initialized SDK context");
        if(!active) throw std::runtime_error("borrowed media context must already exist");
        CUdevice selected=-1;cu(cuCtxGetDevice(&selected),"borrowed context device");
        if(selected!=0) throw std::runtime_error("borrowed media context is not admitted device zero");
        flags=AV_CUDA_USE_CURRENT_CONTEXT;
      }
      av(av_hwdevice_ctx_create(&device,AV_HWDEVICE_TYPE_CUDA,"0",nullptr,flags),"CUDA device");
      auto* device_context=reinterpret_cast<AVHWDeviceContext*>(device->data);
      context=reinterpret_cast<AVCUDADeviceContext*>(device_context->hwctx)->cuda_ctx;
      cu(cuCtxPushCurrent(context),"own CUDA context");pushed=true;
      cu(cuModuleLoad(&module,ptx.c_str()),"load pinned PTX");
      cu(cuModuleGetFunction(&render,module,"render_pattern"),"render entry point");
      cu(cuModuleGetFunction(&render_mesh,module,"render_mesh"),"mesh render entry point");
      cu(cuModuleGetFunction(&project_large,module,"project_large_mesh"),"large mesh projection entry point");
      cu(cuModuleGetFunction(&bin_large,module,"bin_large_mesh"),"large mesh binning entry point");
      cu(cuModuleGetFunction(&scan_large,module,"scan_large_tiles"),"compact tile offsets entry point");
      cu(cuModuleGetFunction(&fill_large,module,"fill_large_tiles"),"compact tile fill entry point");
      cu(cuModuleGetFunction(&render_large,module,"render_large_mesh"),"large mesh render entry point");
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

GpuEncoder::GpuEncoder(const std::string& ptx,int width,int height,int fps,int bitrate,ContextPolicy policy)
  :state_(std::make_unique<State>(ptx,width,height,fps,bitrate,policy)) {}
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

void GpuEncoder::mesh(const std::vector<double>& xyz,const std::vector<std::uint32_t>& triangles) {
  auto& s=*state_;s.thread();
  if(s.large_vertex_count) throw std::runtime_error("full-size research topology already bound");
  if(s.previous_pts>=0 || s.mesh_face_count) throw std::runtime_error("serving geometry is immutable for this owner");
  if(xyz.size()%3 || xyz.size()<9 || xyz.size()>512*3 || triangles.size()%3 || triangles.empty() || triangles.size()>1024*3)
    throw std::invalid_argument("bounded mesh shape mismatch; no padding/truncation");
  for(std::size_t i=0;i<xyz.size();++i) {
    if(!std::isfinite(xyz[i]) || std::abs(xyz[i])>1000000 || (i%3==2 && xyz[i]<=0))
      throw std::invalid_argument("finite positive-depth projected geometry required");
  }
  for(std::size_t i=0;i<triangles.size();i+=3) {
    const auto a=triangles[i],b=triangles[i+1],c=triangles[i+2];
    if(a>=xyz.size()/3 || b>=xyz.size()/3 || c>=xyz.size()/3 || a==b || a==c || b==c)
      throw std::invalid_argument("invalid face binding; no index clamping");
  }
  std::vector<std::int32_t> bounds;
  bounds.reserve(triangles.size()/3*4);
  for(std::size_t i=0;i<triangles.size();i+=3) {
    const auto a=triangles[i]*3,b=triangles[i+1]*3,c=triangles[i+2]*3;
    // Pixel clipping follows the CPU rasterizer's floor/ceil convention. Face
    // indices/geometry remain unchanged; no shape repair occurs here.
    bounds.push_back(std::max(0,static_cast<int>(std::floor(std::min({xyz[a],xyz[b],xyz[c]})))));
    bounds.push_back(std::min(s.width-1,static_cast<int>(std::ceil(std::max({xyz[a],xyz[b],xyz[c]})))));
    bounds.push_back(std::max(0,static_cast<int>(std::floor(std::min({xyz[a+1],xyz[b+1],xyz[c+1]})))));
    bounds.push_back(std::min(s.height-1,static_cast<int>(std::ceil(std::max({xyz[a+1],xyz[b+1],xyz[c+1]})))));
  }
  CUdeviceptr vertices=0,faces=0,boxes=0;
  try {
    cu(cuMemAlloc(&vertices,xyz.size()*sizeof(double)),"immutable geometry allocation");
    cu(cuMemAlloc(&faces,triangles.size()*sizeof(std::uint32_t)),"immutable face allocation");
    cu(cuMemAlloc(&boxes,bounds.size()*sizeof(std::int32_t)),"immutable triangle bounds allocation");
    cu(cuMemcpyHtoD(vertices,xyz.data(),xyz.size()*sizeof(double)),"initial geometry upload");
    cu(cuMemcpyHtoD(faces,triangles.data(),triangles.size()*sizeof(std::uint32_t)),"initial face upload");
    cu(cuMemcpyHtoD(boxes,bounds.data(),bounds.size()*sizeof(std::int32_t)),"initial triangle bounds upload");
    cu(cuCtxSynchronize(),"initial geometry ready for nonblocking render stream");
  } catch(...) {if(vertices) cuMemFree(vertices);if(faces) cuMemFree(faces);if(boxes) cuMemFree(boxes);throw;}
  s.mesh_vertices=vertices;s.mesh_faces=faces;s.mesh_bounds=boxes;s.mesh_face_count=triangles.size()/3;
}

void GpuEncoder::large_mesh(std::size_t vertex_count,const std::vector<std::uint32_t>& triangles,
                            const std::vector<std::uint8_t>& materials,const MeshCamera& camera) {
  auto& s=*state_;s.thread();
  if(s.previous_pts>=0 || s.mesh_face_count || s.large_vertex_count) throw std::runtime_error("immutable topology already bound");
  if(s.width!=512 || s.height!=512 || vertex_count<3 || vertex_count>100000 || triangles.empty() ||
     triangles.size()%3 || triangles.size()>200000*3 || materials.size()!=triangles.size()/3)
    throw std::invalid_argument("full-size research mesh profile mismatch; no truncation");
  for(std::size_t i=0;i<triangles.size();i+=3) {
    const auto a=triangles[i],b=triangles[i+1],c=triangles[i+2];
    if(a>=vertex_count || b>=vertex_count || c>=vertex_count || a==b || a==c || b==c || materials[i/3]>1)
      throw std::invalid_argument("full-size topology/material identity mismatch");
  }
  for(double value:camera.rotation) if(!std::isfinite(value)) throw std::invalid_argument("finite camera rotation required");
  for(double value:camera.translation_m) if(!std::isfinite(value) || std::abs(value)>1000000) throw std::invalid_argument("finite bounded camera translation required");
  for(int a=0;a<3;++a) for(int b=0;b<3;++b) {
    double product=0;for(int k=0;k<3;++k) product+=camera.rotation[a*3+k]*camera.rotation[b*3+k];
    if(std::abs(product-(a==b?1.:0.))>1e-7) throw std::invalid_argument("camera rotation must be orthonormal");
  }
  const auto& r=camera.rotation;
  const double determinant=r[0]*(r[4]*r[8]-r[5]*r[7])-r[1]*(r[3]*r[8]-r[5]*r[6])+r[2]*(r[3]*r[7]-r[4]*r[6]);
  if(std::abs(determinant-1)>1e-7 || !std::isfinite(camera.units_to_meters) || camera.units_to_meters<=0 || camera.units_to_meters>1 ||
     !std::isfinite(camera.half_extent_m) || camera.half_extent_m<.0001 || camera.half_extent_m>10)
    throw std::invalid_argument("explicit proper camera rotation/units/extent required");
  std::array<double,14> packed{};
  std::copy(camera.rotation.begin(),camera.rotation.end(),packed.begin());
  std::copy(camera.translation_m.begin(),camera.translation_m.end(),packed.begin()+9);
  packed[12]=camera.units_to_meters;packed[13]=camera.half_extent_m;
  CUdeviceptr vertices=0,faces=0;std::array<CUdeviceptr,10> buffers{};
  std::array<CUevent,2> geometry{};std::array<CUevent,8> sources{};
  const int columns=(s.width+s.tile_edge-1)/s.tile_edge,rows=(s.height+s.tile_edge-1)/s.tile_edge;
  const std::array<std::size_t,10> sizes{vertex_count*3*sizeof(float),sizeof(packed),materials.size(),
    materials.size()*sizeof(std::uint32_t),static_cast<std::size_t>(columns*rows)*sizeof(std::uint32_t),
    static_cast<std::size_t>(s.reference_capacity)*sizeof(std::uint32_t),5*sizeof(std::uint32_t),
    materials.size()*4*sizeof(std::int32_t),static_cast<std::size_t>(columns*rows)*sizeof(std::uint32_t),
    static_cast<std::size_t>(columns*rows)*sizeof(std::uint32_t)};
  try {
    cu(cuMemAlloc(&vertices,vertex_count*3*sizeof(double)),"full-size projected vertex allocation");
    cu(cuMemAlloc(&faces,triangles.size()*sizeof(std::uint32_t)),"full-size immutable face allocation");
    for(std::size_t i=0;i<buffers.size();++i) cu(cuMemAlloc(&buffers[i],sizes[i]),"bounded research geometry/tile allocation");
    cu(cuMemcpyHtoD(faces,triangles.data(),triangles.size()*sizeof(std::uint32_t)),"initial full-size face upload");
    cu(cuMemcpyHtoD(buffers[1],packed.data(),sizeof(packed)),"explicit camera upload");
    cu(cuMemcpyHtoD(buffers[2],materials.data(),materials.size()),"explicit material identity upload");
    for(auto& event:geometry) cu(cuEventCreate(&event,CU_EVENT_DEFAULT),"geometry timing event");
    for(auto& event:sources) cu(cuEventCreate(&event,CU_EVENT_DISABLE_TIMING),"geometry producer event");
    cu(cuCtxSynchronize(),"full-size immutable inputs ready");
  } catch(...) {
    if(vertices)cuMemFree(vertices);
    if(faces)cuMemFree(faces);
    for(auto pointer:buffers)if(pointer)cuMemFree(pointer);
    for(auto event:geometry)if(event)cuEventDestroy(event);
    for(auto event:sources)if(event)cuEventDestroy(event);
    throw;
  }
  s.mesh_vertices=vertices;s.mesh_faces=faces;s.mesh_face_count=triangles.size()/3;
  s.large_vertex_count=vertex_count;s.tile_columns=columns;s.tile_rows=rows;
  s.large_buffers=buffers;s.geometry_events=geometry;s.source_events=sources;s.large_ready=false;
}

MeshUpdate GpuEncoder::State::finish_geometry() {
  cu(cuMemsetD32Async(large_buffers[4],0,tile_columns*tile_rows,render_stream),"clear bounded tile counters");
  cu(cuMemsetD32Async(large_buffers[6],0,5,render_stream),"clear geometry status");
  void* project_args[]={&large_buffers[0],&mesh_vertices,&large_vertex_count,&large_buffers[1],&width,&height,&large_buffers[6]};
  cu(cuLaunchKernel(project_large,(large_vertex_count+255)/256,1,1,256,1,1,0,render_stream,project_args,nullptr),"project exact full-size vertex set");
  void* bin_args[]={&mesh_vertices,&mesh_faces,&mesh_face_count,&large_buffers[1],&large_buffers[2],&large_buffers[3],
    &width,&height,&tile_columns,&tile_rows,&tile_edge,&large_buffers[4],&large_buffers[7],&large_buffers[6]};
  cu(cuLaunchKernel(bin_large,(mesh_face_count+255)/256,1,1,256,1,1,0,render_stream,bin_args,nullptr),"bin full-size triangles with overflow fencing");
  int tiles=tile_columns*tile_rows;
  void* scan_args[]={&large_buffers[4],&large_buffers[8],&large_buffers[9],&tiles,&reference_capacity,&tile_capacity,&large_buffers[6]};
  cu(cuLaunchKernel(scan_large,1,1,1,1,1,1,0,render_stream,scan_args,nullptr),"bounded compact tile prefix scan");
  void* fill_args[]={&large_buffers[7],&mesh_face_count,&tile_columns,&tile_edge,&large_buffers[8],&large_buffers[9],
    &large_buffers[5],&reference_capacity,&large_buffers[6]};
  cu(cuLaunchKernel(fill_large,(mesh_face_count+255)/256,1,1,256,1,1,0,render_stream,fill_args,nullptr),"fill exact compact tile references");
  cu(cuEventRecord(geometry_events[1],render_stream),"geometry update complete");
  cu(cuEventSynchronize(geometry_events[1]),"source geometry released only after owned copy/projection/binning");
  std::array<std::uint32_t,5> status{};
  cu(cuMemcpyDtoH(status.data(),large_buffers[6],sizeof(status)),"explicit twenty-byte geometry status readback");
  if(status[0]&1) throw std::runtime_error("invalid full-size geometry projection; no coordinate repair");
  if(status[2] || status[4]) throw std::runtime_error("full-size tile/reference capacity exceeded; no triangle deletion (maximum="+
    std::to_string(status[1])+", references="+std::to_string(status[3])+", scan="+std::to_string(status[2])+", fill="+std::to_string(status[4])+")");
  MeshUpdate result;result.maximum_tile_faces=status[1];result.tile_references=status[3];
  cu(cuEventElapsedTime(&result.gpu_ms,geometry_events[0],geometry_events[1]),"geometry GPU service time");
  large_ready=true;return result;
}

MeshUpdate GpuEncoder::update_mesh_host(const std::vector<float>& xyz) {
  auto& s=*state_;s.thread();
  if(!s.large_vertex_count || xyz.size()!=static_cast<std::size_t>(s.large_vertex_count)*3) throw std::invalid_argument("recorded geometry count differs from immutable topology");
  s.large_ready=false;
  cu(cuMemcpyHtoD(s.large_buffers[0],xyz.data(),xyz.size()*sizeof(float)),"explicit recorded-geometry reference upload");
  // The pageable-host reference copy may stage through the default stream;
  // make its completion explicit before using a nonblocking render stream.
  cu(cuCtxSynchronize(),"recorded host geometry ready for independent render stream");
  cu(cuEventRecord(s.geometry_events[0],s.render_stream),"recorded geometry projection start");
  return s.finish_geometry();
}

MeshUpdate GpuEncoder::update_mesh_device(CUdeviceptr xyz,std::size_t count,CUstream producer) {
  return update_mesh_device(std::vector<DeviceMeshPart>{{xyz,count,producer}});
}
MeshUpdate GpuEncoder::update_mesh_device(const std::vector<DeviceMeshPart>& parts) {
  auto& s=*state_;s.thread();
  if(!s.large_vertex_count || parts.empty() || parts.size()>s.source_events.size()) throw std::invalid_argument("bounded device geometry parts required");
  std::size_t count=0;
  for(const auto& part:parts) {
    if(!part.vertex_count || part.vertex_count>static_cast<std::size_t>(s.large_vertex_count)-count) throw std::invalid_argument("device geometry part count mismatch");
    CUcontext owner=nullptr;cu(cuPointerGetAttribute(&owner,CU_POINTER_ATTRIBUTE_CONTEXT,part.xyz),"geometry producer pointer context");
    if(owner!=s.context) throw std::runtime_error("geometry producer/context mismatch; no cross-context fallback");
    CUdeviceptr base=0;std::size_t bytes=0;cu(cuMemGetAddressRange(&base,&bytes,part.xyz),"geometry allocation extent");
    const auto offset=part.xyz-base;
    if(offset>bytes || part.vertex_count*3*sizeof(float)>bytes-offset) throw std::invalid_argument("device geometry exceeds producer allocation");
    count+=part.vertex_count;
  }
  if(count!=static_cast<std::size_t>(s.large_vertex_count)) throw std::invalid_argument("device geometry cannot be padded/truncated");
  s.large_ready=false;
  for(std::size_t i=0;i<parts.size();++i) {
    cu(cuEventRecord(s.source_events[i],parts[i].producer),"external geometry producer completion");
    cu(cuStreamWaitEvent(s.render_stream,s.source_events[i],0),"owned geometry waits for SDK producer");
  }
  cu(cuEventRecord(s.geometry_events[0],s.render_stream),"device geometry copy/projection start");
  std::size_t offset=0;
  for(const auto& part:parts) {
    const auto bytes=part.vertex_count*3*sizeof(float);
    cu(cuMemcpyDtoDAsync(s.large_buffers[0]+offset,part.xyz,bytes,s.render_stream),"copy producer geometry into owned vertex storage");offset+=bytes;
  }
  return s.finish_geometry();
}

std::vector<std::uint8_t> GpuEncoder::qualification_readback_rgba() {
  auto& s=*state_;s.thread();
  if(s.previous_pts<0) throw std::runtime_error("no completed frame to qualify");
  std::vector<std::uint8_t> result(s.width*s.height*4);
  CUDA_MEMCPY2D copy{};copy.srcMemoryType=CU_MEMORYTYPE_DEVICE;copy.srcDevice=s.rgba;copy.srcPitch=s.rgb_pitch;
  copy.dstMemoryType=CU_MEMORYTYPE_HOST;copy.dstHost=result.data();copy.dstPitch=s.width*4;
  copy.WidthInBytes=s.width*4;copy.Height=s.height;
  cu(cuMemcpy2D(&copy),"explicit qualification-only RGBA readback");
  return result;
}

std::vector<double> GpuEncoder::qualification_projected_vertices() {
  auto& s=*state_;s.thread();
  if(!s.large_vertex_count) throw std::runtime_error("full-size geometry qualification requires a bound scene");
  cu(cuStreamSynchronize(s.render_stream),"qualification projection complete");
  std::vector<double> result(s.large_vertex_count*3);
  cu(cuMemcpyDtoH(result.data(),s.mesh_vertices,result.size()*sizeof(double)),"explicit qualification-only projected geometry readback");
  return result;
}

EncodedFrame GpuEncoder::encode(int pattern,std::int64_t pts,bool force) {
  auto& s=*state_;s.thread();
  if(s.large_vertex_count && !s.large_ready) throw std::runtime_error("full-size geometry has no validated update");
  if(pts<=s.previous_pts) throw std::invalid_argument("nonmonotonic video PTS");
  s.previous_pts=pts;
  const auto ready_begin=Clock::now();
  const int index=s.next_slot;s.next_slot=(s.next_slot+1)%3;
  AVFrame* frame=s.slots[index];
  if(!av_frame_is_writable(frame)) throw std::runtime_error("encoder still retains a slot; refusing overwrite/allocation");
  cu(cuEventRecord(s.events[0],s.render_stream),"render start");
  void* render_args[]={&s.rgba,&s.rgb_pitch,&s.width,&s.height,&pattern};
  void* mesh_args[]={&s.rgba,&s.rgb_pitch,&s.width,&s.height,&s.mesh_vertices,&s.mesh_faces,&s.mesh_bounds,&s.mesh_face_count};
  void* large_args[]={&s.rgba,&s.rgb_pitch,&s.width,&s.height,&s.mesh_vertices,&s.mesh_faces,&s.large_buffers[3],
    &s.large_buffers[4],&s.large_buffers[8],&s.large_buffers[5],&s.tile_columns,&s.tile_edge};
  cu(cuLaunchKernel(s.large_vertex_count?s.render_large:(s.mesh_face_count?s.render_mesh:s.render),
    (s.width+15)/16,(s.height+15)/16,1,16,16,1,0,s.render_stream,
    s.large_vertex_count?large_args:(s.mesh_face_count?mesh_args:render_args),nullptr),"render owned surface");
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
