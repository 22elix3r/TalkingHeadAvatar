// Compiled with the normal C++ ABI: exercises the separately linked ABI=0 SDK.
#include "claire_bridge.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
static_assert(_GLIBCXX_USE_CXX11_ABI==1,"qualify the actual LiveKit-side C++ ABI");
namespace {
void require(bool value,const std::string& reason){if(!value)throw std::runtime_error(reason);}
char error[512]{};
void ok(int code){require(code==0,error);}
struct Owner {
  tha_claire* handle=nullptr;
  Owner(const char* ptx,const char* scene,const char* model){ok(tha_claire_open(ptx,scene,model,&handle,error,sizeof(error)));}
  ~Owner(){if(handle)tha_claire_close(handle,error,sizeof(error));}
};
void write_all(int fd,const uint8_t* data,size_t count) {
  while(count){const auto size=write(fd,data,count);require(size>0,"private coded write failed");data+=size;count-=size;}
}
}
int main(int argc,char** argv) {
  int output=-1;
  try {
    require(argc==7,"usage: tha_claire_bridge_probe ptx scene model input.f32 new-output.h264 normal|late|cancel|overflow|invalid|ordering|paced|irregular|short");
    const std::string mode=argv[6];require(mode=="normal" || mode=="late" || mode=="cancel" || mode=="overflow" || mode=="invalid" || mode=="ordering" || mode=="paced" || mode=="irregular" || mode=="short","unknown bridge case");
    struct stat info{};require(lstat(argv[4],&info)==0 && S_ISREG(info.st_mode) && info.st_size>16000*4 && info.st_size<30*16000*4 && info.st_size%4==0,"bounded fixture PCM required");
    std::vector<float> pcm(info.st_size/4);std::ifstream input(argv[4],std::ios::binary);input.read(reinterpret_cast<char*>(pcm.data()),info.st_size);require(bool(input),"truncated fixture");
    if(mode=="short")pcm.resize(177); // Shorter than one pose and the lookahead.
    if(mode=="cancel") {
      Owner cancelled(argv[1],argv[2],argv[3]);
      for(size_t offset=0;offset<16000;offset+=320)ok(tha_claire_push(cancelled.handle,pcm.data()+offset,320,offset,0,error,sizeof(error)));
      tha_claire_metrics metrics{};ok(tha_claire_get_metrics(cancelled.handle,&metrics,error,sizeof(error)));
      require(metrics.neural_frames>0 && metrics.queued_frames>0 && !metrics.eof,"cancel must retire actual pending GPU poses");
    }
    if(mode=="ordering") {
      for(const auto fault:{"gap","quantum","empty_eof","grid","timebase","duplicate"}) {
        Owner failed(argv[1],argv[2],argv[3]);int result=0;tha_claire_packet packet{};
        if(std::string(fault)=="gap")result=tha_claire_push(failed.handle,pcm.data(),320,1,0,error,sizeof(error));
        else if(std::string(fault)=="quantum")result=tha_claire_push(failed.handle,pcm.data(),321,0,0,error,sizeof(error));
        else if(std::string(fault)=="empty_eof")result=tha_claire_push(failed.handle,nullptr,0,0,1,error,sizeof(error));
        else if(std::string(fault)=="grid")result=tha_claire_render(failed.handle,1,1000000,1,&packet,error,sizeof(error));
        else {
          for(size_t offset=0;offset<4480;offset+=320)ok(tha_claire_push(failed.handle,pcm.data()+offset,320,offset,0,error,sizeof(error)));
          require(tha_claire_render(failed.handle,0,1000000,1,&packet,error,sizeof(error))==1,"real pose required before clock fault");
          result=std::string(fault)=="timebase"?tha_claire_render(failed.handle,1600,2000000,0,&packet,error,sizeof(error)):
            tha_claire_render(failed.handle,0,1000000,0,&packet,error,sizeof(error));
        }
        require(result==-1,std::string("ordering fault accepted: ")+fault);
        require(tha_claire_push(failed.handle,pcm.data(),320,0,0,error,sizeof(error))==-1,"failed ordering epoch resumed");
      }
      std::cout<<"{\"case\":\"ordering\",\"retired_failed_epochs\":6,\"gap_quantum_empty_eof_grid_timebase_duplicate_rejected\":true}\n";return 0;
    }
    Owner state(argv[1],argv[2],argv[3]);
    if(mode=="invalid") {
      std::string foreign_error;std::thread other([&]{tha_claire_metrics m{};char reason[512]{};require(tha_claire_get_metrics(state.handle,&m,reason,sizeof(reason))==-1,"foreign thread accepted");foreign_error=reason;});other.join();
      require(foreign_error.find("owner thread")!=std::string::npos,"wrong-thread reason changed");
      float invalid=std::nanf("");require(tha_claire_push(state.handle,&invalid,1,0,0,error,sizeof(error))==-1,"nonfinite PCM accepted");
      require(tha_claire_push(state.handle,pcm.data(),320,0,0,error,sizeof(error))==-1,"failed epoch resumed");
      std::cout<<"{\"case\":\"invalid\",\"wrong_thread_rejected\":true,\"nonfinite_rejected\":true,\"failed_epoch_rejected\":true}\n";return 0;
    }
    if(mode=="overflow") {
      bool failed=false;
      for(size_t offset=0;offset<pcm.size();offset+=320) {
        const auto count=std::min<size_t>(320,pcm.size()-offset);
        if(tha_claire_push(state.handle,pcm.data()+offset,count,offset,0,error,sizeof(error))<0){failed=true;break;}
      }
      require(failed && std::string(error).find("queue bound")!=std::string::npos,"motion overflow did not fail explicitly");
      tha_claire_metrics metrics{};ok(tha_claire_get_metrics(state.handle,&metrics,error,sizeof(error)));
      require(metrics.queue_peak_frames==32 && metrics.queued_frames==32 && metrics.discarded_video_frames==0,"overflow overwrote GPU poses");
      std::cout<<"{\"case\":\"overflow\",\"queue_peak_frames\":32,\"silent_overwrite\":false}\n";return 0;
    }
    output=open(argv[5],O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);require(output>=0,"new private coded output required");
    uint64_t target=mode=="late"?16000:0,packets=0;bool eof=false;
    // No geometry is available before the declared lookahead. Future requests
    // must not consume data or mutate presentation order.
    tha_claire_packet packet{};require(tha_claire_render(state.handle,0,1000000,1,&packet,error,sizeof(error))==0,"uninitialized pose rendered");
    auto began=std::chrono::steady_clock::now(),presentation=began;bool playing=false;double maximum_late_ms=0;
    size_t step=0;
    for(size_t offset=0;offset<pcm.size();) {
      if(mode=="paced")std::this_thread::sleep_until(began+std::chrono::microseconds(offset*1000000/16000));
      constexpr size_t irregular[]={17,320,1,319,73,211,2};
      const auto quantum=mode=="irregular"?irregular[step++%7]:320;
      const auto count=std::min<size_t>(quantum,pcm.size()-offset);eof=offset+count==pcm.size();
      ok(tha_claire_push(state.handle,pcm.data()+offset,count,offset,eof?1:0,error,sizeof(error)));
      offset+=count;
      tha_claire_metrics before{},after{};ok(tha_claire_get_metrics(state.handle,&before,error,sizeof(error)));
      require(tha_claire_render(state.handle,48000*100,101000000,0,&packet,error,sizeof(error))==0,"future geometry rendered");
      ok(tha_claire_get_metrics(state.handle,&after,error,sizeof(error)));require(before.queued_frames==after.queued_frames,"future request consumed queued poses");
      while(true) {
        if(mode=="paced" && playing && !eof && std::chrono::steady_clock::now()<presentation+std::chrono::microseconds(target*1000000/48000))break;
        if(mode=="paced" && playing && eof)std::this_thread::sleep_until(presentation+std::chrono::microseconds(target*1000000/48000));
        const auto result=tha_claire_render(state.handle,target,1000000+target*1000000/48000,packets==0 || packets==31,&packet,error,sizeof(error));
        require(result>=0,error);if(!result)break;
        if(!playing){presentation=std::chrono::steady_clock::now();playing=true;}
        if(mode=="paced")maximum_late_ms=std::max(maximum_late_ms,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-presentation).count()-target*1000./48000);
        require(packet.source_sample_48k==target && target-packet.neural_sample_16k*3<=2 && packet.size>0,"audio/neural rational clock mapping failed");
        require((packets!=0 && packets!=31) || packet.keyframe,"keyframe demand lost across C ABI");
        write_all(output,packet.data,packet.size);++packets;target+=1600;
      }
    }
    require(eof,"missing fixture EOF");
    tha_claire_metrics metrics{};ok(tha_claire_get_metrics(state.handle,&metrics,error,sizeof(error)));
    require(metrics.offered_samples==pcm.size() && metrics.neural_frames==(pcm.size()*60+15999)/16000 && metrics.eof && metrics.queued_frames==0,"EOF neural frame/sample conservation failed");
    require(metrics.rendered_frames+metrics.discarded_video_frames==(metrics.neural_frames+1)/2,"presentation frame conservation failed");
    require((mode=="short"?metrics.first_result_after_samples==177:
      (mode=="irregular"?metrics.first_result_after_samples>=4160 && metrics.first_result_after_samples<4480:metrics.first_result_after_samples==4160)) &&
      metrics.queue_peak_frames<=32 && metrics.retained_audio_peak_samples<=24640,"lookahead/queue bound changed");
    require((mode=="late" && metrics.discarded_video_frames==10) || (mode!="late" && metrics.discarded_video_frames==0),"unexpected video discard");
    require(tha_claire_push(state.handle,pcm.data(),1,pcm.size(),0,error,sizeof(error))==-1,"post-EOF PCM accepted");
    require(fsync(output)==0,"coded persistence failed");close(output);output=-1;
    std::cout<<"{\"case\":\""<<mode<<"\",\"source_samples\":"<<metrics.offered_samples<<",\"neural_frames\":"<<metrics.neural_frames
      <<",\"encoded_frames\":"<<packets<<",\"discarded_video_frames\":"<<metrics.discarded_video_frames<<",\"queue_peak_frames\":"<<metrics.queue_peak_frames
      <<",\"first_result_after_samples\":"<<metrics.first_result_after_samples<<",\"retained_audio_peak_samples\":"<<metrics.retained_audio_peak_samples
      <<",\"maximum_schedule_lateness_ms\":"<<maximum_late_ms<<",\"caller_cxx_abi\":1,\"sdk_cxx_abi\":0,\"geometry_host_stage\":false,\"physical_playback\":false}\n";
    return 0;
  }catch(const std::exception& exception){if(output>=0)close(output);std::cerr<<"Claire bridge: "<<exception.what()<<'\n';return 1;}
}
