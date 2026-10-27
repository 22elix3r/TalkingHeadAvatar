// One research epoch/process. GPU SDK+geometry+NVENC share an owner; the CPU
// audio clock does not wait for a render. SIGTERM bypasses producer backpressure.
#include "../motion/claire_bridge.h"
#include <livekit/livekit.h>
#include <livekit/room.h>
#include <livekit/local_participant.h>
#include <livekit/local_audio_track.h>
#include <livekit/local_video_track.h>
#include <livekit/local_track_publication.h>
#include <livekit/encoded_video_source.h>
#include <livekit/audio_source.h>
#include <livekit/audio_frame.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <deque>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>

namespace {
using Clock=std::chrono::steady_clock;
constexpr auto unset=std::numeric_limits<uint64_t>::max();
std::atomic<bool> stopped{false};static_assert(std::atomic<bool>::is_always_lock_free);
int protocol=-1;
void stop(int){stopped.store(true,std::memory_order_relaxed);}
void require(bool value,const char* reason){if(!value)throw std::runtime_error(reason);}
void event(const std::string& value){const auto line=value+'\n';size_t offset=0;while(offset<line.size()){const auto count=write(protocol,line.data()+offset,line.size()-offset);require(count>0,"parent event pipe failed");offset+=count;}}
bool identifier(const std::string& value){return !value.empty() && value.size()<=100 && std::all_of(value.begin(),value.end(),[](unsigned char c){return std::isalnum(c)||c=='-'||c=='_';});}
std::string line(std::istream& input,size_t bound){std::string value;std::getline(input,value);require(bool(input)&&value.size()<=bound,"bounded bootstrap required");return value;}
uint64_t number(const std::string& value){require(!value.empty()&&value.size()<=19&&std::all_of(value.begin(),value.end(),[](unsigned char c){return c>='0'&&c<='9';}),"bounded bootstrap integer required");return std::stoull(value);}
uint64_t u64(const unsigned char* bytes){uint64_t value=0;for(int i=7;i>=0;--i)value=(value<<8)|bytes[i];return value;}
uint32_t u32(const unsigned char* bytes){uint32_t value=0;for(int i=3;i>=0;--i)value=(value<<8)|bytes[i];return value;}
bool exact(unsigned char* bytes,size_t count){size_t offset=0;while(offset<count&&!stopped.load()){pollfd fd{STDIN_FILENO,POLLIN,0};const auto ready=poll(&fd,1,50);if(ready<0){if(stopped.load())return false;throw std::runtime_error("input poll failed");}if(!ready)continue;const auto n=read(STDIN_FILENO,bytes+offset,count-offset);if(n<=0)return false;offset+=n;}return offset==count;}
struct Motion {std::array<float,320> samples{};size_t count=0;uint64_t start=0;bool eof=false;};
struct State {
  std::mutex mutex;std::condition_variable space;
  std::array<int16_t,48000> audio{};size_t head=0,tail=0,count=0,peak=0,motion_peak=0;
  std::deque<Motion> motions;uint64_t received=0;bool eof=false;std::string failure;
  std::atomic<uint64_t> base_us{0},first{unset},source_samples{0},stream_samples{0},pose_end{0};
  std::atomic<uint64_t> video_frames{0},source_frames{0},idle_frames{0},preroll_frames{0},video_skipped{0},pose_peak{0},key_requests{0},rate_updates{0},eof_padding{0};std::atomic<bool> video_eof{false};
  void fail(const char* reason){std::lock_guard<std::mutex> lock(mutex);if(failure.empty())failure=reason;stopped.store(true);space.notify_all();}
};
void receive(State& state,uint64_t epoch){
  try {
    uint64_t sequence=0;
    while(!stopped.load()){
      std::array<unsigned char,33> header{};if(!exact(header.data(),header.size())){stopped.store(true);break;}
      const auto audio=u32(header.data()+25),motion=u32(header.data()+29);const bool eof=header[0]=='F';
      require((header[0]=='D'||eof)&&u64(header.data()+1)==epoch&&u64(header.data()+9)==sequence++&&
        (eof?(audio<960 && motion==(audio+2)/3):(audio==960 && motion==320)),"invalid Claire epoch/sequence/paired quantum");
      std::vector<unsigned char> bytes(audio*2+motion*4);if(!exact(bytes.data(),bytes.size())){stopped.store(true);break;}
      Motion chunk;chunk.count=motion;chunk.start=u64(header.data()+17)/3;chunk.eof=eof;
      for(size_t i=0;i<motion;++i){const auto bits=u32(bytes.data()+audio*2+i*4);std::memcpy(&chunk.samples[i],&bits,4);require(std::isfinite(chunk.samples[i])&&std::abs(chunk.samples[i])<=4,"finite bounded paired motion required");}
      std::unique_lock<std::mutex> lock(state.mutex);
      require(!state.eof&&u64(header.data()+17)==state.received&&state.received+audio<=48000*120ULL&&(!eof||state.received+audio>0),"Claire audio gap/overlap/EOF/turn bound");
      while((state.count+audio>state.audio.size()||state.motions.size()>=50)&&!stopped.load())state.space.wait_for(lock,std::chrono::milliseconds(10));
      if(stopped.load())break;
      for(size_t i=0;i<audio;++i){state.audio[state.tail]=static_cast<int16_t>(static_cast<uint16_t>(bytes[i*2])|(static_cast<uint16_t>(bytes[i*2+1])<<8));state.tail=(state.tail+1)%state.audio.size();}
      state.count+=audio;state.received+=audio;state.peak=std::max(state.peak,state.count);state.eof=eof;
      state.motions.push_back(chunk);state.motion_peak=std::max(state.motion_peak,state.motions.size());
    }
  }catch(...){state.fail("claire_input_protocol_failure");}
}
struct ClaireOwner {
  tha_claire* handle=nullptr;char error[512]{};
  ClaireOwner(const char* ptx,const char* scene,const char* model){check(tha_claire_open(ptx,scene,model,&handle,error,sizeof(error)));}
  ~ClaireOwner(){if(handle)tha_claire_close(handle,error,sizeof(error));}
  void check(int code){if(code<0)throw std::runtime_error(error);}
};
void render(State& state,const char* ptx,const char* scene,const char* model,const std::shared_ptr<livekit::EncodedVideoSource>& source,
            std::promise<void> ready,uint64_t epoch,const std::string& session,const std::string& turn){
  bool announced=false;
  try {
    ClaireOwner owner(ptx,scene,model);ready.set_value();announced=true;
    uint64_t offered=0,next=0,preroll_next=0,last_key=0;int frame_id=0;bool motion_eof=false,pending_key=false;double encoder_fps=30;
    while(!stopped.load()){
      Motion chunk;bool feed=false,input_eof=false;uint64_t total=0;
      {
        std::lock_guard<std::mutex> lock(state.mutex);input_eof=state.eof;total=state.received;
        // Limit source lead to 800 ms, including Claire's 260 ms lookahead.
        // The remaining host queues backpressure Qwen rather than drop words.
        if(!state.motions.empty()&&offered*3<state.source_samples.load()+38400){chunk=state.motions.front();state.motions.pop_front();feed=true;state.space.notify_all();}
      }
      if(feed){
        owner.check(tha_claire_push(owner.handle,chunk.samples.data(),chunk.count,chunk.start,chunk.eof?1:0,owner.error,sizeof(owner.error)));
        offered+=chunk.count;motion_eof=chunk.eof;
        if(motion_eof)owner.check(tha_claire_audio_end(owner.handle,total,owner.error,sizeof(owner.error)));
        tha_claire_metrics metrics{};owner.check(tha_claire_get_metrics(owner.handle,&metrics,owner.error,sizeof(owner.error)));
        state.pose_end.store(((metrics.neural_frames+1)/2-metrics.eof_padding_frames)*1600);
        state.pose_peak.store(metrics.queue_peak_frames);state.eof_padding.store(metrics.eof_padding_frames);
      }
      uint64_t first=0,cursor=0,stream_sample=0;
      {std::lock_guard<std::mutex> lock(state.mutex);first=state.first.load();cursor=state.source_samples.load();stream_sample=state.stream_samples.load();}
      const bool preroll=first==unset;bool idle=false;
      if(!state.base_us.load()){std::this_thread::sleep_for(std::chrono::milliseconds(1));continue;}
      auto target=preroll?(stream_sample?stream_sample-1:0)/1600*1600:
        (input_eof&&total?std::min(cursor,total-1):cursor)/1600*1600;
      if(preroll&&target<preroll_next){std::this_thread::sleep_for(std::chrono::milliseconds(1));continue;}
      if(!preroll&&target<next){
        if(motion_eof&&input_eof&&next>=((total+1599)/1600)*1600){
          state.video_eof.store(true);const auto stream=state.stream_samples.load();
          target=(stream>first?stream-first:0)/1600*1600;idle=target>=next;
        }
        if(!idle){std::this_thread::sleep_for(std::chrono::milliseconds(1));continue;}
      }
      auto feedback=source->takeFeedback();bool rate_changed=false;
      if(feedback.keyframe_requested)state.key_requests.fetch_add(1);
      pending_key=pending_key||feedback.keyframe_requested;
      if(feedback.rate_control&&feedback.rate_control->target_bitrate_bps>0){
        state.rate_updates.fetch_add(1);
        const auto bitrate=static_cast<int>(std::clamp<uint64_t>(feedback.rate_control->target_bitrate_bps,100000,8000000));
        const auto fps=feedback.rate_control->framerate_fps>0?std::clamp(feedback.rate_control->framerate_fps,1.,60.):encoder_fps;
        owner.check(tha_claire_rate(owner.handle,bitrate,fps,owner.error,sizeof(owner.error)));rate_changed=std::abs(fps-encoder_fps)>.01;encoder_fps=fps;
      }
      pending_key=pending_key||rate_changed;
      tha_claire_packet packet{};const auto pts=state.base_us.load()+((preroll?0:first)+target)*1000000/48000;
      const auto force=frame_id==0||(!preroll&&!idle&&state.source_frames.load()==0)||pending_key||(target>=last_key&&target-last_key>=48000);
      const auto result=preroll?tha_claire_preroll(owner.handle,pts,force,&packet,owner.error,sizeof(owner.error)):
        idle?tha_claire_hold(owner.handle,pts,force,&packet,owner.error,sizeof(owner.error)):
          tha_claire_render(owner.handle,target,pts,force,&packet,owner.error,sizeof(owner.error));
      owner.check(result);if(!result){std::this_thread::sleep_for(std::chrono::milliseconds(1));continue;}if(stopped.load())break;
      if(!preroll&&!idle&&target>next)state.video_skipped.fetch_add((target-next)/1600);
      if(idle)state.idle_frames.fetch_add(1);
      if(packet.keyframe){last_key=target;pending_key=false;}
      livekit::EncodedVideoSource::Frame frame;frame.data.assign(packet.data,packet.data+packet.size);frame.is_keyframe=packet.keyframe;frame.width=512;frame.height=512;frame.timestamp_us=packet.pts_us;
      livekit::VideoFrameMetadata metadata;metadata.user_timestamp_us=packet.pts_us;metadata.frame_id=frame_id++;
      const auto source_identity=preroll?",\"idle_before_audio\":true":",\"source_sample\":"+std::to_string(packet.source_sample_48k)+",\"neural_sample_16k\":"+std::to_string(packet.neural_sample_16k);
      const auto identity="{\"session_id\":\""+session+"\",\"turn_id\":\""+turn+"\",\"epoch\":"+std::to_string(epoch)+source_identity+",\"presentation_sample\":"+std::to_string(target)+",\"idle_after_eof\":"+(idle?"true":"false")+"}";
      metadata.user_data=std::vector<uint8_t>(identity.begin(),identity.end());frame.metadata=metadata;
      require(source->captureFrame(frame),"Claire encoded frame rejected");state.video_frames.fetch_add(1);
      if(preroll)state.preroll_frames.fetch_add(1);
      if(!idle&&!preroll)state.source_frames.fetch_add(1);
      if(preroll)preroll_next=target+1600;else next=target+1600;
    }
  }catch(const std::exception& error){std::cerr<<"Claire GPU owner: "<<error.what()<<'\n';if(!announced){try{ready.set_exception(std::current_exception());}catch(...){}}state.fail("claire_gpu_media_failure");}
}
}

int main(int argc,char** argv){
  bool initialized=false;
  try {
    require(argc==5,"usage: tha_claire_session_media private-bootstrap.txt kernels.ptx research-scene.bin model.json");
    struct stat info{};require(lstat(argv[1],&info)==0&&S_ISREG(info.st_mode)&&info.st_uid==getuid()&&!(info.st_mode&0077)&&info.st_size<16384,"private operator bootstrap required");
    std::ifstream bootstrap(argv[1]);require(line(bootstrap,64)=="THA_CLAIRE_MEDIA_2","Claire bootstrap version mismatch");
    const auto url=line(bootstrap,128),token=line(bootstrap,8192),session=line(bootstrap,100),turn=line(bootstrap,100);const auto epoch=number(line(bootstrap,32)),prebuffer=number(line(bootstrap,32));
    require(identifier(session)&&identifier(turn)&&token.size()>50&&url.rfind("ws://127.0.0.1:",0)==0&&number(url.substr(15))>=1024&&number(url.substr(15))<=65535&&prebuffer>=300&&prebuffer<=500&&prebuffer%10==0,"qualified Claire loopback bootstrap required");
    protocol=dup(STDOUT_FILENO);require(protocol>=0&&dup2(STDERR_FILENO,STDOUT_FILENO)>=0,"vendor stdout isolation failed");
    std::signal(SIGTERM,stop);std::signal(SIGINT,stop);std::signal(SIGPIPE,SIG_IGN);
    require(livekit::initialize(livekit::LogLevel::Warn),"SDK initialization failed");initialized=true;
    {
      livekit::Room room;livekit::RoomOptions connection;connection.auto_subscribe=false;require(room.connect(url,token,connection),"local SFU connection failed");
      auto participant=room.localParticipant().lock();require(bool(participant),"publisher missing");
      auto audio=std::make_shared<livekit::AudioSource>(48000,1,0);auto video=std::make_shared<livekit::EncodedVideoSource>(livekit::VideoCodec::H264,512,512);
      auto vt=livekit::LocalVideoTrack::createLocalVideoTrack("claire-vendor-research-"+std::to_string(epoch),video);
      livekit::TrackPublishOptions options;options.video_codec=livekit::VideoCodec::H264;options.video_encoder=livekit::VideoEncoderBackend::PreEncoded;options.simulcast=false;options.source=livekit::TrackSource::SOURCE_CAMERA;options.stream="claire-"+std::to_string(epoch);
      livekit::VideoEncodingOptions encoding;encoding.max_bitrate=1200000;encoding.max_framerate=30;options.video_encoding=encoding;
      livekit::FrameMetadataFeatures features;features.user_timestamp=true;features.frame_id=true;features.user_data=true;options.frame_metadata_features=features;
      participant->publishTrack(vt,options);auto at=participant->publishAudioTrack("claire-answer-"+std::to_string(epoch),audio,livekit::TrackSource::SOURCE_MICROPHONE);
      require(vt->publication()&&at->publication(),"publication binding missing");const auto audio_sid=at->publication()->sid(),video_sid=vt->publication()->sid();require(identifier(audio_sid)&&identifier(video_sid),"invalid publication identity");
      State state;std::promise<void> gpu_ready;auto available=gpu_ready.get_future();
      std::thread gpu(render,std::ref(state),argv[2],argv[3],argv[4],video,std::move(gpu_ready),epoch,std::cref(session),std::cref(turn));std::thread input;
      try {
        available.get();input=std::thread(receive,std::ref(state),epoch);const auto origin=Clock::now();state.base_us.store(std::chrono::duration_cast<std::chrono::microseconds>(origin.time_since_epoch()).count());
        event("{\"kind\":\"ready\",\"session_id\":\""+session+"\",\"turn_id\":\""+turn+"\",\"epoch\":"+std::to_string(epoch)+",\"audio_sid\":\""+audio_sid+"\",\"video_sid\":\""+video_sid+"\",\"publisher\":\"media-"+session+"\",\"clock_id\":\"cpp_monotonic\"}");
        bool playing=false,done=false,acknowledged=false;int flush=0;uint64_t valid=0,padding=0,first=0;double maximum_late=0;
        for(uint64_t tick=0;!stopped.load();++tick){
          const auto due=origin+std::chrono::microseconds(tick*10000);std::this_thread::sleep_until(due);maximum_late=std::max(maximum_late,std::chrono::duration<double,std::milli>(Clock::now()-due).count());
          std::vector<int16_t> pcm(480,0);
          {
            std::lock_guard<std::mutex> lock(state.mutex);
            if(!playing&&!done&&tick>=10&&state.preroll_frames.load()>=3&&(state.count>=prebuffer*48||state.eof)&&
              state.pose_end.load()>0&&state.pose_end.load()>=std::min<uint64_t>(prebuffer*48,state.received)) {
              playing=true;first=tick*480;state.first.store(first);
            }
            if(playing&&!done){
              if(flush){++flush;padding+=480;if(flush==3)done=true;}
              else {
                require(state.count>=480||state.eof,"active Claire audio underrun; retire epoch");require(state.pose_end.load()>valid,"Claire motion underrun; retire epoch");
                const auto count=std::min<size_t>(480,state.count);for(size_t i=0;i<count;++i){pcm[i]=state.audio[state.head];state.head=(state.head+1)%state.audio.size();}
                state.count-=count;valid+=count;padding+=480-count;state.source_samples.store(valid);
                if(state.eof&&state.count==0)flush=count==0?2:1;
                state.space.notify_all();
              }
            }
          }
          if(stopped.load())break;
          audio->captureFrame(livekit::AudioFrame(std::move(pcm),48000,1,480),0);
          state.stream_samples.store((tick+1)*480);
          if(done&&!acknowledged&&state.video_eof.load()){
            acknowledged=true;event("{\"kind\":\"eof\",\"session_id\":\""+session+"\",\"turn_id\":\""+turn+"\",\"epoch\":"+std::to_string(epoch)+",\"valid_samples\":"+std::to_string(valid)+",\"padding_samples\":"+std::to_string(padding)+",\"first_pcm_stream_sample\":"+std::to_string(first)+",\"source_clock_late_max_ms\":"+std::to_string(maximum_late)+",\"queue_peak_samples\":"+std::to_string(state.peak)+",\"motion_queue_peak_packets\":"+std::to_string(state.motion_peak)+",\"pose_queue_peak_frames\":"+std::to_string(state.pose_peak.load())+",\"keyframe_requests\":"+std::to_string(state.key_requests.load())+",\"rate_updates\":"+std::to_string(state.rate_updates.load())+",\"video_frames\":"+std::to_string(state.video_frames.load())+",\"source_video_frames\":"+std::to_string(state.source_frames.load())+",\"idle_video_frames\":"+std::to_string(state.idle_frames.load())+",\"preroll_video_frames\":"+std::to_string(state.preroll_frames.load())+",\"motion_eof_padding_frames\":"+std::to_string(state.eof_padding.load())+",\"video_skipped\":"+std::to_string(state.video_skipped.load())+",\"presentation_ack\":false}");
          }
          require(tick<360000,"bounded Claire epoch duration exceeded");
        }
      }catch(...){stopped.store(true);state.space.notify_all();if(input.joinable())input.join();gpu.join();throw;}
      stopped.store(true);state.space.notify_all();if(input.joinable())input.join();gpu.join();if(!state.failure.empty())throw std::runtime_error(state.failure);
      participant->unpublishTrack(audio_sid);participant->unpublishTrack(video_sid);audio->clearQueue();room.disconnect();
    }
    livekit::shutdown();close(protocol);return 0;
  }catch(const std::exception& error){if(protocol>=0){try{event("{\"kind\":\"error\",\"code\":\"claire_media_failed\"}");}catch(...){}}std::cerr<<"Claire session: "<<error.what()<<'\n';if(initialized)livekit::shutdown();return 1;}
}
