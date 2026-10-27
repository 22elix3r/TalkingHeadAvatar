// One epoch/process: priority SIGTERM cancellation, fresh track IDs on resume.
// Audio capture has its own CPU clock; GPU render/encode owns another thread.
#include "gpu_encoder.h"
#include "projected_file.h"
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
#include <future>
#include <iostream>
#include <mutex>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

namespace {
using Clock=std::chrono::steady_clock;
std::atomic<bool> stopped{false};
static_assert(std::atomic<bool>::is_always_lock_free);
void stop(int) {stopped.store(true,std::memory_order_relaxed);}
int protocol=-1;
void require(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
void event(const std::string& data) {
  const auto line=data+'\n';std::size_t offset=0;
  while(offset<line.size()) {const auto size=write(protocol,line.data()+offset,line.size()-offset);if(size<=0) throw std::runtime_error("parent event pipe failed");offset+=size;}
}
bool identifier(const std::string& value) {
  return !value.empty() && value.size()<=100 && std::all_of(value.begin(),value.end(),[](unsigned char c){return std::isalnum(c) || c=='-' || c=='_';});
}
std::string line(std::istream& input,std::size_t bound) {
  std::string value;std::getline(input,value);require(bool(input) && value.size()<=bound,"bounded bootstrap field required");return value;
}
std::uint64_t number(const std::string& value) {
  require(!value.empty() && value.size()<=19 && std::all_of(value.begin(),value.end(),[](unsigned char c){return c>='0' && c<='9';}),"bounded bootstrap integer required");
  return std::stoull(value);
}
std::uint64_t u64(const unsigned char* bytes) {std::uint64_t value=0;for(int i=7;i>=0;--i) value=(value<<8)|bytes[i];return value;}
std::uint32_t u32(const unsigned char* bytes) {std::uint32_t value=0;for(int i=3;i>=0;--i) value=(value<<8)|bytes[i];return value;}
bool exact(unsigned char* bytes,std::size_t count) {
  std::size_t offset=0;
  while(offset<count && !stopped.load()) {
    pollfd descriptor{STDIN_FILENO,POLLIN,0};const int ready=poll(&descriptor,1,50);
    if(ready<0) {if(stopped.load()) return false;throw std::runtime_error("PCM poll failed");}
    if(!ready) continue;
    const auto size=read(STDIN_FILENO,bytes+offset,count-offset);
    if(size<=0) return false;
    offset+=size;
  }
  return offset==count;
}
struct State {
  std::mutex mutex;std::condition_variable space;
  std::array<std::int16_t,48000> queue{};
  std::size_t head=0,tail=0,count=0,peak=0;
  std::uint64_t received=0;
  bool eof=false;std::string failure;
  std::atomic<std::uint64_t> samples{0},base_us{0};
  std::atomic<std::uint64_t> video_frames{0},video_skipped{0},key_requests{0},rate_updates{0};
  void fail(const char* reason) {std::lock_guard<std::mutex> lock(mutex);if(failure.empty()) failure=reason;stopped.store(true);space.notify_all();}
};
void receive(State& state,std::uint64_t epoch) {
  try {
    std::uint64_t sequence=0;
    while(!stopped.load()) {
      std::array<unsigned char,29> header{};
      if(!exact(header.data(),header.size())) {stopped.store(true);break;}
      const auto count=u32(header.data()+25);
      require((header[0]=='D' || header[0]=='F') && u64(header.data()+1)==epoch && u64(header.data()+9)==sequence++ && count<=4800 &&
        (header[0]!='D' || (count>0 && count%480==0)),"invalid PCM epoch/sequence/quantum");
      std::vector<unsigned char> bytes(count*2);
      if(!exact(bytes.data(),bytes.size())) {stopped.store(true);break;}
      std::unique_lock<std::mutex> lock(state.mutex);
      require(!state.eof && u64(header.data()+17)==state.received && state.received+count<=48000*120ULL,"PCM gap/overlap/EOF/turn bound");
      while(state.count+count>state.queue.size() && !stopped.load()) state.space.wait_for(lock,std::chrono::milliseconds(10));
      if(stopped.load()) break;
      for(std::uint32_t i=0;i<count;++i) {state.queue[state.tail]=static_cast<std::int16_t>(static_cast<std::uint16_t>(bytes[i*2]) | (static_cast<std::uint16_t>(bytes[i*2+1])<<8));state.tail=(state.tail+1)%state.queue.size();}
      state.count+=count;state.peak=std::max(state.peak,state.count);state.received+=count;
      if(header[0]=='F') state.eof=true;
    }
  } catch(...) {state.fail("pcm_protocol_failure");}
}

void render(State& state,const tha::ProjectedGeometry& geometry,const char* ptx,
            const std::shared_ptr<livekit::EncodedVideoSource>& source,std::promise<void> ready,
            std::uint64_t epoch,const std::string& session,const std::string& turn) {
  bool announced=false;
  try {
    tha::GpuEncoder encoder(ptx,512,512);encoder.mesh(geometry.vertices,geometry.faces);
    ready.set_value();announced=true;
    while(!state.base_us.load() && !stopped.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    std::uint64_t next=0,last_key=0;int frame_id=0;double encoder_fps=30;
    while(!stopped.load()) {
      const auto offered=state.samples.load();
      if(offered<next) {std::this_thread::sleep_for(std::chrono::milliseconds(1));continue;}
      const auto target=offered/1600*1600;
      if(target>next) state.video_skipped.fetch_add((target-next)/1600);
      const auto feedback=source->takeFeedback();bool rate_changed=false;
      if(feedback.keyframe_requested) state.key_requests.fetch_add(1);
      if(feedback.rate_control && feedback.rate_control->target_bitrate_bps>0) {
        encoder.bitrate(static_cast<int>(std::clamp<std::uint64_t>(feedback.rate_control->target_bitrate_bps,100000,8000000)));
        state.rate_updates.fetch_add(1);
        if(feedback.rate_control->framerate_fps>0) {
          const double selected=std::clamp(feedback.rate_control->framerate_fps,1.,60.);
          if(std::abs(selected-encoder_fps)>.01) {encoder_fps=selected;encoder.framerate(selected);rate_changed=true;}
        }
      }
      auto packet=encoder.encode(frame_id,state.base_us.load()+target*1000000/48000,
        frame_id==0 || feedback.keyframe_requested || rate_changed || target-last_key>=48000);
      if(stopped.load()) break;
      if(packet.keyframe) last_key=target;
      livekit::EncodedVideoSource::Frame frame;frame.data=std::move(packet.data);frame.is_keyframe=packet.keyframe;
      frame.width=512;frame.height=512;frame.timestamp_us=packet.pts_us;
      livekit::VideoFrameMetadata metadata;metadata.user_timestamp_us=packet.pts_us;metadata.frame_id=frame_id++;
      const auto identity="{\"session_id\":\""+session+"\",\"turn_id\":\""+turn+"\",\"epoch\":"+std::to_string(epoch)+",\"source_sample\":"+std::to_string(target)+"}";
      metadata.user_data=std::vector<std::uint8_t>(identity.begin(),identity.end());frame.metadata=metadata;
      require(source->captureFrame(frame),"encoded frame rejected");state.video_frames.fetch_add(1);next=target+1600;
    }
  } catch(...) {
    if(!announced) {try {ready.set_exception(std::current_exception());} catch(...) {}}
    state.fail("native_gpu_media_failure");
  }
}
}

int main(int argc,char** argv) {
  bool initialized=false;
  try {
    require(argc==4,"usage: tha_session_media private-bootstrap.txt kernels.ptx projected-mesh.bin");
    struct stat info{};require(lstat(argv[1],&info)==0 && S_ISREG(info.st_mode) && info.st_uid==getuid() && !(info.st_mode&0077) && info.st_size<16384,"private operator bootstrap required");
    std::ifstream bootstrap(argv[1]);require(line(bootstrap,64)=="THA_NATIVE_MEDIA_1","bootstrap version mismatch");
    const auto url=line(bootstrap,128),token=line(bootstrap,8192),session=line(bootstrap,100),turn=line(bootstrap,100);
    const auto epoch=number(line(bootstrap,32)),prebuffer=number(line(bootstrap,32));
    require(identifier(session) && identifier(turn) && token.size()>50 && url.rfind("ws://127.0.0.1:",0)==0 &&
      number(url.substr(15))>=1024 && number(url.substr(15))<=65535 && prebuffer>=10 && prebuffer<=500 && prebuffer%10==0,"qualified loopback bootstrap required");
    const auto geometry=tha::read_projected(argv[3]);
    protocol=dup(STDOUT_FILENO);require(protocol>=0,"protocol pipe unavailable");require(dup2(STDERR_FILENO,STDOUT_FILENO)>=0,"vendor stdout isolation failed");
    std::signal(SIGTERM,stop);std::signal(SIGINT,stop);std::signal(SIGPIPE,SIG_IGN);
    require(livekit::initialize(livekit::LogLevel::Warn),"SDK initialization failed");initialized=true;
    {
      livekit::Room room;livekit::RoomOptions connection;connection.auto_subscribe=false;
      require(room.connect(url,token,connection),"local SFU connection failed");
      auto participant=room.localParticipant().lock();require(bool(participant),"publisher missing");
      auto audio=std::make_shared<livekit::AudioSource>(48000,1,0);
      auto video=std::make_shared<livekit::EncodedVideoSource>(livekit::VideoCodec::H264,512,512);
      auto vt=livekit::LocalVideoTrack::createLocalVideoTrack("native-static-"+std::to_string(epoch),video);
      livekit::TrackPublishOptions options;options.video_codec=livekit::VideoCodec::H264;options.video_encoder=livekit::VideoEncoderBackend::PreEncoded;options.simulcast=false;
      options.source=livekit::TrackSource::SOURCE_CAMERA;options.stream="native-"+std::to_string(epoch);
      livekit::VideoEncodingOptions encoding;encoding.max_bitrate=1200000;encoding.max_framerate=30;options.video_encoding=encoding;
      livekit::FrameMetadataFeatures features;features.user_timestamp=true;features.frame_id=true;features.user_data=true;options.frame_metadata_features=features;
      participant->publishTrack(vt,options);
      auto at=participant->publishAudioTrack("native-answer-"+std::to_string(epoch),audio,livekit::TrackSource::SOURCE_MICROPHONE);
      require(vt->publication() && at->publication(),"publication binding missing");
      const auto audio_sid=at->publication()->sid(),video_sid=vt->publication()->sid();require(identifier(audio_sid) && identifier(video_sid),"invalid publication identity");
      State state;std::promise<void> gpu_ready;auto available=gpu_ready.get_future();
      std::thread gpu(render,std::ref(state),std::cref(geometry),argv[2],video,std::move(gpu_ready),epoch,std::cref(session),std::cref(turn));
      std::thread input;
      try {
        available.get();input=std::thread(receive,std::ref(state),epoch);
        const auto origin=Clock::now();const auto base=std::chrono::duration_cast<std::chrono::microseconds>(origin.time_since_epoch()).count();state.base_us.store(base);
        event("{\"kind\":\"ready\",\"session_id\":\""+session+"\",\"turn_id\":\""+turn+"\",\"epoch\":"+std::to_string(epoch)+",\"audio_sid\":\""+audio_sid+"\",\"video_sid\":\""+video_sid+"\",\"publisher\":\"media-"+session+"\",\"clock_id\":\"cpp_monotonic\"}");
        bool playing=false,done=false;int flush=0;std::uint64_t valid=0,padding=0,first=0;double maximum_late=0;
        for(std::uint64_t tick=0;!stopped.load();++tick) {
          const auto due=origin+std::chrono::microseconds(tick*10000);std::this_thread::sleep_until(due);
          maximum_late=std::max(maximum_late,std::chrono::duration<double,std::milli>(Clock::now()-due).count());
          std::vector<std::int16_t> pcm(480,0);bool acknowledge=false;
          {
            std::lock_guard<std::mutex> lock(state.mutex);
            if(!playing && !done && (state.count>=prebuffer*48 || state.eof)) {playing=true;first=tick*480;}
            if(playing && !done) {
              if(flush) {++flush;padding+=480;if(flush==3) {done=true;acknowledge=true;}}
              else {
                require(state.count>=480 || state.eof,"active audio underrun; no silent speech deletion");
                const auto count=std::min<std::size_t>(480,state.count);
                for(std::size_t i=0;i<count;++i) {pcm[i]=state.queue[state.head];state.head=(state.head+1)%state.queue.size();}
                state.count-=count;valid+=count;padding+=480-count;
                if(state.eof && state.count==0) flush=count==0 ? 2 : 1;
                state.space.notify_all();
              }
            }
          }
          if(stopped.load()) break;
          audio->captureFrame(livekit::AudioFrame(std::move(pcm),48000,1,480),0);state.samples.store((tick+1)*480);
          if(acknowledge) event("{\"kind\":\"eof\",\"session_id\":\""+session+"\",\"turn_id\":\""+turn+"\",\"epoch\":"+std::to_string(epoch)+",\"valid_samples\":"+std::to_string(valid)+",\"padding_samples\":"+std::to_string(padding)+",\"first_pcm_stream_sample\":"+std::to_string(first)+",\"source_clock_late_max_ms\":"+std::to_string(maximum_late)+",\"queue_peak_samples\":"+std::to_string(state.peak)+",\"video_frames\":"+std::to_string(state.video_frames.load())+",\"video_skipped\":"+std::to_string(state.video_skipped.load())+",\"presentation_ack\":false}");
          require(tick<360000,"bounded worker epoch duration exceeded");
        }
      } catch(...) {stopped.store(true);state.space.notify_all();if(input.joinable()) input.join();gpu.join();throw;}
      stopped.store(true);state.space.notify_all();if(input.joinable()) input.join();gpu.join();
      if(!state.failure.empty()) throw std::runtime_error(state.failure);
      participant->unpublishTrack(audio_sid);participant->unpublishTrack(video_sid);audio->clearQueue();room.disconnect();
    }
    livekit::shutdown();close(protocol);return 0;
  } catch(const std::exception& error) {
    if(protocol>=0) {try {event("{\"kind\":\"error\",\"code\":\"native_media_failed\"}");} catch(...) {}}
    std::cerr<<"native session media: "<<error.what()<<'\n';if(initialized) livekit::shutdown();return 1;
  }
}
