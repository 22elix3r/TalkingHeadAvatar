// Local qualification publisher. Its input is synthetic; this is not the product
// conversation adapter. Token material is read from a private file, never argv.
#include "gpu_encoder.h"
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
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <sys/stat.h>

namespace {
using Clock=std::chrono::steady_clock;
void event(const std::string& body) {std::cout<<body<<std::endl;}
}

int main(int argc,char** argv) {
  bool initialized=false;
  try {
    if(argc!=4) throw std::invalid_argument("usage: tha_media_publisher private-grant.txt kernels.ptx seconds");
    struct stat metadata{};
    if(lstat(argv[1],&metadata)!=0 || (metadata.st_mode&0077)!=0 || !S_ISREG(metadata.st_mode))
      throw std::runtime_error("grant must be a private regular file");
    std::ifstream config(argv[1]);std::string url,token;
    std::getline(config,url);std::getline(config,token);
    if(url!="ws://127.0.0.1:17880" || token.size()<50 || token.size()>8192) throw std::runtime_error("local qualification grant required");
    const int seconds=std::stoi(argv[3]);if(seconds<1 || seconds>30) throw std::invalid_argument("bounded probe duration is 1..30 seconds");
    tha::GpuEncoder encoder(argv[2]);
    if(!livekit::initialize(livekit::LogLevel::Warn)) throw std::runtime_error("LiveKit initialization failed");
    initialized=true;
    {
      livekit::Room room;livekit::RoomOptions connection;connection.auto_subscribe=false;
      if(!room.connect(url,token,connection)) throw std::runtime_error("LiveKit connection failed");
      auto participant=room.localParticipant().lock();if(!participant) throw std::runtime_error("publisher unavailable");
      auto video=std::make_shared<livekit::EncodedVideoSource>(livekit::VideoCodec::H264,512,512);
      auto audio=std::make_shared<livekit::AudioSource>(48000,1,0);
      auto vt=livekit::LocalVideoTrack::createLocalVideoTrack("native-synthetic-reference",video);
      livekit::TrackPublishOptions options;options.video_codec=livekit::VideoCodec::H264;
      options.video_encoder=livekit::VideoEncoderBackend::PreEncoded;options.simulcast=false;
      options.source=livekit::TrackSource::SOURCE_CAMERA;
      livekit::FrameMetadataFeatures features;features.user_timestamp=true;features.frame_id=true;
      options.frame_metadata_features=features;
      options.stream="native-reference";
      livekit::VideoEncodingOptions encoding;encoding.max_bitrate=1200000;encoding.max_framerate=30;
      options.video_encoding=encoding;
      participant->publishTrack(vt,options);
      auto at=participant->publishAudioTrack("native-clock-tone",audio,livekit::TrackSource::SOURCE_MICROPHONE);
      if(!vt->publication() || !at->publication()) throw std::runtime_error("missing published track binding");
      const auto video_sid=vt->publication()->sid(),audio_sid=at->publication()->sid();
      event("{\"kind\":\"ready\",\"video_sid\":\""+video_sid+"\",\"audio_sid\":\""+audio_sid+"\",\"codec\":\"H264\",\"input\":\"pre_encoded\"}");
      std::string command;
      if(!std::getline(std::cin,command) || command!="start") throw std::runtime_error("qualification start handshake required");
      const auto origin=Clock::now();
      const std::int64_t base=std::chrono::duration_cast<std::chrono::microseconds>(origin.time_since_epoch()).count();
      std::int64_t next_video=0,last_keyframe_sample=-48000;
      int frames=0,keyframes=0,rate_updates=0,feedback_keyframes=0,fps_updates=0;
      double render_ms=0,convert_ms=0,encode_ms=0,encoder_fps=30;
      // One second of explicitly labelled preparation initializes passthrough
      // and receiver state. It is excluded from valid response samples, never
      // from cold-start latency. Metadata identifies every preparation frame.
      constexpr std::int64_t preparation_samples=48000;
      const std::int64_t total_samples=(seconds+1)*48000LL;
      std::int64_t next_audio=0;
      double maximum_schedule_lateness_ms=0;
      while(next_audio<total_samples || next_video<total_samples) {
        const auto sample=std::min(next_audio,next_video);
        const auto due=origin+std::chrono::microseconds(sample*1000000/48000);
        std::this_thread::sleep_until(due);
        maximum_schedule_lateness_ms=std::max(maximum_schedule_lateness_ms,
          std::chrono::duration<double,std::milli>(Clock::now()-due).count());
        if(next_video==sample && next_video<total_samples) {
          const auto feedback=video->takeFeedback();
          feedback_keyframes+=feedback.keyframe_requested;
          bool cadence_changed=false;
          if(feedback.rate_control && feedback.rate_control->target_bitrate_bps>0) {
            event("{\"kind\":\"rate_feedback\",\"sample\":"+std::to_string(sample)
              +",\"bitrate_bps\":"+std::to_string(feedback.rate_control->target_bitrate_bps)
              +",\"framerate_fps\":"+std::to_string(feedback.rate_control->framerate_fps)+"}");
            encoder.bitrate(static_cast<int>(std::clamp<std::uint64_t>(feedback.rate_control->target_bitrate_bps,100000,8000000)));
            ++rate_updates;
            if(feedback.rate_control->framerate_fps>0) {
              // SetRates uses the incoming-frame-rate estimate for encoder bit
              // allocation. Apply it to NVENC without feeding it back into the
              // independently configured 30 Hz renderer/audio sample clock.
              const double selected=std::clamp(feedback.rate_control->framerate_fps,1.,60.);
              if(std::abs(selected-encoder_fps)>.01) {encoder_fps=selected;encoder.framerate(selected);++fps_updates;cadence_changed=true;}
            }
          }
          const bool preparing=next_video<preparation_samples;
          auto frame=encoder.encode(preparing?-1:frames-30,base+next_video*1000000/48000,
            frames==0 || next_video==preparation_samples || feedback.keyframe_requested || cadence_changed || next_video-last_keyframe_sample>=48000);
          if(frame.keyframe) last_keyframe_sample=next_video;
          livekit::EncodedVideoSource::Frame encoded;encoded.data=std::move(frame.data);
          encoded.timestamp_us=frame.pts_us;encoded.is_keyframe=frame.keyframe;encoded.width=512;encoded.height=512;
          livekit::VideoFrameMetadata tags;tags.frame_id=frames;tags.user_timestamp_us=frame.pts_us;
          encoded.metadata=tags;
          if(!video->captureFrame(encoded)) throw std::runtime_error("encoded source rejected access unit");
          ++frames;keyframes+=frame.keyframe;render_ms+=frame.render_gpu_ms;convert_ms+=frame.convert_gpu_ms;encode_ms+=frame.encode_wall_ms;
          next_video+=1600;
        }
        if(next_audio!=sample || next_audio>=total_samples) continue;
        std::vector<std::int16_t> pcm(480);
        for(int i=0;i<480;++i) {
          const auto position=sample+i;
          pcm[i]=(position>=preparation_samples && position%48000<9600)?static_cast<std::int16_t>(6000*std::sin(2*3.141592653589793*440*position/48000.0)):0;
        }
        // Direct SDK capture is exactly 10 ms and does not create a sender
        // buffering clock. The integer 48 kHz ledger schedules both cadences.
        audio->captureFrame(livekit::AudioFrame(std::move(pcm),48000,1,480),0);
        next_audio+=480;
      }
      // This SDK exposes a sender queue estimate, not an acoustic presentation ack.
      std::this_thread::sleep_until(origin+std::chrono::microseconds(total_samples*1000000/48000));
      event("{\"kind\":\"sender_complete\",\"audio_valid_samples\":"+std::to_string(seconds*48000LL)
        +",\"video_frames\":"+std::to_string(frames-30)+",\"preparation_frames\":30,\"preparation_samples\":48000,\"presentation_ack\":false}");
      if(!std::getline(std::cin,command) || command!="finish") throw std::runtime_error("qualification finish handshake required");
      participant->unpublishTrack(video_sid);participant->unpublishTrack(audio_sid);
      audio->clearQueue();room.disconnect();
      event("{\"kind\":\"complete\",\"audio_valid_samples\":"+std::to_string(seconds*48000LL)
        +",\"video_frames\":"+std::to_string(frames-30)+",\"preparation_frames\":30,\"preparation_samples\":48000,\"keyframes\":"+std::to_string(keyframes)
        +",\"feedback_keyframe_requests\":"+std::to_string(feedback_keyframes)+",\"rate_updates\":"+std::to_string(rate_updates)
        +",\"encoder_framerate_updates\":"+std::to_string(fps_updates)+",\"encoder_rate_framerate\":"+std::to_string(encoder_fps)
        +",\"renderer_framerate\":30"
        +",\"audio_sender_queue_ms\":0,\"maximum_schedule_lateness_ms\":"+std::to_string(maximum_schedule_lateness_ms)
        +",\"coded_bytes\":"+std::to_string(encoder.coded_bytes())+",\"render_gpu_total_ms\":"+std::to_string(render_ms)
        +",\"convert_gpu_total_ms\":"+std::to_string(convert_ms)+",\"encode_wall_total_ms\":"+std::to_string(encode_ms)
        +",\"physical_presentation_qualified\":false}");
    }
    livekit::shutdown();return 0;
  } catch(const std::exception& error) {
    std::cerr<<"native publisher: "<<error.what()<<'\n';
    if(initialized) livekit::shutdown();return 1;
  }
}
