// CPU-copy reference endpoint, separate from the GPU/SFU no-readback boundary.
#include <pulse/simple.h>
#include <pulse/error.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <poll.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>

namespace {
volatile std::sig_atomic_t stopped=0;
void stop(int) {stopped=1;}
constexpr int width=640,height=480;
struct Outputs {
  int camera=-1;pa_simple* audio=nullptr;bool cleared=false,camera_verified=false;
  std::vector<unsigned char> pixels=std::vector<unsigned char>(width*height*4);
  bool black() {
    std::fill(pixels.begin(),pixels.end(),0);
    cleared=camera>=0 && write(camera,pixels.data(),pixels.size())==static_cast<ssize_t>(pixels.size());
    return cleared;
  }
  ~Outputs() {
    if(camera>=0) {if(camera_verified && !cleared) black();close(camera);}
    if(audio) {int error=0;pa_simple_flush(audio,&error);pa_simple_free(audio);}
  }
};
void require(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
}

int main(int argc,char** argv) {
  try {
    require(argc==4,"usage: tha_virtual_reference /dev/video10 tha_v2_probe_sink private-48k-s16le.pcm");
    require(std::string(argv[1])=="/dev/video10","only the existing project Avatar Camera is eligible");
    const std::string sink=argv[2];
    require(sink.rfind("tha_v2_probe_",0)==0 && sink.size()<64,"project-owned virtual sink required");
    struct stat input{};
    require(lstat(argv[3],&input)==0 && S_ISREG(input.st_mode) && input.st_size>0 && input.st_size<=48000*2*30 && input.st_size%2==0,
      "bounded regular PCM input required (at most 30 seconds)");
    std::ifstream source(argv[3],std::ios::binary);
    std::vector<std::int16_t> pcm(input.st_size/2);source.read(reinterpret_cast<char*>(pcm.data()),input.st_size);
    require(source.gcount()==input.st_size,"short PCM input");
    Outputs out;
    out.camera=open(argv[1],O_WRONLY|O_NONBLOCK|O_CLOEXEC|O_NOFOLLOW);
    require(out.camera>=0,"Avatar Camera is unavailable");
    v4l2_capability capability{};
    require(ioctl(out.camera,VIDIOC_QUERYCAP,&capability)==0 &&
      std::string(reinterpret_cast<char*>(capability.card),strnlen(reinterpret_cast<char*>(capability.card),sizeof(capability.card)))=="Avatar Camera",
      "camera card identity mismatch");
    v4l2_format format{};format.type=V4L2_BUF_TYPE_VIDEO_OUTPUT;
    require(ioctl(out.camera,VIDIOC_G_FMT,&format)==0 && format.fmt.pix.width==width && format.fmt.pix.height==height &&
      format.fmt.pix.pixelformat==V4L2_PIX_FMT_BGR32 && format.fmt.pix.bytesperline==width*4 && format.fmt.pix.sizeimage==width*height*4,
      "reference requires the existing 640x480 BGR4 format; it never changes camera format");
    require(flock(out.camera,LOCK_EX|LOCK_NB)==0,"Avatar Camera is owned by another project producer");
    out.camera_verified=true;
    require(out.black(),"camera preparation clear failed");
    pa_sample_spec spec{PA_SAMPLE_S16LE,48000,1};
    pa_buffer_attr buffer{};buffer.maxlength=48000*2;buffer.tlength=480*2*4;
    // Do not run the playback clock through the preparation handshake with an
    // empty stream: prebuf=0 can leave later relative writes behind read time.
    // Start after the first complete 10 ms block instead.
    buffer.prebuf=480*2;buffer.minreq=480*2;
    int error=0;
    out.audio=pa_simple_new(nullptr,"tha-v2-virtual-reference",PA_STREAM_PLAYBACK,sink.c_str(),"project-reference",&spec,nullptr,&buffer,&error);
    require(out.audio!=nullptr,"virtual sink playback connection failed");
    std::signal(SIGTERM,stop);std::signal(SIGINT,stop);
    std::cout<<"{\"kind\":\"ready\",\"camera\":\"Avatar Camera\",\"width\":640,\"height\":480,\"audio_rate\":48000,\"raw_cpu_copy\":true}"<<std::endl;
    pollfd control{STDIN_FILENO,POLLIN,0};
    require(poll(&control,1,20000)>0 && !stopped,"bounded start handshake timed out/cancelled");
    std::string command;require(bool(std::getline(std::cin,command)) && command=="start","start handshake required");
    using Clock=std::chrono::steady_clock;
    const auto origin=Clock::now();
    const std::int64_t valid=pcm.size(),end=(valid+479)/480*480;
    std::int64_t audio=0,video=0;int frames=0;
    double lateness_ms=0;
    while(!stopped && (audio<end || video<end)) {
      const auto sample=std::min(audio,video);
      const auto due=origin+std::chrono::microseconds(sample*1000000/48000);
      std::this_thread::sleep_until(due);
      lateness_ms=std::max(lateness_ms,std::chrono::duration<double,std::milli>(Clock::now()-due).count());
      if(video==sample && video<end) {
        out.cleared=false;
        for(int y=0;y<height;++y) for(int x=0;x<width;++x) {
          auto* p=out.pixels.data()+(y*width+x)*4;
          p[0]=40+(x/32+frames)%180;p[1]=70+(y/24+frames*2)%180;p[2]=180;p[3]=255;
        }
        require(write(out.camera,out.pixels.data(),out.pixels.size())==static_cast<ssize_t>(out.pixels.size()),"short/nonblocking camera output; stopping explicitly");
        ++frames;video+=1600;
      }
      if(audio==sample && audio<end) {
        std::int16_t block[480]{};
        const auto count=std::min<std::int64_t>(480,valid-audio);
        if(count>0) std::copy_n(pcm.data()+audio,count,block);
        require(pa_simple_write(out.audio,block,sizeof(block),&error)==0,"virtual microphone audio write failed");
        audio+=480;
      }
    }
    require(out.black(),"camera privacy clear failed");
    if(stopped) require(pa_simple_flush(out.audio,&error)==0,"virtual microphone cancel flush failed");
    else require(pa_simple_drain(out.audio,&error)==0,"virtual microphone sender drain failed");
    std::cout<<"{\"kind\":\"sender_complete\",\"camera_cleared\":true,\"receiver_ack\":false}"<<std::endl;
    // Closing the producer immediately can prevent capture clients from ever
    // receiving its last black frame. Keep it open for a bounded consumer ack.
    bool acknowledged=false;int clear_frames=1;
    const auto clear_deadline=Clock::now()+std::chrono::seconds(5);
    while(Clock::now()<clear_deadline) {
      control={STDIN_FILENO,POLLIN,0};
      const auto available=poll(&control,1,33);
      require(available>=0,"finish wait interrupted");
      if(available>0) {
        acknowledged=bool(std::getline(std::cin,command)) && command=="finish";
        break;
      }
      // Capture/encode can miss a solitary final camera write. Keep neutral
      // frames flowing until the consumer confirms clear; these are not valid
      // response frames or additional speech samples.
      require(out.black(),"camera clear refresh failed");++clear_frames;
    }
    require(acknowledged,"bounded finish handshake required");
    std::cout<<"{\"kind\":\"complete\",\"cancelled\":"<<(stopped?"true":"false")
      <<",\"valid_samples\":"<<std::min(audio,valid)<<",\"padding_samples\":"<<(audio>valid?audio-valid:0)
      <<",\"video_frames\":"<<frames<<",\"camera_cleared\":true,\"camera_format_changed\":false,\"maximum_schedule_lateness_ms\":"<<lateness_ms
      <<",\"post_eof_black_frames\":"<<clear_frames
      <<",\"physical_presentation_ack\":false}"<<std::endl;
    return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
