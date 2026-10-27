// First-party deterministic pattern and BT.709 limited-range RGB -> NV12.
// The intermediate RGB surface and encoder input remain on the CUDA device.
#include <cuda_runtime.h>

extern "C" __global__ void render_pattern(unsigned char* rgba, unsigned long long pitch,
                                        int width, int height, int frame) {
  const int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=width || y>=height) return;
  unsigned char* p=rgba+y*pitch+x*4;
  if(frame<0) {p[0]=p[1]=p[2]=0;p[3]=255;return;}
  p[0]=(x*255/width+frame*3)&255;
  p[1]=(y*255/height+frame*2)&255;
  p[2]=((x/32+y/32+frame/4)&1)?210:40;
  p[3]=255;
}

__device__ int bounded(int value,int low,int high) {return value<low?low:(value>high?high:value);}

extern "C" __global__ void rgba_to_nv12(const unsigned char* rgba,unsigned long long rgb_pitch,
                                      unsigned char* yy,unsigned long long y_pitch,
                                      unsigned char* uv,unsigned long long uv_pitch,int width,int height) {
  const int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=width || y>=height) return;
  const unsigned char* p=rgba+y*rgb_pitch+x*4;
  yy[y*y_pitch+x]=bounded(((47*p[0]+157*p[1]+16*p[2]+128)>>8)+16,16,235);
  if((x&1)==0 && (y&1)==0) {
    int r=0,g=0,b=0;
    for(int dy=0;dy<2;++dy) for(int dx=0;dx<2;++dx) {
      const unsigned char* q=rgba+(y+dy)*rgb_pitch+(x+dx)*4;r+=q[0];g+=q[1];b+=q[2];
    }
    r=(r+2)/4;g=(g+2)/4;b=(b+2)/4;
    uv[(y/2)*uv_pitch+x]=bounded(((-26*r-87*g+112*b+128)>>8)+128,16,240);
    uv[(y/2)*uv_pitch+x+1]=bounded(((112*r-102*g-10*b+128)>>8)+128,16,240);
  }
}
