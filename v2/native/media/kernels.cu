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

__device__ double edge(const double* a,const double* b,double x,double y) {
  return (x-a[0])*(b[1]-a[1])-(y-a[1])*(b[0]-a[0]);
}

// Matched neutral CPU reference: projected OpenCV x-right/y-down geometry,
// constant material, exact face order and nearest positive depth. No topology
// repair or coefficient adaptation occurs inside this bounded renderer.
extern "C" __global__ void render_mesh(unsigned char* rgba,unsigned long long pitch,int width,int height,
                                      const double* vertices,const unsigned int* faces,const int* bounds,int count) {
  const int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=width || y>=height) return;
  auto* p=rgba+y*pitch+x*4;p[0]=18;p[1]=24;p[2]=34;p[3]=255;
  double depth=1.0e300;
  for(int face=0;face<count;++face) {
    const auto* box=bounds+face*4;
    if(x<box[0] || x>box[1] || y<box[2] || y>box[3]) continue;
    const auto* a=vertices+faces[face*3]*3;
    const auto* b=vertices+faces[face*3+1]*3;
    const auto* c=vertices+faces[face*3+2]*3;
    const double area=edge(a,b,c[0],c[1]);
    if(fabs(area)<1.0e-12) continue;
    const double w0=edge(b,c,x+.5,y+.5)/area,w1=edge(c,a,x+.5,y+.5)/area,w2=edge(a,b,x+.5,y+.5)/area;
    if(w0< -1.0e-12 || w1< -1.0e-12 || w2< -1.0e-12) continue;
    const double z=w0*a[2]+w1*b[2]+w2*c[2];
    if(z<depth) {depth=z;p[0]=112;p[1]=196;p[2]=220;}
  }
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
