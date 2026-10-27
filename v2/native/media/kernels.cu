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

// Full-size research profile. Input is exact float3 geometry in declared model
// units; projection/camera state is separate from immutable face identity.
extern "C" __global__ void project_large_mesh(const float* world,double* projected,int count,
                                              const double* camera,int width,int height,unsigned int* status) {
  const int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=count) return;
  const double x=world[index*3],y=world[index*3+1],z=world[index*3+2];
  if(!isfinite(x) || !isfinite(y) || !isfinite(z) || fabs(x)>1e6 || fabs(y)>1e6 || fabs(z)>1e6) {atomicOr(status,1u);return;}
  double p[3];
  for(int row=0;row<3;++row) p[row]=(camera[row*3]*x+camera[row*3+1]*y+camera[row*3+2]*z)*camera[12]+camera[9+row];
  p[0]=(p[0]/camera[13]+1)*(width-1)/2;p[1]=(p[1]/camera[13]+1)*(height-1)/2;
  if(!isfinite(p[0]) || !isfinite(p[1]) || !isfinite(p[2]) || fabs(p[0])>1e6 || fabs(p[1])>1e6 || p[2]<=0) {atomicOr(status,1u);return;}
  for(int coordinate=0;coordinate<3;++coordinate) projected[index*3+coordinate]=p[coordinate];
}

extern "C" __global__ void bin_large_mesh(const double* vertices,const unsigned int* faces,int count,
    const double* camera,const unsigned char* materials,unsigned int* colors,int width,int height,
    int columns,int rows,int tile_edge,unsigned int* counts,int* face_bounds,unsigned int* status) {
  const int face=blockIdx.x*blockDim.x+threadIdx.x;if(face>=count) return;
  auto* box=face_bounds+face*4;box[0]=1;box[1]=0;box[2]=1;box[3]=0;
  if(status[0]&1) return;
  const auto* a=vertices+faces[face*3]*3;const auto* b=vertices+faces[face*3+1]*3;const auto* c=vertices+faces[face*3+2]*3;
  if(fabs(edge(a,b,c[0],c[1]))<1e-12) return; // Zero projected area has no covered pixels.
  const int x0=max(0,static_cast<int>(floor(fmin(a[0],fmin(b[0],c[0])))));
  const int x1=min(width-1,static_cast<int>(ceil(fmax(a[0],fmax(b[0],c[0])))));
  const int y0=max(0,static_cast<int>(floor(fmin(a[1],fmin(b[1],c[1])))));
  const int y1=min(height-1,static_cast<int>(ceil(fmax(a[1],fmax(b[1],c[1])))));
  if(x0>x1 || y0>y1) return;
  const double scale_x=2*camera[13]/(width-1),scale_y=2*camera[13]/(height-1);
  const double ux=(b[0]-a[0])*scale_x,uy=(b[1]-a[1])*scale_y,uz=b[2]-a[2];
  const double vx=(c[0]-a[0])*scale_x,vy=(c[1]-a[1])*scale_y,vz=c[2]-a[2];
  const double nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;
  // Explicit two-sided flat reference material, no texture/identity claim.
  const double lighting=.25+.75*fabs((-.25*nx-.35*ny-nz)/sqrt((nx*nx+ny*ny+nz*nz)*1.185));
  const int red=static_cast<int>(floor((materials[face]?167:188)*lighting+.5));
  const int green=static_cast<int>(floor((materials[face]?83:151)*lighting+.5));
  const int blue=static_cast<int>(floor((materials[face]?87:129)*lighting+.5));
  colors[face]=static_cast<unsigned int>(red | (green<<8) | (blue<<16));
  box[0]=x0;box[1]=x1;box[2]=y0;box[3]=y1;
  for(int y=y0/tile_edge;y<=y1/tile_edge && y<rows;++y) for(int x=x0/tile_edge;x<=x1/tile_edge && x<columns;++x)
    atomicAdd(counts+y*columns+x,1u);
}

// At most 4096 tiles in this profile. A fixed single-thread scan avoids an
// allocation/sort dependency and publishes exact offsets before any list write.
extern "C" __global__ void scan_large_tiles(const unsigned int* counts,unsigned int* offsets,unsigned int* cursors,
    int tiles,int reference_capacity,int maximum_tile_faces,unsigned int* status) {
  if(blockIdx.x || threadIdx.x) return;
  unsigned int total=0,maximum=0;
  for(int tile=0;tile<tiles;++tile) {offsets[tile]=total;cursors[tile]=0;total+=counts[tile];maximum=max(maximum,counts[tile]);}
  status[1]=maximum;status[3]=total;
  if(total>static_cast<unsigned int>(reference_capacity) || maximum>static_cast<unsigned int>(maximum_tile_faces)) status[2]=1;
}

extern "C" __global__ void fill_large_tiles(const int* bounds,int count,int columns,int tile_edge,
    const unsigned int* offsets,unsigned int* cursors,unsigned int* lists,int reference_capacity,unsigned int* status) {
  const int face=blockIdx.x*blockDim.x+threadIdx.x;if(face>=count || status[0] || status[2]) return;
  const auto* box=bounds+face*4;if(box[0]>box[1] || box[2]>box[3]) return;
  for(int y=box[2]/tile_edge;y<=box[3]/tile_edge;++y) for(int x=box[0]/tile_edge;x<=box[1]/tile_edge;++x) {
    const int tile=y*columns+x;const unsigned int target=offsets[tile]+atomicAdd(cursors+tile,1u);
    if(target>=static_cast<unsigned int>(reference_capacity)) {atomicOr(status+4,1u);continue;}
    lists[target]=face;
  }
}

extern "C" __global__ void render_large_mesh(unsigned char* rgba,unsigned long long pitch,int width,int height,
    const double* vertices,const unsigned int* faces,const unsigned int* colors,const unsigned int* counts,
    const unsigned int* offsets,const unsigned int* lists,int columns,int tile_edge) {
  const int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=width || y>=height) return;
  auto* pixel=rgba+y*pitch+x*4;pixel[0]=18;pixel[1]=24;pixel[2]=34;pixel[3]=255;
  const int tile=(y/tile_edge)*columns+x/tile_edge;double depth=1e300;unsigned int best=0xffffffffu;
  for(unsigned int item=0;item<counts[tile];++item) {
    const auto face=lists[offsets[tile]+item];
    const auto* a=vertices+faces[face*3]*3;const auto* b=vertices+faces[face*3+1]*3;const auto* c=vertices+faces[face*3+2]*3;
    const double area=edge(a,b,c[0],c[1]);
    const double w0=edge(b,c,x+.5,y+.5)/area,w1=edge(c,a,x+.5,y+.5)/area,w2=edge(a,b,x+.5,y+.5)/area;
    if(w0< -1e-12 || w1< -1e-12 || w2< -1e-12) continue;
    const double z=w0*a[2]+w1*b[2]+w2*c[2];
    if(z>0 && (z<depth || (z==depth && face<best))) {depth=z;best=face;}
  }
  if(best!=0xffffffffu) {const auto color=colors[best];pixel[0]=color&255;pixel[1]=(color>>8)&255;pixel[2]=(color>>16)&255;}
}
