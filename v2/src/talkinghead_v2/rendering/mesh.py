"""Small CPU triangle rasterizer for timing fallback, with explicit copy ownership."""

from dataclasses import dataclass
from fractions import Fraction
import math

from talkinghead_v2.assets.avatar import MeshAsset
from talkinghead_v2.contracts import ContractError, GenerationKey, MotionFrame, integer


def rotate(q, v):
    x,y,z,w = q
    tx,ty,tz = 2*(y*v[2]-z*v[1]),2*(z*v[0]-x*v[2]),2*(x*v[1]-y*v[0])
    return (v[0]+w*tx+y*tz-z*ty,v[1]+w*ty+z*tx-x*tz,v[2]+w*tz+x*ty-y*tx)


def deform(asset: MeshAsset, frame: MotionFrame) -> tuple:
    if frame.rig != asset.rig:
        raise ContractError("renderer basis/topology/order mismatch")
    if asset.rig.pose_semantics != "absolute_from_canonical":
        raise ContractError("mesh fallback requires absolute canonical pose")
    for value,(low,high) in zip(frame.coefficients,asset.bounds,strict=True):
        if not low <= value <= high:
            raise ContractError("motion exceeds calibrated coefficient bounds")
    vertices = [[v[k]+sum(weight*delta[i][k] for weight,delta in zip(frame.coefficients,asset.deltas,strict=True))
                 for k in range(3)] for i,v in enumerate(asset.vertices)]
    for i in asset.jaw_vertices:
        relative = tuple(vertices[i][k]-asset.jaw_pivot[k] for k in range(3))
        moved = rotate(frame.jaw_xyzw,relative)
        vertices[i] = [moved[k]+asset.jaw_pivot[k] for k in range(3)]
    return tuple(tuple(x+y for x,y in zip(rotate(frame.head_xyzw,tuple(v)),frame.translation_m,strict=True)) for v in vertices)


@dataclass(frozen=True)
class VideoFrame:
    key: GenerationKey
    pts_samples: int
    sample_rate: int
    duration: Fraction
    width: int
    height: int
    rgb: bytes
    boundary: str = "CPU completed pixels; no device presentation acknowledgment"

    def __post_init__(self):
        if not isinstance(self.key,GenerationKey):
            raise ContractError("video needs a typed generation identity")
        integer(self.pts_samples,"video PTS");integer(self.sample_rate,"video sample rate",1)
        integer(self.width,"video width",1);integer(self.height,"video height",1)
        if not isinstance(self.duration,Fraction) or self.duration<=0:
            raise ContractError("video needs a positive exact duration")
        if not isinstance(self.rgb,bytes) or len(self.rgb)!=self.width*self.height*3:
            raise ContractError("video RGB ownership/shape mismatch")


class MeshRenderer:
    def __init__(self,asset: MeshAsset,*,resolution=128):
        integer(resolution,"resolution",16)
        if resolution > 512:
            raise ContractError("CPU fallback resolution exceeds 512")
        self.asset,self.resolution = asset,resolution

    def render(self,frame: MotionFrame,*,duration: Fraction) -> VideoFrame:
        if not isinstance(duration,Fraction) or duration <= 0:
            raise ContractError("frame duration must be a positive exact fraction")
        vertices = deform(self.asset,frame)
        size = self.resolution
        pixels = bytearray(bytes((18,24,34))*(size*size))
        depth = [math.inf]*(size*size)
        projected = []
        for v in vertices:
            c = tuple(sum(row[k]*v[k] for k in range(3))+t for row,t in
                      zip(self.asset.camera_rotation,self.asset.camera_translation,strict=True))
            projected.append(((c[0]/self.asset.half_extent_m+1)*(size-1)/2,
                              (c[1]/self.asset.half_extent_m+1)*(size-1)/2,c[2]))
        def edge(a,b,x,y): return (x-a[0])*(b[1]-a[1])-(y-a[1])*(b[0]-a[0])
        for triangle in self.asset.faces:
            a,b,c = (projected[i] for i in triangle)
            if min(a[2],b[2],c[2]) <= 0:
                raise ContractError("mesh crosses the camera near plane; clipping is not implemented")
            area = edge(a,b,*c[:2])
            if abs(area) < 1e-12: continue
            xmin,xmax = max(0,math.floor(min(a[0],b[0],c[0]))),min(size-1,math.ceil(max(a[0],b[0],c[0])))
            ymin,ymax = max(0,math.floor(min(a[1],b[1],c[1]))),min(size-1,math.ceil(max(a[1],b[1],c[1])))
            color = (112,196,220)
            for y in range(ymin,ymax+1):
                for x in range(xmin,xmax+1):
                    weights = edge(b,c,x+.5,y+.5)/area,edge(c,a,x+.5,y+.5)/area,edge(a,b,x+.5,y+.5)/area
                    if min(weights) < -1e-12: continue
                    z = sum(w*p[2] for w,p in zip(weights,(a,b,c),strict=True))
                    index = y*size+x
                    if z < depth[index]:
                        depth[index] = z
                        pixels[index*3:index*3+3] = bytes(color)
        return VideoFrame(frame.envelope.key,frame.audio_sample,frame.sample_rate,duration,size,size,bytes(pixels))
