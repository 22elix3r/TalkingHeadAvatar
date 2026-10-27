"""Prepare the separate full-size research renderer profile from verified geometry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np

from talkinghead_v2.local import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry-directory',type=Path,required=True)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    if args.directory.exists():raise ValueError('preserve existing research scene')
    report_path=ROOT/'benchmarks/reports/p03-claire-assets.json'
    report=json.loads(report_path.read_text())
    if not report['acceptance']['vertex_identity_parity'] or any(digest(ROOT/name)!=expected for name,expected in report['source_sha256'].items()):
        raise ValueError('matching source/model geometry qualification required')
    parts=[]
    for name in ['skin','tongue']:
        path=args.geometry_directory/(name+'.npz');case=report['cases'][name]
        if digest(path)!=case['derived_artifact_sha256']:raise ValueError('qualified geometry artifact hash mismatch')
        with np.load(path,allow_pickle=False) as archive:
            neutral=archive['neutral_cm'];faces=archive['triangles']
            if neutral.shape!=(case['vertices'],3) or faces.shape!=(case['derived_triangles'],3) or not np.isfinite(neutral).all():
                raise ValueError('qualified geometry shape mismatch')
            parts.append((neutral.astype('<f4'),faces.astype('<u4')))
    skin,tongue=parts
    vertices=np.concatenate((skin[0],tongue[0]))
    faces=np.concatenate((skin[1],tongue[1]+len(skin[0])))
    materials=np.concatenate((np.zeros(len(skin[1]),dtype='u1'),np.ones(len(tongue[1]),dtype='u1')))
    if len(vertices)>100000 or len(faces)>200000 or np.max(faces)>=len(vertices):raise ValueError('separate research profile exceeded; no decimation/clamping')
    low=vertices.min(axis=0).astype(float);high=vertices.max(axis=0).astype(float);center=(low+high)/2
    extent=float(max(high[0]-low[0],high[1]-low[1])*.01*.55)
    rotation=(1.,0.,0.,0.,-1.,0.,0.,0.,-1.)
    translation=(-float(center[0])*.01,float(center[1])*.01,1.)
    camera=(*rotation,*translation,.01,extent)
    payload=struct.pack('<8sII14d',b'THALRG1\0',len(vertices),len(faces),*camera)+vertices.tobytes()+faces.tobytes()+materials.tobytes()
    args.directory.mkdir(parents=True,mode=0o700)
    path=args.directory/'research-scene.bin'
    with open(path,'xb',opener=lambda name,flags:os.open(name,flags,0o600)) as target:target.write(payload)
    record={'schema_version':1,'profile':'claire_full_size_research_mesh','width':512,'height':512,
        'source_geometry_parity_report_sha256':digest(report_path),'geometry_sha256':digest(path),
        'vertex_count':len(vertices),'triangle_count':len(faces),'parts':{'skin':{'first_vertex':0,'vertices':len(skin[0]),'first_triangle':0,'triangles':len(skin[1])},
            'tongue':{'first_vertex':len(skin[0]),'vertices':len(tongue[0]),'first_triangle':len(skin[1]),'triangles':len(tongue[1])}},
        'topology_material_sha256':hashlib.sha256(faces.tobytes()+materials.tobytes()).hexdigest(),
        'camera':{'rotation':rotation,'translation_m':translation,'model_units':'centimeter','units_to_meters':.01,'half_extent_m':extent,'projection':'orthographic OpenCV x-right/y-down/z-forward'},
        'material':'explicit two-sided flat skin/tongue reference colors; no source texture, identity-quality or Gaussian parity claim',
        'source_assets_modified':False,'decimated':False,'small_serving_profile_changed':False}
    with open(args.directory/'scene.json','xb',opener=lambda name,flags:os.open(name,flags,0o600)) as target:target.write((json.dumps(record,indent=2)+'\n').encode())
    print(json.dumps({'research_scene_prepared':True,'vertices':len(vertices),'triangles':len(faces),'decimated':False}))


if __name__=='__main__':main()
