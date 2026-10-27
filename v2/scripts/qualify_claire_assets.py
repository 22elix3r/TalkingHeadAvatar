"""Verify Claire/vendor geometry parity without executing scene scripts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from provision_claire import inputs, ROOT
from talkinghead_v2.assets.maya_geometry import extract_geometry
from talkinghead_v2.assets.pinned import verify
from talkinghead_v2.local import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.directory.exists() or args.report.exists():raise ValueError('preserve previous evidence')
    args.directory.mkdir(parents=True,mode=0o700)
    roots={}
    for kind in ['model','avatar']:
        revision,entries=inputs(kind)
        roots[kind]=verify(ROOT/'var/artifacts'/('claire-'+kind+'-'+revision),entries)
    inspected={}
    for path in roots['model'].glob('*.npz'):
        with zipfile.ZipFile(path) as archive:
            info=archive.infolist()
            if len(info)>512 or sum(x.file_size for x in info)>512*1024**2:raise ValueError('numeric archive resource bound exceeded')
        with np.load(path,allow_pickle=False) as archive:
            record={}
            for name in archive.files:
                data=archive[name]
                if data.dtype.hasobject:raise ValueError('pickle/object arrays forbidden')
                if data.dtype.kind in 'fci' and not np.isfinite(data).all():raise ValueError('nonfinite model data')
                record[name]={'dtype':str(data.dtype),'shape':list(data.shape)}
            inspected[path.name]=record
    source=roots['avatar']/'sample_project/maya_blendshape_v2/claire_arkit_v2_topo1.ma'
    text=source.read_text(encoding='cp1252')
    cases={}
    for name,node,basis in [('skin','neutralShapeOrig','bs_skin.npz'),('tongue','polySurfaceShape1_rest','bs_tongue.npz')]:
        geometry=extract_geometry(text,node)
        with np.load(roots['model']/basis,allow_pickle=False) as archive:
            neutral=archive['neutral']
            if neutral.shape!=(len(geometry.vertices),3):raise ValueError('source/model vertex shape mismatch; no truncation')
            error=float(np.abs(np.asarray(geometry.vertices)-neutral).max())
            if error>1e-6:raise ValueError('source/model vertex ordering/neutral mismatch')
            destination=args.directory/(name+'.npz')
            np.savez(destination,neutral_cm=neutral,triangles=np.asarray(geometry.triangles,dtype='<u4'),
                triangle_source_polygons=np.asarray(geometry.triangle_source_polygons,dtype='<u4'))
            destination.chmod(0o600)
            cases[name]={'node':node,'vertices':len(geometry.vertices),'source_polygons':len(geometry.polygon_vertices),
                'derived_triangles':len(geometry.triangles),'neutral_max_absolute_error_cm':error,
                'source_model_units':'centimeter','conversion_to_meters':0.01,
                'triangulation':'fixed source-quad diagonal preserving vertex order and triangle-to-source-polygon identity; no decimation',
                'derived_artifact_sha256':digest(destination),'basis_archive_sha256':digest(roots['model']/basis)}
    wrong=extract_geometry(text,'c_tongue_hiShapeOrig')
    with np.load(roots['model']/'bs_tongue.npz',allow_pickle=False) as archive:
        wrong_error=float(np.abs(np.asarray(wrong.vertices)-archive['neutral']).max())
    if wrong_error<=1e-6:raise ValueError('negative compatibility diagnostic was not reproduced')
    report={'schema_version':1,'phase':'P03','phase_status':'in_progress','evidence_category':'local_source_and_numeric_parity_observation',
        'scope':'official research-authorized vendor sample geometry matched to pinned Claire neutral arrays; no neural execution/quality inference',
        'source_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in [Path(__file__),ROOT/'src/talkinghead_v2/assets/maya_geometry.py',ROOT/'src/talkinghead_v2/assets/pinned.py',ROOT/'configs/motion/claire-artifacts.json',ROOT/'configs/motion/claire-maya-source.json']},
        'cases':cases,'negative_same_dimension_case':{'node':'c_tongue_hiShapeOrig','max_absolute_error_cm':wrong_error,'accepted':False},
        'numeric_archive_headers_and_finiteness':inspected,
        'rights':{'model':'NVIDIA Open Model License, retained pinned model card; specific user download authorization 2026-10-27',
            'vendor_sample':'NVIDIA/Maya-ACE sample-project README + repository MIT license retained; local research authorization',
            'human_consent_record':'not supplied; vendor sample is not user human-enrollment evidence',
            'training_dataset':'not fetched; separate evaluation-only license does not govern this sample-project import by implication',
            'commercial_distribution_clearance':False},
        'safety':{'pickle_loaded':False,'maya_or_mel_executed':False,'external_scene_references_executed':False,'source_assets_modified':False},
        'acceptance':{'vertex_identity_parity':True,'neural_execution':False,'rig_gaussian_retarget':False,'real_avatar_quality':False,'physical_av':False,'release':False}}
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'skin_neutral_parity':True,'tongue_rest_neutral_parity':True,'wrong_tongue_neutral_rejected':True,'neural_execution':False}))


if __name__=='__main__':main()
