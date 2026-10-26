"""One bounded canonical mesh import recipe. No fitting/training or neural QC."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
import zlib

from talkinghead_v2.assets.avatar import load_avatar
from talkinghead_v2.contracts import ContractError,Envelope,GenerationKey,MotionFrame
from talkinghead_v2.rendering.mesh import MeshRenderer
from talkinghead_v2.server.lifecycle import MAX_UPLOAD,strict_json


def png(rgb,width=128,height=128):
    def chunk(kind,data): return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(kind+data)&0xffffffff)
    rows=b''.join(b'\0'+rgb[y*width*3:(y+1)*width*3] for y in range(height))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(rows))+chunk(b'IEND',b'')


def validate_import(lifecycle,claim):
    job,asset,consent=claim
    lifecycle.stage(claim,'validate')
    path=lifecycle.upload_path(job.tenant_id,job.upload_id)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>MAX_UPLOAD: raise ValueError('upload_not_regular')
    raw=path.read_bytes()
    from talkinghead_v2.server.lifecycle import Upload
    uploaded=lifecycle.get(Upload,job.tenant_id,job.upload_id)
    if len(raw)!=uploaded.size or hashlib.sha256(raw).hexdigest()!=uploaded.sha256: raise ValueError('immutable_upload_changed')
    bundle=strict_json(raw)
    if not isinstance(bundle,dict) or set(bundle)!={'schema_version','manifest','mesh'} or type(bundle['schema_version']) is not int or bundle['schema_version']!=1:
        raise ValueError('unsupported_import_bundle')
    if not isinstance(bundle['manifest'],str) or not isinstance(bundle['mesh'],str): raise ValueError('bundle_files_must_be_json_text')
    manifest,mesh=strict_json(bundle['manifest']),strict_json(bundle['mesh'])
    if not isinstance(manifest,dict) or not isinstance(mesh,dict): raise ValueError('invalid_mesh_metadata')
    expected={'synthetic_fixture':('procedural_fixture','timing_fixture_only'),
              'human_consented':('human_consented','research_avatar'),
              'research_authorized':('research_authorized','research_avatar')}[consent.kind]
    if manifest.get('kind')!=expected[0] or expected[1] not in consent.allowed_uses or expected[1] not in manifest.get('allowed_uses',[]):
        raise ValueError('consent_scope_mismatch')
    if manifest.get('appearance',{}).get('path')!='mesh.json': raise ValueError('import_requires_flat_mesh_json')
    if not isinstance(mesh.get('vertices'),list) or not 3<=len(mesh['vertices'])<=512: raise ValueError('mesh_reference_vertex_limit')
    if not isinstance(mesh.get('faces'),list) or not 1<=len(mesh['faces'])<=1024: raise ValueError('mesh_reference_face_limit')
    if not 1<=len(manifest.get('rig',{}).get('coefficient_names',[]))<=16: raise ValueError('mesh_reference_channel_limit')
    # Source bytes remain immutable. These explicit ownership/approval fields form
    # a derived manifest, whose publication is controlled by durable review state.
    manifest.update(tenant_id=asset.tenant_id,avatar_id=asset.id,status='approved')
    manifest['allowed_uses']=[expected[1]]
    manifest.setdefault('provenance',{})['authorization_record_id']=consent.id
    if consent.kind=='human_consented': manifest['provenance']['consent_record_id']=consent.id
    manifest_bytes=(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    mesh_bytes=bundle['mesh'].encode()
    revision=hashlib.sha256(json.dumps({'asset':asset.id,'tenant':asset.tenant_id,'manifest':hashlib.sha256(manifest_bytes).hexdigest(),
        'mesh':hashlib.sha256(mesh_bytes).hexdigest(),'tts':lifecycle.tts_revision,'consent_generation':job.consent_generation},sort_keys=True).encode()).hexdigest()
    directory=lifecycle.root/asset.tenant_id/'assets'/asset.id/revision
    lifecycle.stage(claim,'export')
    lifecycle.write_import_object(claim,directory/'mesh.json',mesh_bytes)
    lifecycle.write_import_object(claim,directory/'avatar.json',manifest_bytes)
    loaded=load_avatar(directory/'avatar.json',tenant_id=asset.tenant_id)
    lifecycle.stage(claim,'evaluate')
    neutral=MotionFrame(Envelope(GenerationKey('import-qc','neutral',0),0,'import-qc'),loaded.rig,0,48000,48000,
                        tuple(0. for _ in loaded.rig.coefficient_names))
    frame=MeshRenderer(loaded,resolution=128).render(neutral,duration=Fraction(1,30))
    visible=sum(frame.rgb[i:i+3]!=bytes((18,24,34)) for i in range(0,len(frame.rgb),3))
    if visible<4: raise ValueError('neutral_mesh_not_visible')
    lifecycle.write_import_object(claim,directory/'preview.png',png(frame.rgb))
    evaluation={'profile':'static_mesh_audio_reference','structural_compatibility':True,'neutral_visible_pixels':visible,
        'vertices':len(loaded.vertices),'faces':len(loaded.faces),'source_sha256':uploaded.sha256,
        'manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),'mesh_sha256':hashlib.sha256(mesh_bytes).hexdigest(),
        'rig_schema':loaded.rig.schema_id,'basis_hash':loaded.rig.basis_hash,'topology_hash':loaded.rig.topology_hash,
        'speaking_avatar_qualified':False,'human_quality_qualified':False,'held_out_identity_qc':'not_implemented_for_static_import',
        'voice_kind':'existing_synthetic_fixture','client_distribution':False}
    lifecycle.finish(claim,revision,evaluation)
    return revision
