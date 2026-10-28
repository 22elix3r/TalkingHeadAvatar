"""Offline provenance and bounded metadata for an isolated publisher probe."""
from dataclasses import dataclass
import os
from pathlib import Path
import re
import stat

from talkinghead_v2.contracts import ContractError
from talkinghead_v2.server.inbound_config import private_json,file_hash,RUST_REVISION,CARGO_LOCK_SHA256

PATCH_SHA256='5aca3d55bcf694000757e675240d4a23df03590953da60aa7fe2638aca0604c6'
RATE_PATCH_SHA256='a08f5363be5ff8166b495c21e738d0647f022d774fec48e070b398a297211cd7'
MAX_EVENTS=16384


@dataclass(frozen=True)
class PublisherTraceConfig:
    executable:Path
    ffi_library:Path
    cpp_library:Path
    revision:str
    files:dict
    supports_external_rate_control:bool=False

    @classmethod
    def load(cls,path):
        path=Path(path);data=private_json(path)
        expected_support={PATCH_SHA256:False,RATE_PATCH_SHA256:True}.get(data.get('patch_sha256'))
        if (type(data.get('schema_version')) is not int or data['schema_version']!=1 or data.get('purpose')!='isolated_publisher_stage_trace' or
            expected_support is None or data.get('rust_revision')!=RUST_REVISION or
            type(data.get('abi_version')) is not int or data['abi_version']!=1 or
            type(data.get('max_events')) is not int or data['max_events']!=MAX_EVENTS or
            data.get('external_rate_control_opt_in',False) is not expected_support):
            raise ContractError('unqualified publisher trace policy')
        records=data.get('files')
        if not isinstance(records,dict) or set(records)!={'executable','ffi_library','cpp_library','build_report'}:
            raise ContractError('publisher trace inventory mismatch')
        paths={}
        for role,record in records.items():
            if (not isinstance(record,dict) or set(record)!={'path','size','sha256'} or not isinstance(record['path'],str) or
                not isinstance(record['sha256'],str) or not re.fullmatch('[a-f0-9]{64}',record['sha256'])):
                raise ContractError('publisher trace inventory record invalid')
            candidate=path.parent/record['path'];info=candidate.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o077 or info.st_nlink!=1 or
                type(record['size']) is not int or not 0<info.st_size==record['size']<=128*1024*1024 or file_hash(candidate)!=record['sha256']):
                raise ContractError('publisher trace inventory ownership/hash mismatch')
            paths[role]=candidate.resolve(strict=True)
        if paths['ffi_library'].parent!=paths['cpp_library'].parent:
            raise ContractError('isolated SDK library directory required')
        build=private_json(paths['build_report'])
        if (build.get('rust_revision')!=RUST_REVISION or build.get('patch_sha256')!=data['patch_sha256'] or
            build.get('cargo_lock_sha256')!=CARGO_LOCK_SHA256 or build.get('library_sha256')!=records['ffi_library']['sha256'] or
            build.get('external_rate_control_opt_in',False) is not expected_support or
            type(build.get('trace_tests_passed')) is not int or build['trace_tests_passed']!=5 or
            type(build.get('abi_version')) is not int or build['abi_version']!=1 or build.get('max_events')!=MAX_EVENTS):
            raise ContractError('publisher trace build provenance mismatch')
        return cls(paths['executable'],paths['ffi_library'],paths['cpp_library'],file_hash(path),records,expected_support)


def publish_stage_trace(lines,captures):
    """Keep stage identity and declared clocks; do not infer packet delivery."""
    bounds=[];rows=[]
    for line in lines:
        if line.startswith('THA_PUBLISH_TRACE_INCOMPLETE'):raise ContractError('incomplete publisher trace')
        if line.startswith('THA_PUBLISH_TRACE_BOUND '):
            fields=line.split()
            if len(fields)!=3:raise ContractError('publisher trace bound invalid')
            try:bounds.append(tuple(map(int,fields[1:])))
            except ValueError:raise ContractError('publisher trace bound invalid') from None
        elif line.startswith('THA_PUBLISH_TRACE '):
            fields=line.split()
            if len(fields)!=6 or len(rows)>=MAX_EVENTS:raise ContractError('publisher trace row/count invalid')
            try:stage,has_id,fid,pts,observed=map(int,fields[1:])
            except ValueError:raise ContractError('publisher trace row invalid') from None
            if stage not in (1,2,3) or has_id not in (0,1) or not 0<=fid<=2**32-1 or pts<0 or observed<=0:
                raise ContractError('publisher trace row outside bounds')
            rows.append({'stage':stage,'has_frame_id':bool(has_id),'frame_id':fid,'source_pts_us':pts,'observed_unix_us':observed})
    if len(bounds)!=1 or bounds[0]!=(len(rows),0) or not rows:raise ContractError('publisher trace missing, truncated or overflowed')
    accepted={row['frame_id']:row for row in captures}
    stages={stage:[] for stage in (1,2,3)};unknown={stage:0 for stage in (1,2,3)}
    for row in rows:
        if not row['has_frame_id'] or row['frame_id'] not in accepted:
            unknown[row['stage']]+=1;continue
        capture=accepted[row['frame_id']]
        if row['source_pts_us']!=capture['source_pts_us']:raise ContractError('publisher stage/source PTS mismatch')
        if row['frame_id'] in stages[row['stage']]:raise ContractError('duplicate publisher stage identity')
        stages[row['stage']].append(row['frame_id'])
    if set(stages[1])!=set(accepted):raise ContractError('source capture missing at encoder upload')
    if not set(stages[3])<=set(stages[2]):raise ContractError('packetizer stage lacks encoder output')
    return {'events':rows,'source_frame_ids_by_stage':{str(k):v for k,v in stages.items()},
        'non_source_or_unidentified_events_by_stage':{str(k):v for k,v in unknown.items()},
        'missing_source_frame_ids_at_encoder_output':sorted(set(accepted)-set(stages[2])),
        'missing_source_frame_ids_at_packetizer':sorted(set(accepted)-set(stages[3])),
        'observed_clock':'SDK Unix microseconds','source_pts_clock':'application nominal PTS; not subtracted from Unix clock',
        'packet_delivery_claimed':False,'scope':'bounded observation window sealed after capture stop and sender-stat settling'}
