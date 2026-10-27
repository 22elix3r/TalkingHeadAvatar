"""Export declared synthetic-import evidence; never raw geometry, voices or secrets."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import xml.etree.ElementTree as ET

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from talkinghead_v2.server.config import Settings
from talkinghead_v2.server.db import Store,Base
from talkinghead_v2.server import lifecycle

ROOT=Path(__file__).resolve().parents[1]
PRIVATE=ROOT/'var/p05'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def tests(path):
    cases=list(ET.parse(path).getroot().iter('testcase'))
    if not cases or any(list(case.iter(tag)) for case in cases for tag in ['failure','error','skipped']):
        raise ValueError('declared tests must actually pass without skips')
    return len(cases)


def browser(path,count):
    data=json.loads(path.read_text())
    if data['stats']['expected']!=count or any(data['stats'][k] for k in ['unexpected','skipped','flaky']) or data.get('errors'):
        raise ValueError('browser cohort did not pass')
    names=[]
    def visit(rows):
        for row in rows:
            for spec in row.get('specs',[]):
                for case in spec['tests']:
                    if case['status']!='expected' or any(r['status']!='passed' for r in case['results']): raise ValueError('unexpected browser outcome')
                    names.append(spec['title'])
            visit(row.get('suites',[]))
    visit(data['suites'])
    return {'sha256':sha(path),'cases':names,'stats':data['stats']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/reports/p05-import-lifecycle.json')
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('preserving historical evidence: choose a new output')
    settings=Settings.load(ROOT/'var/p04/runtime.json')
    store=Store(settings.database_url)
    store.require_schema()
    with store.engine.connect() as connection:
        differences=compare_metadata(MigrationContext.configure(connection),Base.metadata)
    store.engine.dispose()
    if differences: raise ValueError('applied schema differs from qualified models')
    outcome=json.loads((PRIVATE/'browser-lifecycle-evidence.json').read_text())
    for key in ['import_review_select','real_qwen_received','revocation_gated_receiver','old_join_grant_denied']:
        if outcome[key] is not True: raise ValueError('missing lifecycle observation')
    if outcome['deletion_receipt']!='complete' or outcome['still_valid_sfu_token']['remoteTracks']!=0:
        raise ValueError('revoked source/objects were not qualified')
    installed=json.loads((PRIVATE/'firefox-system.json').read_text())
    if installed['loopback_ice_enabled'] or not installed['retry_after_failure_passed'] or installed['decoded_audio_observations']<2:
        raise ValueError('installed Firefox regression failed')
    source={}
    for name in ['src','scripts','web/src','web/tests','runtimes/session-api/migrations','runtimes/session-api/tests']:
        for path in sorted((ROOT/name).rglob('*')):
            if path.suffix in {'.py','.ts','.tsx','.css','.json'}: source[path.relative_to(ROOT).as_posix()]=sha(path)
    for name in ['web/playwright.config.ts','web/playwright.firefox.config.ts','web/playwright.lifecycle.config.ts']:
        source[name]=sha(ROOT/name)
    report={'schema_version':1,'phase':'P05','evidence_category':'local_observation',
        'status':'functional_static_import_and_revocation_passed_full_enrollment_and_quality_pending',
        'git_parent':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_files':source,'core_tests':tests(PRIVATE/'core-tests.xml'),'runtime_tests':tests(PRIVATE/'runtime-tests.xml'),
        'core_junit_sha256':sha(PRIVATE/'core-tests.xml'),'runtime_junit_sha256':sha(PRIVATE/'runtime-tests.xml'),
        'locks':{name:sha(ROOT/name) for name in ['uv.lock','runtimes/session-api/uv.lock','web/package-lock.json']},
        'tts_descriptor_sha256':sha(settings.tts_runtime),'p04_source_lock_sha256':sha(ROOT/'configs/p04/source-lock.json'),
        'schema_revision':'p05_0003','schema_matches_models':True,
        'browser_lifecycle':browser(PRIVATE/'playwright-lifecycle-final.json',2),
        'browser_p04_regression':browser(PRIVATE/'playwright-p04-regression-final.json',8),
        'browser_firefox':browser(PRIVATE/'playwright-firefox.json',1),
        'installed_firefox':installed,'installed_firefox_result_sha256':sha(PRIVATE/'firefox-system.json'),
        'lifecycle':outcome,'model_weight_downloads':False,'new_dependencies':False,
        'fixture_manifest_sha256':sha(ROOT/'src/talkinghead_v2/fixtures/p03-avatar/avatar.json'),
        'fixture_mesh_sha256':sha(ROOT/'src/talkinghead_v2/fixtures/p03-avatar/mesh.json'),
        'limitations':['Static canonical CPU mesh import only; no reconstruction/training or Claire/Gaussian path',
            'Existing synthetic Qwen conditioning only; human voice enrollment not implemented',
            'No independent human likeness/lip-sync/voice quality or three-identity qualification',
            'Receiver graph gating is not a calibrated physical stale-speech timing measurement',
            'Old SFU token may join an empty room; instantaneous token invalidation is not claimed',
            'Eight-hour local development identity only; hosted OIDC/roles/account recovery pending',
            'Backup retention/restore/tombstone replay and distributed native lease enforcement pending']}
    with urllib.request.urlopen(settings.origin+'/openapi.json',timeout=5) as response: schema=json.load(response)
    (ROOT/'docs/p05-openapi.json').write_text(json.dumps(schema,indent=2)+'\n')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'core_tests':report['core_tests'],'runtime_tests':report['runtime_tests'],
        'lifecycle_browser_checks':len(report['browser_lifecycle']['cases']),
        'revoked_token_rejoined_empty_room':outcome['still_valid_sfu_token']['connected']}))


if __name__=='__main__': main()
