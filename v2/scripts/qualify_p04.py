"""Publish synthetic-only P04 observations after real browser tests; no media export."""

from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
PRIVATE=ROOT/'var/p04'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def count_junit(path):
    data=ET.parse(path).getroot()
    cases=list(data.iter('testcase'))
    if not cases or any(list(case.iter('failure')) or list(case.iter('error')) or list(case.iter('skipped')) for case in cases):
        raise ValueError('all declared tests must actually run and pass')
    return len(cases)


def main():
    from talkinghead_v2.server.config import Settings
    from talkinghead_v2.server.providers import Asr
    from talkinghead_v2.replay import _source_digest
    import av
    settings=Settings.load(PRIVATE/'runtime.json')
    browser=json.loads((PRIVATE/'playwright-final.json').read_text())
    if browser['stats']['expected']!=7 or browser['stats']['unexpected'] or browser['stats']['skipped'] or browser.get('errors'):
        raise ValueError('the complete real browser cohort did not pass')
    cases=[]
    def visit(suites):
        for suite in suites:
            for spec in suite.get('specs',[]):
                for test in spec['tests']:
                    if test['status']!='expected' or any(r['status']!='passed' for r in test['results']):
                        raise ValueError('unexpected browser outcome')
                    cases.append({'name':spec['title'],'status':'passed','duration_ms':test['results'][-1]['duration']})
            visit(suite.get('suites',[]))
    visit(browser['suites'])
    # This harness records only the declared synthetic input case. No general
    # human recording/export command is provided here.
    recorded=PRIVATE/'browser-microphone-answer.webm'
    container=av.open(str(recorded));resampler=av.AudioResampler(format='s16',layout='mono',rate=16000)
    chunks=[]
    for frame in container.decode(audio=0):
        for out in resampler.resample(frame): chunks.append(out.to_ndarray().tobytes())
    for out in resampler.resample(None): chunks.append(out.to_ndarray().tobytes())
    container.close()
    if sum(map(len,chunks))>16000*2*60: raise ValueError('receiver test capture exceeds 60-second bound')
    asr=Asr(settings.asr_path,settings.asr_files)
    transcript=asr.decode(b''.join(chunks));asr.close()
    if '4' not in transcript and 'four' not in transcript.lower(): raise ValueError('answer tail was not recognized in receiver audio')
    latency=json.loads((PRIVATE/'client-graph-latency.json').read_text())
    source=json.loads((ROOT/'configs/p04/source-lock.json').read_text())
    sources={}
    for folder in ['src','web/src','web/tests','runtimes/session-api/migrations']:
        for p in sorted((ROOT/folder).rglob('*')):
            if p.suffix in {'.py','.ts','.tsx','.css','.json'}: sources[p.relative_to(ROOT).as_posix()]=sha(p)
    scopes=['v2/src','v2/web','v2/runtimes/session-api']
    dirty=subprocess.check_output(['git','diff','--binary','HEAD','--',*scopes],cwd=ROOT.parent)
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z','--',*scopes],cwd=ROOT.parent)
    for name in filter(None,untracked.split(b'\0')): dirty+=name+b'\0'+(ROOT.parent/name.decode()).read_bytes()
    report={'schema_version':1,'phase':'P04','evidence_category':'local_observation',
        'status':'browser_audio_reference_passed_real_avatar_and_quality_pending',
        'profile':'loopback_audio_reference_synthetic_microphone','human_user_evaluation':False,
        'git_parent':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'package_source_sha256':_source_digest(),'source_files':sources,
        'working_diff_sha256':hashlib.sha256(dirty).hexdigest(),
        'core_junit_sha256':sha(PRIVATE/'core-tests.xml'),'runtime_junit_sha256':sha(PRIVATE/'runtime-tests.xml'),
        'tts_runtime_descriptor_sha256':sha(settings.tts_runtime),
        'tts_build_report_sha256':sha(ROOT/'benchmarks/reports/p02-native-build.json'),
        'core_lock_sha256':sha(ROOT/'uv.lock'),'server_lock_sha256':sha(ROOT/'runtimes/session-api/uv.lock'),
        'web_lock_sha256':sha(ROOT/'web/package-lock.json'),
        'runtime_versions':{name:version(name) for name in ['fastapi','livekit','livekit-api','faster-whisper','ctranslate2','sqlalchemy','psycopg','webrtcvad-wheels']},
        'sources':source,'core_tests':count_junit(PRIVATE/'core-tests.xml'),
        'postgres_api_fault_tests':count_junit(PRIVATE/'runtime-tests.xml'),
        'browser_tests':cases,'browser_stats':browser['stats'],
        'input_fixture_sha256':sha(PRIVATE/'microphone.wav'),
        'receiver_audio':{'boundary':'MediaRecorder of received LiveKit track, not acoustic output',
            'recording_sha256':sha(recorded),'samples_16khz':sum(map(len,chunks))//2,
            'asr_transcript':transcript,'answer_final_word_recognized':True},
        'latency':latency,'initial_asr_failure':json.loads((PRIVATE/'initial-asr-failure.json').read_text()),
        'limitations':['No real-human cohort, natural cutoff/echo rates, or human listening acceptance',
            'Static procedural video only; no real speaking avatar or presented lip-sync qualification',
            'Graph latency is n=1 with uncalibrated buffering; no meaningful acoustic or audiovisual p95/p99',
            'No real A/V soak, Qwen+motion+render GPU co-load, distributed admission or hosted release pass']}
    out=ROOT/'benchmarks/reports/p04-browser-reference.json';out.write_text(json.dumps(report,indent=2)+'\n')
    with urllib.request.urlopen(settings.origin+'/openapi.json',timeout=5) as response: schema=json.load(response)
    (ROOT/'docs/p04-openapi.json').write_text(json.dumps(schema,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'core_tests':report['core_tests'],
        'runtime_tests':report['postgres_api_fault_tests'],'browser_tests':len(cases),
        'receiver_audio_transcript':transcript,'observed_graph_ms':latency['observed_ms']}))


if __name__=='__main__': main()
