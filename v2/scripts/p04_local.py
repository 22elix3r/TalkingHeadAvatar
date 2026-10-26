"""Initialize/start only TalkingHeadAvatar's isolated local P04 services."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
PRIVATE=ROOT/'var/p04'
PG='postgres@sha256:4e6e670bb069649261c9c18031f0aded7bb249a5b6664ddec29c013a89310d50'
SFU='livekit/livekit-server@sha256:25cd902de7cacbd5f1f779f6a6ac1eae20ba8f56b0935934ce6b0ab80d1e3d20'


def initialize():
    path=PRIVATE/'runtime.json'
    if path.exists(): raise SystemExit('configuration already exists; preserving existing secrets and evidence')
    cache=Path.home()/'.cache/huggingface/hub/models--Systran--faster-whisper-base.en/snapshots/3d3d5dee26484f91867d81cb899cfcf72b96be6c'
    required=['config.json','tokenizer.json','vocabulary.txt','model.bin']
    if any(not (cache/name).is_file() for name in required): raise SystemExit('cached ASR missing; no download attempted')
    with urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=5) as r: tags=json.load(r)
    model=next((m for m in tags['models'] if m['name']=='qwen3:14b'),None)
    if model is None: raise SystemExit('configured cached CPU LLM missing; no download attempted')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    os.chmod(PRIVATE,0o700)
    asr=PRIVATE/'asr';asr.mkdir()
    hashes={}
    for name in required:
        shutil.copyfile(cache/name,asr/name)
        hashes[name]=hashlib.sha256((asr/name).read_bytes()).hexdigest()
        os.chmod(asr/name,0o444)
    password,key,secret=secrets.token_hex(24),'tha-local',secrets.token_hex(32)
    settings={'database_url':f'postgresql+psycopg://tha:{password}@127.0.0.1:15432/tha_v2',
        'livekit_url':'ws://127.0.0.1:17880','livekit_key':key,'livekit_secret':secret,
        'ollama_url':'http://127.0.0.1:11434','llm_model':model['name'],'llm_digest':model['digest'],
        'asr_path':'asr','asr_files':hashes,'tts_runtime':'../native-runtime-serial.json',
        'web_dist':'../../web/dist','private_dir':'.','origin':'http://127.0.0.1:18700'}
    path.write_text(json.dumps(settings,indent=2)+'\n');os.chmod(path,0o600)
    sfu=PRIVATE/'livekit.yaml'
    sfu.write_text(f'port: 7880\nbind_addresses: ["0.0.0.0"]\nrtc:\n  node_ip: 127.0.0.1\n  tcp_port: 17881\n  udp_port: 17882\n  use_external_ip: false\nkeys:\n  {key}: {secret}\nlogging:\n  level: warn\n')
    os.chmod(sfu,0o600)
    compose={'services':{
        'postgres':{'image':PG,'ports':['127.0.0.1:15432:5432'],'environment':{'POSTGRES_USER':'tha','POSTGRES_PASSWORD':password,'POSTGRES_DB':'tha_v2'},
            'volumes':['pgdata:/var/lib/postgresql/data'],'healthcheck':{'test':['CMD-SHELL','pg_isready -U tha -d tha_v2'],'interval':'2s','timeout':'2s','retries':20}},
        'livekit':{'image':SFU,'command':['--config','/etc/livekit.yaml'],
            'ports':['127.0.0.1:17880:7880','127.0.0.1:17881:17881','127.0.0.1:17882:17882/udp'],
            'volumes':[str(sfu)+':/etc/livekit.yaml:ro,Z']}
        },'volumes':{'pgdata':{}}}
    (PRIVATE/'compose.json').write_text(json.dumps(compose,indent=2)+'\n');os.chmod(PRIVATE/'compose.json',0o600)
    public={'schema_version':1,'asr':{'model_id':'Systran/faster-whisper-base.en','revision':cache.name,'files':hashes,
        'runtime':'CTranslate2 CPU int8','stream_mode':'rolling-window revisions, not a native streaming decoder','weight_terms':'MIT (Whisper conversion)'},
        'llm':{'model':model['name'],'digest':model['digest'],'precision':model['details']['quantization_level'],
               'cpu_only':True,'context_tokens':2048,'weight_terms':'Apache-2.0'},
        'images':{'postgres':PG,'livekit':SFU},'serving_checkpoint_downloads':False}
    bundled=ROOT/'runtimes/session-api/.venv/lib/python3.12/site-packages/faster_whisper/assets/silero_vad_v6.onnx'
    if bundled.is_file():
        public['dependency_bundled_weight']={'path':'faster_whisper/assets/silero_vad_v6.onnx','bytes':bundled.stat().st_size,
            'sha256':hashlib.sha256(bundled.read_bytes()).hexdigest(),'loaded':False,'serving_vad':'webrtcvad-wheels',
            'constraint_deviation':'Installing the ASR dependency also downloaded this unused bundled weight file; not a zero-weight-download operation'}
    (ROOT/'configs/p04').mkdir(exist_ok=True)
    (ROOT/'configs/p04/source-lock.json').write_text(json.dumps(public,indent=2)+'\n')
    print('Initialized isolated loopback services with copied/hashed cached ASR and explicit cached CPU LLM; no model downloads')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['init','up','down'])
    action=parser.parse_args().action
    if action=='init': initialize();return
    if not (PRIVATE/'compose.json').is_file(): raise SystemExit('run init first')
    command=['docker','compose','-p','tha-v2-p04','-f',str(PRIVATE/'compose.json')]
    subprocess.run(command+(['up','-d','--wait'] if action=='up' else ['down']),check=True)


if __name__=='__main__': main()
