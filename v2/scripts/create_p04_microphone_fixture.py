"""Create declared synthetic browser input. This is not the serving TTS provider."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--directory',type=Path,default=Path('var/p04'))
args=parser.parse_args()
root=args.directory
root.mkdir(parents=True,exist_ok=True)
if any((root/name).exists() for name in ['question.wav','microphone.wav']):
    raise SystemExit('fixture already exists; preserve prior input/evidence')
text='What is two plus two?'
subprocess.run(['espeak-ng','-v','en-us','-s','145','-w',str(root/'question.wav'),text],check=True)
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-i',str(root/'question.wav'),
    '-af','adelay=3000,apad=pad_dur=45','-ar','48000','-ac','1','-c:a','pcm_s16le',str(root/'microphone.wav')],check=True)
(root/'microphone-fixture.json').write_text(json.dumps({'schema_version':1,'kind':'synthetic_microphone_fixture',
    'text':text,'source':'espeak-ng en-us, speed 145','role':'test input, not serving TTS',
    'leading_silence_ms':3000,'trailing_silence_seconds':45,'sample_rate':48000,
    'pcm_format':'mono signed PCM16','sha256':hashlib.sha256((root/'microphone.wav').read_bytes()).hexdigest()},indent=2)+'\n')
print('Created synthetic test microphone input; no model downloads or human recordings')
