"""Package an already authorized canonical mesh as a bounded P05 import bundle."""
import argparse
import json
import os
from pathlib import Path

from talkinghead_v2.assets.avatar import load_avatar,read_json

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
metadata=read_json(args.manifest)
load_avatar(args.manifest,tenant_id=metadata['tenant_id'])
if metadata['appearance']['path']!='mesh.json': raise SystemExit('P05 currently requires the explicit flat mesh.json profile')
bundle=json.dumps({'schema_version':1,'manifest':args.manifest.read_text(),
                   'mesh':(args.manifest.parent/'mesh.json').read_text()},ensure_ascii=False).encode()
if len(bundle)>4*1024*1024: raise SystemExit('bundle exceeds the 4 MiB import limit')
args.output.parent.mkdir(parents=True,exist_ok=True)
fd=os.open(args.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'wb') as stream: stream.write(bundle)
print('Packaged immutable source JSON; the browser still requires consent and separate review')
