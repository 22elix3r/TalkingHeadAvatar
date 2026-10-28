"""Run the P04 local reference. This command cannot bind a public interface."""
import argparse
import os
from pathlib import Path
from urllib.parse import urlparse
import uvicorn
from talkinghead_v2.server.config import Settings

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config',type=Path,required=True)
args=parser.parse_args()
settings=Settings.load(args.config)
inbound=None
if settings.inbound_audio_runtime is not None:
    from talkinghead_v2.server.inbound_config import InboundAudioConfig
    inbound=InboundAudioConfig.load(settings.inbound_audio_runtime)
    selected=inbound.environment(os.environ)
    for key in ('LD_PRELOAD','LD_AUDIT'):os.environ.pop(key,None)
    os.environ['LIVEKIT_LIB_PATH']=selected['LIVEKIT_LIB_PATH']
from talkinghead_v2.server.app import create_app
origin=urlparse(settings.origin)
if origin.scheme!='http' or origin.hostname!='127.0.0.1' or origin.port is None or origin.path or origin.query or origin.fragment or origin.username:
    parser.error('local serving requires an explicit http://127.0.0.1:port origin')
app=create_app(settings)
try:
    uvicorn.run(app,host='127.0.0.1',port=origin.port,access_log=False,log_level='warning')
finally:
    if inbound is not None:
        from talkinghead_v2.server.inbound_audio import dispose_owned_runtime
        dispose_owned_runtime(inbound)
