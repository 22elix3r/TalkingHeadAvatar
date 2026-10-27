"""Run the P04 local reference. This command cannot bind a public interface."""
import argparse
from pathlib import Path
from urllib.parse import urlparse
import uvicorn
from talkinghead_v2.server.config import Settings
from talkinghead_v2.server.app import create_app

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config',type=Path,required=True)
args=parser.parse_args()
settings=Settings.load(args.config)
origin=urlparse(settings.origin)
if origin.scheme!='http' or origin.hostname!='127.0.0.1' or origin.port is None or origin.path or origin.query or origin.fragment or origin.username:
    parser.error('local serving requires an explicit http://127.0.0.1:port origin')
app=create_app(settings)
uvicorn.run(app,host='127.0.0.1',port=origin.port,access_log=False,log_level='warning')
