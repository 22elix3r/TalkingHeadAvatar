"""Run the P04 local reference. This command cannot bind a public interface."""
import argparse
from pathlib import Path
import uvicorn
from talkinghead_v2.server.config import Settings
from talkinghead_v2.server.app import create_app

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config',type=Path,required=True)
args=parser.parse_args()
app=create_app(Settings.load(args.config))
uvicorn.run(app,host='127.0.0.1',port=18700,access_log=False,log_level='warning')
