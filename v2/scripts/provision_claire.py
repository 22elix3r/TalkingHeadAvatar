"""Explicit pinned Claire/vendor-mesh provisioning; verify remains offline."""
import argparse
import json
from pathlib import Path

from talkinghead_v2.assets.pinned import PinnedFile, provision, verify
from talkinghead_v2.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = 'a46eafb067adfdd7cc8f5c7941586b309004561c'
AVATAR_REVISION = 'f0dc13670952c495450dab1655c94e9056c9d7d9'


def inputs(kind):
    filename = 'claire-artifacts.json' if kind == 'model' else 'claire-maya-source.json'
    data = json.loads((ROOT / 'configs/motion' / filename).read_text())
    revision = MODEL_REVISION if kind == 'model' else AVATAR_REVISION
    if data.get('schema_version') != 1 or data.get('revision') != revision:
        raise ContractError('only the inspected immutable Claire/vendor-sample revisions are permitted')
    entries = tuple(PinnedFile(**entry) for entry in data['files'])
    for entry in entries:
        if kind == 'model':
            expected = f'https://huggingface.co/nvidia/Audio2Face-3D-v2.3.1-Claire/resolve/{revision}/{entry.path}'
            if Path(entry.path).suffix not in {'.json', '.npz', '.onnx', '.md'}: raise ContractError('unknown model artifact type')
        else:
            host = 'media.githubusercontent.com/media' if entry.path.endswith('.ma') else 'raw.githubusercontent.com'
            expected = f'https://{host}/NVIDIA/Maya-ACE/{revision}/{entry.path}'
            if entry.path not in {'LICENSE.txt', 'sample_project/README.md', 'sample_project/maya_blendshape_v2/claire_arkit_v2_topo1.ma'}:
                raise ContractError('unknown vendor sample artifact')
        if entry.url != expected: raise ContractError('source URL does not match inspected repository/revision/path')
    if kind == 'model' and {entry.path for entry in entries} != {'README.md','a2f_ms_config.json','bs_skin.npz','bs_skin_config.json',
        'bs_tongue.npz','bs_tongue_config.json','config.json','implicit_emo_db.npz','model.json','model_config.json','model_data.npz','network.onnx','network_info.json','trt_info.json'}:
        raise ContractError('Claire package inventory mismatch; no alternate models or optional emotion weights')
    return revision, entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'provision'))
    parser.add_argument('--target', choices=('model','avatar','both'), required=True)
    parser.add_argument('--cache', type=Path, default=ROOT / 'var/artifacts')
    args = parser.parse_args()
    results = {}
    for kind in (('model','avatar') if args.target == 'both' else (args.target,)):
        revision, entries = inputs(kind)
        name = 'claire-' + kind + '-' + revision
        destination = (provision(args.cache, name, entries) if args.action == 'provision' else verify(args.cache / name, entries))
        results[kind] = {'verified': True, 'files': len(entries), 'bytes': sum(entry.size for entry in entries), 'snapshot': str(destination)}
    print(json.dumps({'action': args.action, 'results': results}, indent=2))


if __name__ == '__main__': main()
