"""Verify private offline replay containers without loading neural models."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import threading


def decoded_hash(path: Path, *, video: bool):
    command=['ffmpeg','-hide_banner','-loglevel','error','-i',str(path)]
    command+=['-map','0:v:0','-pix_fmt','rgb24','-f','rawvideo'] if video else ['-map','0:a:0','-f','f32le']
    digest=hashlib.sha256()
    count=0
    with tempfile.TemporaryFile() as diagnostics:
        process=subprocess.Popen([*command,'pipe:1'],stdout=subprocess.PIPE,stderr=diagnostics)
        timer=threading.Timer(60,process.kill)
        timer.start()
        try:
            while chunk:=process.stdout.read(65536):
                count+=len(chunk);digest.update(chunk)
            if process.wait(timeout=5)!=0:
                diagnostics.seek(0)
                raise RuntimeError('decoder failed or exceeded deadline: '+diagnostics.read(4096).decode(errors='replace'))
        finally:
            timer.cancel()
            process.stdout.close()
            if process.poll() is None: process.kill()
            process.wait()
    return digest.hexdigest(),count


def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(65536): digest.update(chunk)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--publish',type=Path,help='explicitly export synthetic-only metrics to this directory')
    args=parser.parse_args()
    root=args.directory
    report=json.loads((root/'report.json').read_text())
    if report.get('source',{}).get('kind') not in {'procedural_audio_visual_calibration','recorded_qwen_synthetic_reference'}:
        raise ValueError('verifier/export is scoped to the non-human P03 pilot')
    if report.get('actual_device_presentation') is not False or report.get('neural_motion_inference') is not False:
        raise ValueError('offline fixture cannot claim actual presentation or neural inference')
    if file_hash(root/'audio.wav')!=report['audio']['copied_wave_sha256']:
        raise ValueError('source audio changed after the replay')
    container=root/'preview.mkv'
    pcm_hash,pcm_bytes=decoded_hash(container,video=False)
    original_hash,original_bytes=decoded_hash(root/'audio.wav',video=False)
    rgb_hash,rgb_bytes=decoded_hash(container,video=True)
    expected=report['render']
    if pcm_hash!=original_hash or pcm_bytes!=original_bytes or pcm_bytes!=4*report['audio']['samples']:
        raise ValueError('mux changed or truncated source PCM')
    if rgb_hash!=expected['rgb_sha256'] or rgb_bytes!=expected['frames']*expected['resolution']**2*3:
        raise ValueError('mux changed or truncated completed RGB frames')
    streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams',
                                               '-of','json',str(container)],timeout=60))['streams']
    video=next(s for s in streams if s['codec_type']=='video')
    if (int(video['nb_read_frames'])!=expected['frames'] or
        [video['width'],video['height']]!=[expected['resolution']]*2 or video['avg_frame_rate']!=f"{expected['fps']}/1"):
        raise ValueError('video cadence/count/dimensions changed during mux')
    validation={'schema_version':1,'evidence_category':'local_observation','kind':'offline_container_roundtrip',
        'source':report['source']['kind'],'video_frames':expected['frames'],'resolution':[video['width'],video['height']],
        'audio_samples':pcm_bytes//4,'audio_pcm_sha256':pcm_hash,'audio_samples_preserved':True,
        'rgb_sha256':rgb_hash,'rgb_unchanged':True,'actual_device_presentation':False,
        'container_sha256':file_hash(container),'verifier_sha256':file_hash(Path(__file__)),
        'ffmpeg_version':subprocess.check_output(['ffmpeg','-version'],text=True,timeout=5).splitlines()[0]}
    (root/'container-verification.json').write_text(json.dumps(validation,indent=2)+'\n')
    if args.publish:
        args.publish.mkdir(parents=True,exist_ok=True)
        for suffix,data in [('',report),('-container',validation)]:
            (args.publish/(root.name+suffix+'.json')).write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'status':'passed','audio_samples':pcm_bytes//4,'video_frames':expected['frames'],
                      'actual_device_presentation':False}))


if __name__=='__main__':
    main()
