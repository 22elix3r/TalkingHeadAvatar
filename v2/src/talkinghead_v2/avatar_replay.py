"""Runnable P03 offline timing path. Procedural controls are never neural motion."""

from array import array
from dataclasses import asdict
from fractions import Fraction
import hashlib
from importlib.resources import files
import json
import math
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import time
import wave

from talkinghead_v2.assets.artifacts import sha256_file
from talkinghead_v2.assets.avatar import load_avatar
from talkinghead_v2.audio.resample import RationalResampler
from talkinghead_v2.contracts import AudioChunk, AudioEOF, ContractError, Envelope, GenerationKey, MotionFrame
from talkinghead_v2.providers.motion.claire import ClaireContext
from talkinghead_v2.rendering.mesh import MeshRenderer
from talkinghead_v2.rendering.timeline import FrameMailbox, MotionTimeline


def run_identity():
    from talkinghead_v2.replay import _source_digest
    identity={'package_source_sha256':_source_digest(),'python':platform.python_version(),
              'kernel':platform.release(),'gpu_inference':False,'git_head':None,'working_diff_sha256':None}
    root=Path(__file__).resolve().parents[3]
    try:
        def git(*args):
            return subprocess.check_output(['git',*args],cwd=root,stderr=subprocess.DEVNULL,timeout=3)
        identity['git_head']=git('rev-parse','HEAD').decode().strip()
        dirty=git('diff','HEAD','--','v2/src')
        for name in git('ls-files','--others','--exclude-standard','-z','--','v2/src').split(b'\0'):
            if name: dirty+=name+b'\0'+(root/name.decode()).read_bytes()
        identity['working_diff_sha256']=hashlib.sha256(dirty).hexdigest()
        lock=root/'v2/uv.lock'
        if lock.exists(): identity['core_lock_sha256']=sha256_file(lock)
    except (OSError,subprocess.SubprocessError):
        pass  # Wheel installs retain a package digest without claiming Git metadata.
    return identity


def run(args):
    if bool(args.wave) != bool(args.tts_report):
        raise ContractError("recorded Qwen replay requires both wave and benchmark provenance report")
    if args.output.exists():
        raise ContractError("replay output already exists; preserve previous evidence")
    asset=load_avatar(Path(str(files('talkinghead_v2')/'fixtures/p03-avatar/avatar.json')),tenant_id='procedural')
    renderer=MeshRenderer(asset,resolution=args.resolution)
    source={"kind":"procedural_audio_visual_calibration","model_inference":False}
    request=None
    if args.wave:
        metadata=json.loads(args.tts_report.read_text())
        if metadata.get('voice_kind')!='synthetic_fixture' or metadata.get('incremental_audio_output') is not True:
            raise ContractError("this replay command accepts only the synthetic-reference Qwen pilot")
        request=next((r for r in metadata['requests'] if r['id']==args.case),None)
        if request is None:
            raise ContractError("requested recorded case is absent")
        import soundfile as sf  # Optional inference environment, never a core dependency.
        info=sf.info(args.wave)
        if info.channels!=1 or info.samplerate!=24000 or not 0<info.frames<=60*24000:
            raise ContractError("recorded replay requires bounded mono 24 kHz PCM")
        pcm,rate=sf.read(args.wave,dtype='float32')
        raw=pcm.astype('<f4',copy=False).tobytes()
        if hashlib.sha256(raw).hexdigest()!=request['pcm_sha256'] or len(pcm)!=request['samples']:
            raise ContractError("recorded PCM differs from qualified request")
        source={"kind":"recorded_qwen_synthetic_reference","model_inference":False,
                "model_revision":metadata['model_revision'],"runtime_revision":metadata['runtime_revision'],
                "worker_environment":metadata['worker_environment'],"source_wave_sha256":sha256_file(args.wave),
                "source_report_sha256":sha256_file(args.tts_report),"case":args.case}
    else:
        rate=24000
        pcm=array('f',(0.15*math.sin(2*math.pi*880*i/rate) if i%rate<480 else 0. for i in range(3*rate)))
    args.output.mkdir(parents=True)
    audio_path=args.output/'audio.wav'
    if args.wave: shutil.copyfile(args.wave,audio_path)
    else:
        with wave.open(str(audio_path),'wb') as stream:
            stream.setparams((1,2,rate,0,'NONE','not compressed'))
            stream.writeframes(b''.join(struct.pack('<h',round(x*32767)) for x in pcm))
    key=GenerationKey('p03-offline-timing','recorded-audio',0)
    context=ClaireContext(key)
    branch=RationalResampler(rate,16000)
    first_ready_input=None
    seq=context_cursor=windows=right_padded=0
    first_ready_callback=None
    def feed(values):
        nonlocal seq,context_cursor,windows,right_padded,first_ready_input,first_ready_callback
        for start in range(0,len(values),320):
            chunk=values[start:start+320]
            context.push(AudioChunk(Envelope(key,seq,'recorded-context'),'source',16000,context_cursor,chunk))
            seq+=1;context_cursor+=len(chunk)
            collect()
    def collect():
        nonlocal windows,right_padded,first_ready_input,first_ready_callback
        while (window:=context.pop()) is not None:
            if first_ready_input is None:
                first_ready_input=branch.input_count
                if request:
                    first_ready_callback=next((c['received_ns'] for c in request['callbacks']
                                              if c['cumulative_samples']>=first_ready_input),None)
            windows+=1
            right_padded+=bool(window.right_padding)
    for start in range(0,len(pcm),480):
        feed(branch.push(tuple(float(x) for x in pcm[start:start+480])))
    feed(branch.finish())
    context.push(AudioEOF(Envelope(key,seq,'recorded-context'),16000,context_cursor,branch.valid_end_sample))
    collect()
    # The reference rig follows a declared test pattern, never audio-amplitude or
    # invented phoneme timing. Audio zero and flashes calibrate the file timeline.
    output_rate,fps,motion_fps=48000,30,50
    valid_end=Fraction(len(pcm)*output_rate,rate)
    frame_count=(len(pcm)*fps+rate-1)//rate
    timeline=MotionTimeline(key,asset.rig,output_rate)
    mailbox=FrameMailbox(key)
    motion_index=0
    render_ns=[]
    trace=[]
    image_hash=hashlib.sha256()
    with (args.output/'video.rgb').open('wb') as stream:
        for index in range(frame_count):
            target=index*output_rate//fps
            timeline.advance(target)
            while (center:=motion_index*output_rate//motion_fps)<valid_end and center<=target+output_rate//motion_fps:
                phase=(center%output_rate)/output_rate
                coefficients=(1-abs(2*phase-1),0.)
                timeline.push(MotionFrame(Envelope(key,motion_index,'procedural-controls'),asset.rig,center,
                    output_rate,min(center+output_rate//motion_fps,math.ceil(valid_end)),coefficients))
                motion_index+=1
            selected=timeline.select(target)
            if selected is None: raise RuntimeError("timing fixture lost valid motion")
            started=time.perf_counter_ns()
            rendered=renderer.render(selected,duration=Fraction(1,fps))
            render_ns.append(time.perf_counter_ns()-started)
            if source['kind']=='procedural_audio_visual_calibration' and index%fps==0:
                from dataclasses import replace
                rendered=replace(rendered,rgb=bytes((255,255,255))*(args.resolution**2))
            mailbox.submit(rendered,playhead=target)
            ready=mailbox.take_due(playhead=target)
            if ready is None or ready.key!=key: raise RuntimeError("video queue lost/fenced current frame")
            stream.write(ready.rgb);image_hash.update(ready.rgb)
            if index==1:
                (args.output/'mesh.ppm').write_bytes(f'P6\n{args.resolution} {args.resolution}\n255\n'.encode()+ready.rgb)
            trace.append({'frame':index,'pts_samples':ready.pts_samples,'sample_rate':output_rate,
                          'motion_sequence':selected.envelope.sequence_no,'logical_skew_samples':ready.pts_samples-target})
    # Exercise already-completed CPU callbacks after cancellation, separately from
    # device playback and neural-compute cancellation (not present in this run).
    timeline.cancel();mailbox.cancel();context.cancel()
    stale_rejected=not timeline.push(selected) and not mailbox.submit(rendered,playhead=target)
    if not stale_rejected: raise RuntimeError("completed stale callback escaped cancellation fence")
    report={'schema_version':1,'phase':'P03','evidence_category':'local_observation',
        'run_identity':run_identity(),
        'status':'timing_fixture_passed_real_avatar_pending','source':source,
        'visual_kind':'procedural_mesh_timing_only','neural_motion_inference':False,
        'actual_device_presentation':False,'human_quality_qualified':False,
        'avatar_manifest_sha256':asset.manifest_sha256,'rig':asdict(asset.rig),
        'audio':{'samples':len(pcm),'sample_rate':rate,'output_valid_end_samples':str(valid_end),
                 'copied_wave_sha256':sha256_file(audio_path)},
        'claire_frontend':{'sample_rate':16000,'window_samples':8320,'right_context_samples':4160,
            'right_context_ms':260,'resampler_right_context_seconds':str(branch.right_context),
            'context_windows':windows,'right_padded_eof_windows':right_padded,
            'first_window_available_after_source_samples':first_ready_input,
            'recorded_callback_observation_ns':first_ready_callback,
            'peak_buffered_samples':context.peak_buffered_samples,'max_buffered_samples':context.max_buffered_samples,
            'model_ran':False},
        'render':{'frames':frame_count,'fps':fps,'motion_fixture_fps':motion_fps,'resolution':args.resolution,
            'rgb_sha256':image_hash.hexdigest(),'cpu_service_max_ns':max(render_ns),'frame_trace':trace,
            'motion_peak_items':timeline.peak_items,'motion_max_items':timeline.max_items,
            'motion_future_bound_samples':timeline.max_samples,'video_peak_items':mailbox.peak_items,
            'video_max_items':mailbox.max_items,'completed_stale_callbacks_rejected':stale_rejected},
        'limitations':['Offline file timestamps and synthetic controls do not prove presented lip sync',
            'Claire inference/retarget calibration, real Gaussian avatar and Q0 clips are unavailable',
            'No Qwen+motion+render GPU co-load, wall-clock 30-minute A/V soak or stale-audio suppression claim']}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
