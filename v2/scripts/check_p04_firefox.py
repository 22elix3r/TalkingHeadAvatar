"""Real installed-Firefox regression via an isolated WebDriver profile; no models fetched.

Requires an operator-provisioned geckodriver binary and the real P04 services.
Uses only stdlib HTTP, rather than changing the user's Firefox profile or prefs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request


ROOT=Path(__file__).resolve().parents[1]


def run(driver: Path, output: Path):
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1',0))
        port=reservation.getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    output.parent.mkdir(parents=True,exist_ok=True)
    log=output.with_suffix('.driver.log').open('w')
    process=subprocess.Popen([str(driver.resolve()),'--host','127.0.0.1','--port',str(port)],stdout=log,stderr=log)
    session=None

    def request(method,path,data=None):
        body=None if data is None else json.dumps(data).encode()
        req=urllib.request.Request(base+path,data=body,method=method,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=20) as response:
            result=json.load(response)['value']
        return result

    def command(path,data): return request('POST','/session/'+session+path,data)
    def js(script): return command('/execute/sync',{'script':script,'args':[]})
    def wait(script,seconds=45):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            if js(script): return
            time.sleep(.1)
        raise AssertionError('Firefox condition failed: '+script+'; '+str(js('return document.body.innerText')))
    def click(name):
        element=command('/element',{'using':'xpath','value':f'//button[contains(normalize-space(.),"{name}")]'})
        command('/element/'+element['element-6066-11e4-a52e-4f735466cecf']+'/click',{})
    def send(text):
        js('document.querySelector("#message").focus()')
        element=command('/element',{'using':'css selector','value':'#message'})
        command('/element/'+element['element-6066-11e4-a52e-4f735466cecf']+'/value',{'text':text})
        click('↑')
    def received(count):
        end=time.monotonic()+60
        while time.monotonic()<end:
            if js(f'return window.__observations.filter(e=>e.kind==="decoded_audio_observed").length>={count}'): return
            if js('return !!document.querySelector(".audio-unlock")'): click('Enable audio playback')
            time.sleep(.1)
        raise AssertionError('No decoded Firefox audio: '+str(js('return document.body.innerText')))
    try:
        deadline=time.monotonic()+10
        while True:
            if process.poll() is not None: raise RuntimeError('geckodriver failed; inspect private log')
            try:
                request('GET','/status');break
            except (OSError,urllib.error.URLError):
                if time.monotonic()>deadline: raise
                time.sleep(.1)
        value=request('POST','/session',{'capabilities':{'alwaysMatch':{
            'browserName':'firefox','moz:firefoxOptions':{'binary':'/usr/bin/firefox','args':['-headless'],
                'prefs':{'media.peerconnection.ice.loopback':False,'media.navigator.streams.fake':True,
                         'media.navigator.permission.disabled':True}}}}})
        session=value['sessionId']
        command('/timeouts',{'script':5000,'pageLoad':15000,'implicit':0})
        command('/url',{'url':'http://127.0.0.1:18700'})
        js('''window.__observations=[];window.__peers=[];window.__forceLoopback=true;
            window.addEventListener('tha-observation',e=>window.__observations.push(e.detail));
            const Original=window.RTCPeerConnection;
            function rewrite(line) {
              if(!window.__forceLoopback || !/^(a=)?candidate:/.test(line))return line;
              const parts=line.split(' ');parts[4]='127.0.0.1';return parts.join(' ');
            }
            window.RTCPeerConnection=class extends Original {
              constructor(config){super(config);window.__peers.push(this);}
              setRemoteDescription(desc) {
                return super.setRemoteDescription({...desc,sdp:desc.sdp?.split('\\r\\n').map(rewrite).join('\\r\\n')});
              }
              addIceCandidate(candidate) {
                if(!candidate)return super.addIceCandidate(candidate);
                const data=candidate.toJSON ? candidate.toJSON() : candidate;
                return super.addIceCandidate({...data,candidate:rewrite(data.candidate || '')});
              }
            };''')
        # Fault fixture recreates the previous SFU's advertised address without
        # restarting services or touching the user's profile. Then retry normally.
        click('Use typed messages instead')
        wait('return document.querySelector("[role=alert]")?.textContent.includes("The audio connection could not be established")')
        wait('return ![...document.querySelectorAll("button")].find(b=>b.textContent.includes("Use typed messages instead")).disabled')
        assert not js('return window.__observations.some(e=>e.kind==="connected")')
        js('window.__forceLoopback=false;window.__peers=[]')
        click('Use typed messages instead')
        wait('return document.querySelector("[role=status]").textContent.includes("Ready for a message")')
        send('What is two plus two?');received(1)
        click('Interrupt')
        wait('return document.querySelector("audio").srcObject===null || document.querySelector("audio").muted')
        send('Say green and count from one to five.');received(2)
        stats=command('/execute/async',{'script':'''const done=arguments[arguments.length-1];
            Promise.all(window.__peers.map(async p=>[...(await p.getStats()).values()]))
            .then(rows=>done(rows.flat().filter(s=>s.type==='remote-candidate' || s.type==='candidate-pair')));''','args':[]})
        events=js('return window.__observations')
        bindings=[e for e in events if e['kind']=='playback_binding']
        assert len(bindings)>=2 and bindings[-1]['audio_sid']!=bindings[0]['audio_sid']
        assert bindings[-1]['epoch']>bindings[0]['epoch']
        candidates={r['id']:r for r in stats if r['type']=='remote-candidate'}
        pairs=[r for r in stats if r['type']=='candidate-pair' and r.get('state')=='succeeded' and r.get('nominated')]
        assert pairs,'no selected ICE path'
        assert all(candidates[p['remoteCandidateId']]['address'] not in {'127.0.0.1','::1'} for p in pairs)
        click('End conversation')
        wait('return document.querySelector("#message").disabled')
        click('Start conversation')
        wait('return document.querySelector("[role=status]").textContent.includes("Listening")')
        wait('return window.__observations.some(e=>e.kind==="microphone_published")')
        click('End conversation')
        wait('return document.querySelector("#message").disabled')
        report={'schema_version':1,'browser':value['capabilities']['browserVersion'],
            'driver':subprocess.check_output([str(driver.resolve()),'--version'],text=True).splitlines()[0],
            'driver_sha256':hashlib.sha256(driver.read_bytes()).hexdigest(),
            'loopback_ice_enabled':False,'profile':'disposable synthetic microphone; real cached LLM/Qwen/LiveKit',
            'forced_loopback_failure_reproduced':True,'retry_after_failure_passed':True,
            'passed':['typed connection','received nonzero audio','interrupt gates output','fresh epoch and replacement audio',
                      'microphone publication','end and new session'],
            'selected_remote_is_non_loopback':True,'selected_protocols':sorted({candidates[p['remoteCandidateId']]['protocol'] for p in pairs}),
            'decoded_audio_observations':len([e for e in events if e['kind']=='decoded_audio_observed']),
            'physical_output_qualified':False,'human_microphone_quality_qualified':False}
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:
        if session:
            try: js('document.querySelectorAll("button").forEach(b=>{if(b.textContent.includes("End conversation"))b.click()})')
            except Exception: pass
            try: request('DELETE','/session/'+session)
            except Exception: pass
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.kill();process.wait()
        log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geckodriver',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.geckodriver,args.output)
