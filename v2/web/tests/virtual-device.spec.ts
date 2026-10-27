import { test, expect } from './virtual-fixtures';
import { spawn, execFileSync } from 'node:child_process';
import { readFile, writeFile, mkdir, unlink } from 'node:fs/promises';
import { createHash, createHmac, randomUUID } from 'node:crypto';
import { createInterface } from 'node:readline';
import { resolve } from 'node:path';

for (const cancel of [false, true]) test(`actual virtual camera/microphone through browser and SFU: ${cancel ? 'cancel/clear' : 'EOF/drain'}`, async ({ page }) => {
  const run = (args: string[]) => execFileSync(args[0], args.slice(1), { encoding: 'utf8', timeout: 5000 }).trim();
  const defaults = () => [run(['pactl', 'get-default-source']), run(['pactl', 'get-default-sink'])];
  const before = defaults(), formatBefore = run(['v4l2-ctl', '--device', '/dev/video10', '--get-fmt-video-out']);
  // Reject any current camera user before opening it. Never use /dev/video0.
  try { execFileSync('fuser', ['/dev/video10'], { encoding: 'utf8', stdio: 'pipe' }); throw new Error('Avatar Camera is already in use'); }
  catch (error) { if (!(error && typeof error === 'object' && 'status' in error && error.status === 1)) throw error; }
  const directory = resolve('../var/p06/virtual'); await mkdir(directory, { recursive: true, mode: 0o700 });
  const name = 'tha_v2_probe_' + randomUUID().replaceAll('-', '').slice(0, 10);
  const description = 'THA_V2_Probe_Microphone_' + name.slice(-10), modules: string[] = [];
  const sourceHashes: Record<string, string> = {};
  for (const path of ['native/virtual/reference.cpp', 'native/virtual/CMakeLists.txt', 'web/tests/virtual-device.spec.ts',
    'web/tests/virtual-fixtures.ts', 'web/playwright.virtual.config.ts', 'web/playwright.virtual.firefox.config.ts']) {
    sourceHashes[path] = createHash('sha256').update(await readFile(resolve('..', path))).digest('hex');
  }
  const binarySha256 = createHash('sha256').update(await readFile(resolve('../var/build/virtual-reference/tha_virtual_reference'))).digest('hex');
  const wave = await readFile(resolve('../var/p06/virtual-input/microphone.wav'));
  const fixture = JSON.parse(await readFile(resolve('../var/p06/virtual-input/microphone-fixture.json'), 'utf8'));
  expect(fixture.kind).toBe('synthetic_microphone_fixture');
  expect(createHash('sha256').update(wave).digest('hex')).toBe(fixture.sha256);
  expect(wave.toString('ascii', 0, 4)).toBe('RIFF');
  let pcm: Buffer | undefined, formatQualified = false;
  for (let offset = 12; offset + 8 <= wave.length;) {
    const length = wave.readUInt32LE(offset + 4);
    if (wave.toString('ascii', offset, offset + 4) === 'fmt ') formatQualified = wave.readUInt16LE(offset + 8) === 1 &&
      wave.readUInt16LE(offset + 10) === 1 && wave.readUInt32LE(offset + 12) === 48000 && wave.readUInt16LE(offset + 22) === 16;
    if (wave.toString('ascii', offset, offset + 4) === 'data') pcm = wave.subarray(offset + 8, offset + 8 + Math.min(length, 8 * 48000 * 2));
    offset += 8 + length + (length % 2);
  }
  expect(pcm?.length).toBe(8 * 48000 * 2);
  expect(formatQualified).toBe(true);
  const pcmSha256 = createHash('sha256').update(pcm!).digest('hex');
  const input = resolve(directory, `${name}.pcm`); await writeFile(input, pcm!, { mode: 0o600 });
  let child: ReturnType<typeof spawn> | undefined, exit: number | null = null, stderr = '';
  const events: Record<string, any>[] = [], caseName = cancel ? 'cancel' : 'eof';
  try {
    modules.push(run(['pactl', 'load-module', 'module-null-sink', `sink_name=${name}`, 'rate=48000', 'channels=1', 'sink_properties=device.description=THA_V2_Probe_Sink priority.session=1']));
    modules.push(run(['pactl', 'load-module', 'module-remap-source', `master=${name}.monitor`, `source_name=${name}_mic`,
      'channels=1', 'channel_map=mono', 'master_channel_map=mono', 'remix=no', `source_properties=device.description=${description} priority.session=1`]));
    expect(defaults()).toEqual(before);
    child = spawn(resolve('../var/build/virtual-reference/tha_virtual_reference'), ['/dev/video10', name, input], { stdio: ['pipe', 'pipe', 'pipe'] });
    const lines = createInterface({ input: child.stdout! });
    lines.on('line', line => { if (line.startsWith('{')) events.push(JSON.parse(line)); });
    child.stderr!.on('data', data => { stderr += data.toString(); });
    child.on('exit', code => { exit = code; });
    child.on('error', error => { stderr += error.message; exit = -1; });
    await expect.poll(() => events.some(event => event.kind === 'ready') || exit !== null, { timeout: 10000 }).toBe(true);
    expect(exit).toBeNull();
    const config = JSON.parse(await readFile(resolve('../var/p04/runtime.json'), 'utf8')), room = 'tha-virtual-' + randomUUID();
    function token(identity: string, publish: boolean) {
      const encode = (value: unknown) => Buffer.from(JSON.stringify(value)).toString('base64url'), now = Math.floor(Date.now() / 1000);
      const body = `${encode({ alg: 'HS256', typ: 'JWT' })}.${encode({ iss: config.livekit_key, sub: identity, nbf: now - 5, exp: now + 120,
        video: { room, roomJoin: true, canPublish: publish, canSubscribe: !publish, canPublishData: false } })}`;
      return body + '.' + createHmac('sha256', config.livekit_secret).update(body).digest('base64url');
    }
    await page.goto('/'); await page.addScriptTag({ path: resolve('node_modules/livekit-client/dist/livekit-client.umd.js') });
    await writeFile(resolve(directory, `${caseName}-devices.json`), JSON.stringify(await page.evaluate(async () =>
      (await navigator.mediaDevices.enumerateDevices()).map(device => ({ kind: device.kind, label: device.label }))), null, 2), { mode: 0o600 });
    await page.evaluate(async credentials => {
      const target = window as any, client = target.LivekitClient;
      client.setLogLevel('warn');
      const devices = await navigator.mediaDevices.enumerateDevices();
      const camera = devices.find(device => device.kind === 'videoinput' && device.label.startsWith('Avatar Camera'));
      const microphone = devices.find(device => device.kind === 'audioinput' && device.label === credentials.description);
      if (!camera || !microphone) throw new Error('Exact project virtual devices missing; no default/physical input fallback');
      const stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: camera.deviceId }, width: { exact: 640 }, height: { exact: 480 }, frameRate: { ideal: 30 } },
        audio: { deviceId: { exact: microphone.deviceId }, channelCount: { exact: 1 }, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
      const publisher = new client.Room({ adaptiveStream: false, dynacast: false }), receiver = new client.Room({ adaptiveStream: false, dynacast: false });
      const context = new AudioContext(), audio = document.createElement('audio'), video = document.createElement('video');
      audio.muted = true; video.muted = true; video.autoplay = true; video.width = 640; video.height = 480;
      const panel = document.createElement('section'); panel.style.cssText = 'position:fixed;top:0;left:0;z-index:10000;background:#111;padding:12px';
      panel.append(audio, video); document.body.append(panel);
      const state: any = target.__virtual = { publisher, receiver, context, stream, audio, video, frames: 0, nonblack: 0, audioPeak: 0, subscribed: 0, colors: [],
        localPeak: 0, microphoneSettings: stream.getAudioTracks()[0].getSettings(), microphoneLabel: stream.getAudioTracks()[0].label,
        exactDevices: stream.getAudioTracks()[0].getSettings().deviceId === microphone.deviceId && stream.getVideoTracks()[0].getSettings().deviceId === camera.deviceId,
        cameraSettings: stream.getVideoTracks()[0].getSettings() };
      const localSource = context.createMediaStreamSource(new MediaStream(stream.getAudioTracks()));
      const localAnalyser = context.createAnalyser(); localAnalyser.fftSize = 1024;
      const localSilent = context.createGain(); localSilent.gain.value = 0;
      localSource.connect(localAnalyser); localAnalyser.connect(localSilent); localSilent.connect(context.destination);
      state.localNodes = [localSource, localAnalyser, localSilent]; const localValues = new Float32Array(1024);
      state.localTimer = setInterval(() => { localAnalyser.getFloatTimeDomainData(localValues); state.localPeak = Math.max(state.localPeak, ...localValues.map(value => Math.abs(value))); }, 10);
      receiver.on(client.RoomEvent.TrackSubscribed, (track: any) => {
        state.subscribed++;
        if (track.kind === 'video') {
          track.attach(video); const canvas = document.createElement('canvas'); canvas.width = 1; canvas.height = 1; const pixels = canvas.getContext('2d', { willReadFrequently: true })!;
          const frame = () => { state.frames++; pixels.drawImage(video, Math.floor(video.videoWidth / 2), Math.floor(video.videoHeight / 2), 1, 1, 0, 0, 1, 1); const rgb = [...pixels.getImageData(0, 0, 1, 1).data].slice(0, 3);
            if (rgb.some(value => value > 30)) state.nonblack++; if (state.colors.length < 600) state.colors.push(rgb); state.callback = video.requestVideoFrameCallback(frame); };
          state.callback = video.requestVideoFrameCallback(frame);
        } else {
          track.attach(audio); const analyser = context.createAnalyser(); analyser.fftSize = 1024;
          const received = new MediaStream([track.mediaStreamTrack]);
          const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/ogg;codecs=opus';
          state.recording = new MediaRecorder(received, { mimeType }); state.recorded = [];
          state.recording.ondataavailable = (event: BlobEvent) => { if (event.data.size) state.recorded.push(event.data); }; state.recording.start(100);
          const source = context.createMediaStreamSource(received); source.connect(analyser);
          const silent = context.createGain(); silent.gain.value = 0; analyser.connect(silent); silent.connect(context.destination);
          state.audioNodes = [source, analyser, silent]; const values = new Float32Array(1024);
          state.timer = setInterval(() => { analyser.getFloatTimeDomainData(values); state.audioPeak = Math.max(state.audioPeak, ...values.map(value => Math.abs(value))); }, 10);
        }
      });
      await receiver.connect(credentials.url, credentials.receiver); await publisher.connect(credentials.url, credentials.publisher, { autoSubscribe: false });
      for (const track of stream.getTracks()) await publisher.localParticipant.publishTrack(track, { source: track.kind === 'video' ? client.Track.Source.Camera : client.Track.Source.Microphone, simulcast: false });
      const unlock = document.createElement('button'); unlock.id = 'virtual-start'; unlock.textContent = 'Start virtual reference';
      unlock.onclick = () => { void context.resume(); void video.play(); void audio.play(); }; panel.append(unlock);
    }, { url: config.livekit_url, publisher: token('virtual-publisher', true), receiver: token('virtual-viewer', false), description });
    await expect.poll(() => page.evaluate(() => (window as any).__virtual.subscribed)).toBe(2);
    await page.locator('#virtual-start').click(); child.stdin!.write('start\n');
    await expect.poll(() => page.evaluate(() => (window as any).__virtual.nonblack), { timeout: 5000 }).toBeGreaterThan(10);
    await expect.poll(() => page.evaluate(() => (window as any).__virtual.audioPeak), { timeout: 8000 }).toBeGreaterThan(.001);
    if (cancel) child.kill('SIGTERM');
    await expect.poll(() => events.some(event => event.kind === 'sender_complete') || exit !== null, { timeout: 12000 }).toBe(true);
    expect(exit).toBeNull();
    // Keep the camera producer open until the actual receiver observes black.
    await expect.poll(() => page.evaluate(() => { const colors = (window as any).__virtual.colors; return colors.length > 0 && colors.at(-1).every((value: number) => value < 20); }), { timeout: 4000 }).toBe(true);
    child.stdin!.write('finish\n');
    await expect.poll(() => exit, { timeout: 5000 }).toBe(0);
    if (!cancel) {
      const recording = await page.evaluate(async () => { const state = (window as any).__virtual;
        await new Promise<void>(resolve => { state.recording.onstop = () => resolve(); state.recording.stop(); });
        return [...new Uint8Array(await new Blob(state.recorded).arrayBuffer())]; });
      await writeFile(resolve(directory, 'received-synthetic-speech.media'), Buffer.from(recording), { mode: 0o600 });
    }
    const result = await page.evaluate(() => { const state = (window as any).__virtual;
      return { frames: state.frames, nonblack: state.nonblack, audioPeak: state.audioPeak, exactDevices: state.exactDevices,
        width: state.cameraSettings.width, height: state.cameraSettings.height, colors: state.colors }; });
    expect(result.exactDevices).toBe(true); expect(result.width).toBe(640); expect(result.height).toBe(480);
    const finished = events.find(event => event.kind === 'complete'); expect(finished?.camera_cleared).toBe(true); expect(finished?.cancelled).toBe(cancel);
    if (!cancel) { expect(finished?.valid_samples).toBe(384000); expect(finished?.video_frames).toBe(240); expect(result.nonblack).toBeGreaterThan(150); }
    // The loopback device retains frames after producer close. The final frame
    // must become black in the actual conference receiver, not just in a log.
    await writeFile(resolve(directory, `${caseName}.json`), JSON.stringify({ source: finished, receiver: result, fixture_kind: fixture.kind, pcm_sha256: pcmSha256,
      source_sha256_at_run: sourceHashes, runtime_binary_sha256: binarySha256,
      browser_user_agent: await page.evaluate(() => navigator.userAgent),
      defaults_unchanged: defaults().every((value, index) => value === before[index]),
      raw_cpu_camera_copy: true, camera_format_unchanged: run(['v4l2-ctl', '--device', '/dev/video10', '--get-fmt-video-out']) === formatBefore,
      conference_target: 'local browser LiveKit qualification room', third_party_conference_qualified: false, physical_av_qualified: false }, null, 2));
  } finally {
    await writeFile(resolve(directory, `${caseName}-browser-state.json`), JSON.stringify(await page.evaluate(async () => {
      const state = (window as any).__virtual; if (!state) return null;
      const stats = [];
      for (const room of [state.publisher, state.receiver]) for (const participant of [room.localParticipant, ...room.remoteParticipants.values()])
        for (const publication of participant.trackPublications.values()) if (publication.track?.getRTCStatsReport) {
          const report = await publication.track.getRTCStatsReport(); if (report) stats.push(...[...report.values()].filter(item => ['inbound-rtp', 'outbound-rtp', 'codec'].includes(item.type)));
        }
      return { frames: state.frames, nonblack: state.nonblack, audioPeak: state.audioPeak, colors: state.colors,
        localPeak: state.localPeak, microphoneSettings: state.microphoneSettings, microphoneLabel: state.microphoneLabel, exactDevices: state.exactDevices,
        video: { width: state.video.videoWidth, height: state.video.videoHeight, paused: state.video.paused, readyState: state.video.readyState }, stats };
    }).catch(() => null), null, 2), { mode: 0o600 });
    await page.evaluate(async () => { const state = (window as any).__virtual; if (!state) return;
      clearInterval(state.timer); clearInterval(state.localTimer); state.video.cancelVideoFrameCallback(state.callback); state.stream.getTracks().forEach((track: MediaStreamTrack) => track.stop());
      if (state.recording?.state !== 'inactive') state.recording?.stop();
      await state.publisher.disconnect(); await state.receiver.disconnect(); await state.context.close(); }).catch(() => {});
    if (child && exit === null) { child.stdin!.end(); child.kill('SIGTERM'); await new Promise(resolve => setTimeout(resolve, 500)); if (exit === null) child.kill('SIGKILL'); }
    const cleanupErrors: string[] = [];
    for (const id of modules.reverse()) {
      try { run(['pactl', 'unload-module', id]); } catch { cleanupErrors.push(id); }
    }
    await unlink(input); await writeFile(resolve(directory, `${caseName}-events.json`), JSON.stringify(events, null, 2));
    await writeFile(resolve(directory, `${caseName}-stderr.log`), stderr, { mode: 0o600 });
    expect(defaults()).toEqual(before); expect(run(['v4l2-ctl', '--device', '/dev/video10', '--get-fmt-video-out'])).toBe(formatBefore);
    expect(cleanupErrors).toEqual([]);
  }
});
