import { test, expect, type Page } from '@playwright/test';
import { createHmac, randomUUID } from 'node:crypto';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { spawn, execFile } from 'node:child_process';
import { createInterface } from 'node:readline';
import { resolve } from 'node:path';

function cohort(recovery: boolean) { return async ({ page }: { page: Page }) => {
  const directory = resolve('../var/p06'); await mkdir(directory, { recursive: true, mode: 0o700 });
  const caseName = recovery ? 'native-recovery' : 'native-baseline';
  const config = JSON.parse(await readFile(resolve('../var/p04/runtime.json'), 'utf8'));
  const room = `tha-p06-${randomUUID()}`;
  function grant(identity: string, publish: boolean) {
    const now = Math.floor(Date.now() / 1000);
    const encode = (value: unknown) => Buffer.from(JSON.stringify(value)).toString('base64url');
    const body = `${encode({ alg: 'HS256', typ: 'JWT' })}.${encode({ iss: config.livekit_key, sub: identity, nbf: now - 5, exp: now + 120,
      video: { room, roomJoin: true, canPublish: publish, canSubscribe: !publish, canPublishData: false,
        ...(publish ? { canPublishSources: ['microphone', 'camera'] } : {}) } })}`;
    return `${body}.${createHmac('sha256', config.livekit_secret).update(body).digest('base64url')}`;
  }
  const privateGrant = resolve(directory, 'native-grant.txt');
  await writeFile(privateGrant, `${config.livekit_url}\n${grant('native-publisher', true)}\n`, { mode: 0o600 });
  const child = spawn(resolve('../var/build/native-media/tha_media_publisher'), [privateGrant,
    resolve('../var/build/native-media/media_kernels.ptx'), '8'], { stdio: ['pipe', 'pipe', 'pipe'] });
  const events: Record<string, unknown>[] = []; let stderr = '', nativeExit: number | null = null;
  const browserLogs: string[] = [];
  page.on('console', message => browserLogs.push(message.type() + ': ' + message.text()));
  page.on('pageerror', error => browserLogs.push(error.message));
  const memoryMiB: number[] = []; let sampling = false;
  const sampler = setInterval(() => {
    if (sampling || nativeExit !== null) return; sampling = true;
    execFile('nvidia-smi', ['--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], { timeout: 1000 }, (error, stdout) => {
      sampling = false; if (error) return;
      for (const line of stdout.trim().split('\n')) { const [pid, memory] = line.split(',').map(Number); if (pid === child.pid && Number.isFinite(memory)) memoryMiB.push(memory); }
    });
  }, 200);
  const lines = createInterface({ input: child.stdout });
  lines.on('line', line => { if (line.startsWith('{')) { try { events.push(JSON.parse(line)); } catch { /* Private SDK diagnostics are not events. */ } } });
  child.stderr.on('data', data => { stderr = (stderr + data.toString()).slice(-256 * 1024); });
  child.on('exit', code => { nativeExit = code; });
  try {
    await expect.poll(() => events.some(e => e.kind === 'ready') || nativeExit !== null, { timeout: 30000 }).toBe(true);
    expect(nativeExit).toBeNull();
    await page.goto('/');
    await page.addScriptTag({ path: resolve('node_modules/livekit-client/dist/livekit-client.umd.js') });
    const workerSource = await readFile(resolve('node_modules/livekit-client/dist/livekit-client.fm.worker.js'), 'utf8');
    await page.evaluate(async credentials => {
      const target = window as unknown as { LivekitClient: any; __native: any };
      const client = target.LivekitClient;
      const workerURL = URL.createObjectURL(new Blob([credentials.workerSource], { type: 'text/javascript' }));
      const worker = new Worker(workerURL); URL.revokeObjectURL(workerURL);
      const audio = document.createElement('audio'), video = document.createElement('video');
      audio.id = 'native-audio'; video.id = 'native-video'; video.muted = true; video.autoplay = true;
      video.width = 512; video.height = 512;
      const panel = document.createElement('section'); panel.id = 'native-qualification';
      panel.style.cssText = 'position:fixed;top:16px;left:16px;z-index:10000;background:#111923;color:white;padding:16px;border:1px solid #80988a';
      const title = document.createElement('p'); title.textContent = 'Native synthetic media qualification';
      panel.append(title, audio, video); document.body.append(panel);
      const state: { room: any; subscribed: number; videoFrames: number; decodedAudio: boolean;
        mediaTimes: { generation: number; mediaTime: number; rtpTimestamp?: number; frameId?: number; sourcePTS?: string }[]; audio: HTMLAudioElement; video: HTMLVideoElement;
        context?: AudioContext; timer?: ReturnType<typeof setInterval>; videoCallback?: number; videoGeneration?: number } =
        target.__native = { room: new client.Room({ adaptiveStream: false, dynacast: false, frameMetadata: { worker } }),
          subscribed: 0, videoFrames: 0, decodedAudio: false, mediaTimes: [], audio, video };
      const context = new AudioContext(); state.context = context;
      let timer: ReturnType<typeof setInterval>;
      state.room.on(client.RoomEvent.TrackSubscribed, (track: any, _publication: any, participant: any) => {
        if (participant.identity !== 'native-publisher') return;
        state.subscribed++;
        if (track.kind === 'audio') {
          // Pinned JS SDK enables insertable streams for the entire Chromium
          // receiver PC when metadata is requested, but installs only a video
          // transform. Explicit identity audio forwarding prevents a stalled
          // audio decoder. This compatibility workaround is not an A/V ack.
          if (typeof track.receiver?.createEncodedStreams === 'function') {
            const encoded = track.receiver.createEncodedStreams();
            void encoded.readable.pipeTo(encoded.writable).catch(() => {});
          }
          track.attach(audio);
          const analyser = context.createAnalyser(); analyser.fftSize = 1024;
          context.createMediaStreamSource(new MediaStream([track.mediaStreamTrack])).connect(analyser);
          const values = new Float32Array(1024);
          timer = setInterval(() => { analyser.getFloatTimeDomainData(values); if (values.some(v => Math.abs(v) > .001)) state.decodedAudio = true; }, 10);
          state.timer = timer;
        } else {
          track.attach(video);
          if (state.videoCallback) video.cancelVideoFrameCallback(state.videoCallback);
          const generation = (state.videoGeneration ?? 0) + 1; state.videoGeneration = generation;
          const presented = (_now: number, metadata: VideoFrameCallbackMetadata) => {
            if (generation !== state.videoGeneration) return;
            const rtpTimestamp = (metadata as VideoFrameCallbackMetadata & { rtpTimestamp?: number }).rtpTimestamp;
            const tags = rtpTimestamp === undefined ? undefined : track.lookupFrameMetadata({ rtpTimestamp });
            state.videoFrames++; state.mediaTimes.push({ generation, mediaTime: metadata.mediaTime,
              rtpTimestamp, frameId: tags?.frameId, sourcePTS: tags?.userTimestamp?.toString() });
            state.videoCallback = video.requestVideoFrameCallback(presented);
          };
          state.videoCallback = video.requestVideoFrameCallback(presented);
        }
      });
      await state.room.connect(credentials.url, credentials.token);
      const unlock = document.createElement('button'); unlock.id = 'native-unlock'; unlock.textContent = 'Start native qualification';
      unlock.onclick = () => { void context.resume(); void state.room.startAudio(); void audio.play(); };
      panel.append(unlock);
    }, { url: config.livekit_url, token: grant('native-viewer', false), workerSource });
    await expect.poll(() => page.evaluate(() => (window as any).__native.subscribed), { timeout: 20000 }).toBe(2);
    await page.locator('#native-unlock').click(); child.stdin.write('start\n');
    await expect.poll(() => page.evaluate(() => (window as any).__native.decodedAudio), { timeout: 15000 }).toBe(true);
    await expect.poll(() => page.evaluate(() => (window as any).__native.videoFrames), { timeout: 15000 }).toBeGreaterThan(20);
    // Reattachment requires a recovery keyframe from the external encoder.
    if (recovery) {
      await page.evaluate(() => { const room = (window as any).__native.room; room.remoteParticipants.forEach((p: any) => p.videoTrackPublications.forEach((t: any) => t.setSubscribed(false))); });
      await page.waitForTimeout(250);
      await page.evaluate(() => { const room = (window as any).__native.room; room.remoteParticipants.forEach((p: any) => p.videoTrackPublications.forEach((t: any) => t.setSubscribed(true))); });
      await expect.poll(() => page.evaluate(() => (window as any).__native.videoGeneration), { timeout: 10000 }).toBe(2);
    }
    await expect.poll(() => events.some(e => e.kind === 'sender_complete') || nativeExit !== null, { timeout: 15000 }).toBe(true);
    expect(events.some(e => e.kind === 'sender_complete')).toBe(true);
    await page.waitForTimeout(500); // Receiver drain observation, not an acoustic acknowledgment.
    const receiver = await page.evaluate(async () => {
      const s = (window as any).__native;
      const stats: unknown[] = [];
      for (const participant of s.room.remoteParticipants.values()) for (const publication of participant.trackPublications.values()) {
        if (publication.track) stats.push(...await publication.track.getRTCStatsReport().then((r?: RTCStatsReport) => r ? [...r.values()].filter(x => ['codec', 'inbound-rtp'].includes(x.type)) : []));
      }
      return { decodedAudio: s.decodedAudio, videoFrames: s.videoFrames, mediaTimes: s.mediaTimes, width: s.video.videoWidth, height: s.video.videoHeight, stats };
    });
    expect(receiver.width).toBe(512); expect(receiver.height).toBe(512);
    await page.screenshot({ path: resolve(directory, `${caseName}.png`), fullPage: true });
    child.stdin.write('finish\n');
    await expect.poll(() => events.some(e => e.kind === 'complete') || nativeExit !== null, { timeout: 15000 }).toBe(true);
    const finished = events.find(e => e.kind === 'complete'); expect(finished).toBeTruthy();
    expect(finished!.video_frames).toBe(240); expect(finished!.audio_valid_samples).toBe(384000);
    expect(Number(finished!.feedback_keyframe_requests)).toBeGreaterThan(0);
    expect(Number(finished!.rate_updates)).toBeGreaterThan(0);
    await expect.poll(() => nativeExit, { timeout: 10000 }).toBe(0);
    await writeFile(resolve(directory, `${caseName}-evidence.json`), JSON.stringify({ native: finished, receiver,
      process_memory: { samples: memoryMiB.length, sampled_peak_mib: memoryMiB.length ? Math.max(...memoryMiB) : null,
        poll_ms: 200, loading_peak_qualified: false, qwen_motion_coload_qualified: false },
      cohort: caseName, synthetic: true, speaking_avatar_qualified: false, physical_av_skew_qualified: false }, null, 2));
    if (!recovery) {
      const ids = receiver.mediaTimes.filter((m: any) => m.frameId >= 30 && m.frameId < 270).map((m: any) => m.frameId);
      expect(new Set(ids).size).toBeGreaterThanOrEqual(237); // Actual valid frames; preparation cannot inflate this diagnostic.
      const pts = receiver.mediaTimes.filter((m: any) => m.sourcePTS && m.frameId >= 30);
      expect(pts.length).toBeGreaterThanOrEqual(237);
      for (let i = 1; i < pts.length; i++) {
        const delta = BigInt(pts[i].sourcePTS!) - BigInt(pts[i - 1].sourcePTS!);
        expect(Number(delta)).toBeGreaterThanOrEqual((pts[i].frameId! - pts[i - 1].frameId!) * 33333);
      }
    } else expect(receiver.mediaTimes.filter((m: any) => m.generation === 2).length).toBeGreaterThan(20);
  } finally {
    clearInterval(sampler);
    child.stdin.end(); if (nativeExit === null) child.kill('SIGTERM');
    lines.close(); await writeFile(resolve(directory, `${caseName}-events.json`), JSON.stringify(events, null, 2), { mode: 0o600 });
    await writeFile(resolve(directory, `${caseName}-stderr.log`), stderr, { mode: 0o600 });
    await writeFile(resolve(directory, `${caseName}-browser.log`), browserLogs.join('\n'), { mode: 0o600 });
    await page.evaluate(async () => { const s = (window as any).__native; if (s) { clearInterval(s.timer); await s.room.disconnect(); await s.context.close(); } }).catch(() => {});
  }
}; }

test('prepared CUDA/NVENC stream presents tagged valid frames at the source cadence', cohort(false));
test('external encoder keyframe/rate feedback recovers after deliberate resubscription', cohort(true));
