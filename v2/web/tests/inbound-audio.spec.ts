import { test, expect } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

async function evidence(name: string, value: unknown) {
  const cohort = process.env.THA_INBOUND_COHORT ?? 'first';
  if (!/^[a-z0-9-]{1,64}$/.test(cohort)) throw new Error('Private cohort name required');
  const directory = resolve('../var/p06-claire/inbound-browser', cohort);
  await mkdir(directory, { recursive: true, mode: 0o700 });
  await writeFile(resolve(directory, `${name}.json`), JSON.stringify(value, null, 2), { flag: 'wx', mode: 0o600 });
}

for (const microphone of [false, true]) {
  test(`bounded native input: ${microphone ? 'microphone ASR' : 'typed interrupt and resume'}`, async ({ page, context, baseURL, browser }) => {
    await page.addInitScript(() => {
      (window as any).__inbound = [];
      window.addEventListener('tha-observation', event => (window as any).__inbound.push((event as CustomEvent).detail));
    });
    let sid = '', csrf = '';
    page.on('response', async response => {
      if (response.url().endsWith('/v1/dev-login')) csrf = (await response.json()).csrf_token;
      if (response.url().endsWith('/v1/sessions') && response.status() === 202) sid = (await response.json()).id;
    });
    try {
      const ready = await context.request.get('/health/ready');
      expect((await ready.json()).inbound_audio_queue).toBe('native_fail_closed');
      await page.goto('/');
      await page.getByRole('button', { name: microphone ? 'Start conversation' : 'Use typed messages instead' }).click();
      await expect(page.getByText(microphone ? 'Listening' : 'Ready for a message', { exact: true })).toBeVisible({ timeout: 45000 });
      if (microphone) await expect(page.locator('.caption.user')).toContainText(/two|2/i, { timeout: 45000 });
      else {
        await page.getByLabel('Your message').fill('What is two plus two?');
        await page.getByRole('button', { name: 'Send message' }).click();
      }
      const decoded = () => page.evaluate(() => (window as any).__inbound.filter((event: any) => event.kind === 'decoded_audio_observed').length);
      await expect.poll(decoded, { timeout: 60000 }).toBeGreaterThan(0);
      if (!microphone) {
        await page.getByRole('button', { name: 'Interrupt', exact: true }).click();
        await expect.poll(() => page.locator('audio').evaluate(audio => (audio as HTMLAudioElement).srcObject === null || (audio as HTMLAudioElement).muted)).toBe(true);
        await page.getByLabel('Your message').fill('Say hello again.');
        await page.getByRole('button', { name: 'Send message' }).click();
        await expect.poll(decoded, { timeout: 60000 }).toBeGreaterThan(1);
      }
      await expect.poll(() => page.evaluate(() => (window as any).__inbound.some((event: any) => event.kind === 'sender_complete')), { timeout: 45000 }).toBe(true);
      const observations = await page.evaluate(() => (window as any).__inbound);
      expect(observations.some((event: any) => event.kind === 'receiver_recovery_required')).toBe(false);
      await evidence(microphone ? 'microphone' : 'typed', { origin: baseURL, browser_version: browser.version(), observations,
        real_providers: true, input_queue: 'native_fail_closed', synthetic_microphone: microphone, physical_av_qualified: false });
      await page.getByRole('button', { name: 'End conversation' }).click();
      await expect(page.getByRole('button', { name: 'Start conversation' })).toBeVisible();
    } finally {
      if (sid && csrf) await context.request.post(`/v1/sessions/${sid}/close`, { data: {}, headers: { Origin: baseURL!, 'X-CSRF-Token': csrf } });
    }
  });
}

test('unpublished microphone fails the real session and releases admission', async ({ page, context, baseURL }) => {
  let sid = '', csrf = '';
  const failures: { host: string; path: string; reason: string | null }[] = [];
  page.on('requestfailed', request => {
    const url = new URL(request.url());
    failures.push({ host: url.host, path: url.pathname, reason: request.failure()?.errorText ?? null });
  });
  const network = await context.newCDPSession(page);
  await network.send('Network.enable');
  network.on('Network.webSocketHandshakeResponseReceived', event => {
    failures.push({ host: 'local-sfu', path: 'websocket-handshake', reason: String(event.response.status) });
  });
  network.on('Network.loadingFailed', event => {
    failures.push({ host: 'browser-network', path: 'loading-failed', reason: event.corsErrorStatus?.corsError ?? event.blockedReason ?? event.errorText });
  });
  // Use a real local response so Chromium retains its local address-space
  // classification; a fulfilled document triggers Local Network Access denial.
  await page.goto('/health/live');
  try {
    const login = await context.request.post('/v1/dev-login', { data: {}, headers: { Origin: baseURL! } });
    csrf = (await login.json()).csrf_token;
    const headers = { Origin: baseURL!, 'X-CSRF-Token': csrf };
    const create = async () => {
      const response = await context.request.post('/v1/sessions', { data: { profile: 'audio_reference', asset_id: 'synthetic-audio-reference' },
        headers: { ...headers, 'Idempotency-Key': crypto.randomUUID() } });
      expect(response.status()).toBe(202);
      sid = (await response.json()).id;
      await expect.poll(async () => (await (await context.request.get(`/v1/sessions/${sid}`)).json()).status, { timeout: 45000 }).toBe('READY');
    };
    await create();
    const grantResponse = await context.request.post(`/v1/sessions/${sid}/join-grant`, { data: {}, headers });
    expect(grantResponse.status()).toBe(200);
    const grant = await grantResponse.json();
    await page.addScriptTag({ path: resolve('node_modules/livekit-client/dist/livekit-client.umd.js') });
    await page.evaluate(async credentials => {
      const client = (window as any).LivekitClient;
      const room = new client.Room({ adaptiveStream: false, dynacast: false });
      await room.connect(credentials.url, credentials.token, { autoSubscribe: false });
      const audio = new AudioContext();
      const oscillator = audio.createOscillator(), gain = audio.createGain(), destination = audio.createMediaStreamDestination();
      gain.gain.value = .02; oscillator.frequency.value = 440;
      oscillator.connect(gain).connect(destination); oscillator.start();
      const track = destination.stream.getAudioTracks()[0];
      await room.localParticipant.publishTrack(track, { source: client.Track.Source.Microphone });
      (window as any).__microphoneFault = { room, audio, oscillator, track };
    }, grant);
    await page.waitForTimeout(350);
    await page.evaluate(async () => {
      const state = (window as any).__microphoneFault;
      await state.room.localParticipant.unpublishTrack(state.track);
      state.oscillator.stop();
    });
    await expect.poll(async () => (await (await context.request.get(`/v1/sessions/${sid}`)).json()).status, { timeout: 10000 }).toBe('FAILED');
    const failed = await (await context.request.get(`/v1/sessions/${sid}`)).json();
    expect(failed.error_code).toBe('microphone_discontinuity');
    await context.request.post(`/v1/sessions/${sid}/close`, { data: {}, headers });
    await page.evaluate(async () => { const state = (window as any).__microphoneFault; await state.room.disconnect(); await state.audio.close(); });
    await create();
    await evidence('microphone-unpublish', { real_api: true, real_cached_qwen_preparation: true, failed_code: failed.error_code,
      fresh_session_ready: true, synthetic_microphone: true, physical_av_qualified: false });
  } finally {
    if (failures.length) await evidence('eof-network-observations', { observations: failures });
    await page.evaluate(async () => { const state = (window as any).__microphoneFault; if (state) { await state.room.disconnect(); await state.audio.close().catch(() => {}); } }).catch(() => {});
    if (sid && csrf) await context.request.post(`/v1/sessions/${sid}/close`, { data: {}, headers: { Origin: baseURL!, 'X-CSRF-Token': csrf } });
  }
});

test('Firefox synthetic microphone barge-in resumes after mute', async ({ page, context, baseURL, browserName, browser }) => {
  test.skip(browserName !== 'firefox', 'Firefox capture compatibility cohort');
  const idleMs = Number(process.env.THA_INBOUND_IDLE_MS ?? '0');
  if (!Number.isInteger(idleMs) || idleMs < 0 || idleMs > 60000) throw new Error('Idle fixture must be 0..60000 ms');
  let sid = '', csrf = '';
  await page.addInitScript(() => {
    (window as any).__inbound = [];
    (window as any).__inboundServer = [];
    const Original = window.EventSource;
    window.EventSource = class extends Original {
      constructor(url: string | URL, options?: EventSourceInit) {
        super(url, options);
        this.addEventListener('message', event => {
          const data = JSON.parse(event.data);
          (window as any).__inboundServer.push({ kind: data.kind, epoch: data.epoch, reason: data.reason, code: data.code });
        });
      }
    };
    window.addEventListener('tha-observation', event => (window as any).__inbound.push((event as CustomEvent).detail));
  });
  page.on('response', async response => {
    if (response.url().endsWith('/v1/dev-login')) csrf = (await response.json()).csrf_token;
    if (response.url().endsWith('/v1/sessions') && response.status() === 202) sid = (await response.json()).id;
  });
  try {
    const ready = await context.request.get('/health/ready');
    expect((await ready.json()).inbound_audio_queue).toBe('native_fail_closed');
    await page.goto('/');
    await page.getByRole('button', { name: 'Start conversation' }).click();
    await expect(page.getByText('Listening', { exact: true })).toBeVisible({ timeout: 45000 });
    await expect.poll(() => page.evaluate(() => (window as any).__inbound.some((event: any) => event.kind === 'microphone_published'))).toBe(true);
    await page.getByLabel('Your message').fill('What is two plus two?');
    await page.getByRole('button', { name: 'Send message' }).click();
    await expect.poll(() => page.evaluate(() => (window as any).__inboundServer.some((event: any) => event.kind === 'interrupted' && event.reason === 'vad_barge_in')), { timeout: 10000 }).toBe(true);
    await page.getByLabel('Mute microphone').check();
    await expect.poll(() => page.evaluate(() => {
      const events = (window as any).__inboundServer;
      const barge = events.filter((event: any) => event.kind === 'interrupted' && event.reason === 'vad_barge_in').at(-1);
      return barge && events.some((event: any) => event.kind === 'cancel_ack' && event.epoch === barge.epoch);
    })).toBe(true);
    // Deliberately cross multiple SSE and browser-lease heartbeat intervals.
    if (idleMs) await page.waitForTimeout(idleMs);
    await expect(page.getByRole('alert')).toHaveCount(0);
    expect(await page.evaluate(() => (window as any).__inbound.some((event: any) => event.kind === 'receiver_recovery_required'))).toBe(false);
    await page.getByLabel('Your message').fill('Say hello again.');
    const before = await page.evaluate(() => {
      const bindings = (window as any).__inbound.filter((event: any) => event.kind === 'playback_binding');
      return { count: bindings.length, epoch: bindings.at(-1)?.epoch ?? -1 };
    });
    const response = page.waitForResponse(value => value.url().endsWith('/controls') &&
      value.request().method() === 'POST' && value.request().postDataJSON()?.kind === 'text' &&
      value.request().postDataJSON()?.text === 'Say hello again.');
    await page.getByRole('button', { name: 'Send message' }).click();
    const sent = await response;
    expect(sent.ok()).toBe(true);
    const freshEpoch = (await sent.json()).epoch;
    expect(freshEpoch).toBeGreaterThan(before.epoch);
    await expect.poll(() => page.evaluate(({ count, epoch }) => {
      const events = (window as any).__inbound;
      const binding = events.filter((event: any) => event.kind === 'playback_binding').slice(count)
        .find((event: any) => event.epoch === epoch);
      return !!binding && events.some((event: any) => event.kind === 'decoded_audio_observed' &&
        event.epoch === epoch && event.audio_sid === binding.audio_sid);
    }, { count: before.count, epoch: freshEpoch }), { timeout: 60000 }).toBe(true);
    await expect(page.getByRole('button', { name: 'End conversation' })).toBeVisible();
    await expect(page.getByRole('alert')).toHaveCount(0);
    await evidence('firefox-microphone', { browser_version: browser.version(), real_providers: true, synthetic_microphone: true,
      vad_barge_in_observed: true, muted_resume_answer_decoded: true, controlled_idle_ms: idleMs,
      fresh_control_epoch_verified: true,
      input_queue: 'native_fail_closed', asr_quality_qualified: false, physical_av_qualified: false });
    await page.getByRole('button', { name: 'End conversation' }).click();
  } finally {
    await evidence('firefox-microphone-attempt', await page.evaluate(() => ({
      observations: (window as any).__inbound, server_events: (window as any).__inboundServer,
      status: document.querySelector('[role=status]')?.textContent,
      alert: document.querySelector('[role=alert]')?.textContent,
    })));
    if (sid && csrf) await context.request.post(`/v1/sessions/${sid}/close`, { data: {}, headers: { Origin: baseURL!, 'X-CSRF-Token': csrf } });
  }
});
