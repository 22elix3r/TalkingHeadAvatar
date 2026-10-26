import { test, expect, type Page } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

async function observations(page: Page) {
  await page.addInitScript(() => {
    const target = window as unknown as { __observations: Record<string, unknown>[] };
    target.__observations = [];
    window.addEventListener('tha-observation', e => target.__observations.push((e as CustomEvent).detail));
  });
  await page.addInitScript(() => {
    const records: { recorder: MediaRecorder; chunks: Blob[] }[] = [];
    const target = window as unknown as { __finishRecording: () => Promise<number[]> };
    window.addEventListener('tha-observation', e => {
      if ((e as CustomEvent).detail.kind !== 'decoded_audio_observed') return;
      const stream = document.querySelector('audio')?.srcObject;
      if (!(stream instanceof MediaStream)) return;
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      const row = { recorder, chunks: [] as Blob[] }; records.push(row);
      recorder.ondataavailable = event => { if (event.data.size) row.chunks.push(event.data); };
      recorder.start(100);
    });
    target.__finishRecording = async () => {
      await Promise.all(records.map(row => new Promise<void>(resolve => {
        if (row.recorder.state === 'inactive') { resolve(); return; }
        row.recorder.onstop = () => resolve(); row.recorder.stop();
      })));
      const data = await new Blob(records.flatMap(row => row.chunks)).arrayBuffer();
      return [...new Uint8Array(data)];
    };
  });
}

test('real typed LLM/Qwen/LiveKit audio and replacement-track interruption', async ({ page, context }) => {
  await observations(page);
  let sid = '', csrf = '';
  page.on('response', async response => {
    if (response.url().endsWith('/v1/dev-login')) csrf = (await response.json()).csrf_token;
    if (response.url().endsWith('/v1/sessions') && response.status() === 202) sid = (await response.json()).id;
  });
  try {
    await page.goto('/');
    await page.getByRole('button', { name: 'Use typed messages instead' }).click();
    await expect(page.getByText('Ready for a message', { exact: true })).toBeVisible({ timeout: 45000 });
    await page.getByLabel('Your message').fill('What color is a clear daytime sky?');
    await page.getByRole('button', { name: 'Send message' }).click();
    await expect.poll(() => page.evaluate(() => (window as unknown as { __observations: { kind: string }[] }).__observations.filter(e => e.kind === 'decoded_audio_observed').length), { timeout: 60000 }).toBeGreaterThan(0);
    await page.getByRole('button', { name: 'Interrupt', exact: true }).click();
    await expect.poll(() => page.locator('audio').evaluate(e => (e as HTMLAudioElement).muted || (e as HTMLAudioElement).srcObject === null)).toBe(true);
    await page.getByLabel('Your message').fill('Say green and count from one to five.');
    await page.getByRole('button', { name: 'Send message' }).click();
    await expect.poll(() => page.evaluate(() => (window as unknown as { __observations: { kind: string }[] }).__observations.filter(e => e.kind === 'decoded_audio_observed').length), { timeout: 60000 }).toBeGreaterThan(1);
    const events = await page.evaluate(() => (window as unknown as { __observations: Record<string, unknown>[] }).__observations);
    const bindings = events.filter(e => e.kind === 'playback_binding');
    expect(bindings.length).toBeGreaterThan(1);
    expect(bindings[0].audio_sid).not.toBe(bindings[bindings.length - 1].audio_sid);
    expect(Number(bindings[bindings.length - 1].epoch)).toBeGreaterThan(Number(bindings[0].epoch));
    await page.screenshot({ path: resolve('../var/p04/browser-typed.png'), fullPage: true });
    await writeFile(resolve('../var/p04/browser-typed-events.json'), JSON.stringify(events, null, 2));
    await page.getByRole('button', { name: 'End conversation' }).click();
    await expect(page.getByRole('button', { name: 'Start conversation', exact: false })).toBeVisible();
  } finally {
    if (sid && csrf) await context.request.post(`/v1/sessions/${sid}/close`, { data: {}, headers: { Origin: 'http://127.0.0.1:18700', 'X-CSRF-Token': csrf } });
  }
});

test('synthetic microphone through real CPU ASR, LLM, Qwen and browser receiver', async ({ page, context }) => {
  await observations(page);
  let sid = '', csrf = '';
  page.on('response', async response => {
    if (response.url().endsWith('/v1/dev-login')) csrf = (await response.json()).csrf_token;
    if (response.url().endsWith('/v1/sessions') && response.status() === 202) sid = (await response.json()).id;
  });
  try {
    await page.goto('/'); await page.getByRole('button', { name: 'Start conversation' }).click();
    await expect(page.getByText('Listening', { exact: true })).toBeVisible({ timeout: 45000 });
    await expect(page.locator('.caption.user')).toContainText(/two|2/i, { timeout: 45000 });
    await expect.poll(() => page.evaluate(() => (window as unknown as { __observations: { kind: string }[] }).__observations.some(e => e.kind === 'decoded_audio_observed')), { timeout: 60000 }).toBe(true);
    await expect(page.locator('.caption.assistant').filter({ hasText: /four|4/i }).first()).toBeVisible({ timeout: 20000 });
    await expect.poll(() => page.evaluate(() => (window as unknown as { __observations: { kind: string }[] }).__observations.some(e => e.kind === 'sender_complete')), { timeout: 45000 }).toBe(true);
    await page.waitForTimeout(500); // Explicit receiver-tail capture, not a latency measurement.
    const recording = await page.evaluate(() => (window as unknown as { __finishRecording: () => Promise<number[]> }).__finishRecording());
    await writeFile(resolve('../var/p04/browser-microphone-answer.webm'), Buffer.from(recording));
    await page.screenshot({ path: resolve('../var/p04/browser-microphone.png'), fullPage: true });
    await writeFile(resolve('../var/p04/browser-microphone-events.json'), JSON.stringify(await page.evaluate(() => (window as unknown as { __observations: unknown[] }).__observations), null, 2));
    await page.getByRole('checkbox', { name: 'Mute microphone' }).check();
    await expect(page.getByRole('checkbox', { name: 'Mute microphone' })).toBeChecked();
    await page.getByRole('button', { name: 'End conversation' }).click();
  } finally {
    if (sid && csrf) await context.request.post(`/v1/sessions/${sid}/close`, { data: {}, headers: { Origin: 'http://127.0.0.1:18700', 'X-CSRF-Token': csrf } });
  }
});

test('autoplay failure has a user recovery and disconnect detaches old output', async ({ page, context }) => {
  let sid = '', csrf = '';
  page.on('response', async response => {
    if (response.url().endsWith('/v1/dev-login')) csrf = (await response.json()).csrf_token;
    if (response.url().endsWith('/v1/sessions') && response.status() === 202) sid = (await response.json()).id;
  });
  await page.addInitScript(() => {
    const original = HTMLMediaElement.prototype.play;
    const state = window as unknown as { __blockAudio: boolean }; state.__blockAudio = true;
    HTMLMediaElement.prototype.play = function () {
      if (this instanceof HTMLAudioElement && state.__blockAudio) return Promise.reject(new DOMException('explicit autoplay fault fixture', 'NotAllowedError'));
      return original.call(this);
    };
  });
  try {
    await page.goto('/'); await page.getByRole('button', { name: 'Use typed messages instead' }).click();
    await expect(page.getByText('Ready for a message', { exact: true })).toBeVisible({ timeout: 45000 });
    await page.getByLabel('Your message').fill('Say hello.'); await page.getByRole('button', { name: 'Send message' }).click();
    await expect(page.getByRole('button', { name: 'Enable audio playback' })).toBeVisible({ timeout: 30000 });
    await page.evaluate(() => { (window as unknown as { __blockAudio: boolean }).__blockAudio = false; });
    await page.getByRole('button', { name: 'Enable audio playback' }).click();
    await expect(page.getByRole('button', { name: 'Enable audio playback' })).toHaveCount(0);
    await context.setOffline(true);
    await expect(page.getByRole('alert')).toContainText(/interrupt|connection/i, { timeout: 15000 });
    await expect.poll(() => page.locator('audio').evaluate(e => (e as HTMLAudioElement).srcObject === null)).toBe(true);
    await context.setOffline(false);
    await expect(page.getByRole('button', { name: 'Start conversation' })).toBeVisible({ timeout: 10000 });
  } finally {
    await context.setOffline(false);
    if (sid && csrf) await context.request.post(`/v1/sessions/${sid}/close`, { data: {}, headers: { Origin: 'http://127.0.0.1:18700', 'X-CSRF-Token': csrf } });
  }
});

test('changing microphone ends the old session before a fresh connection', async ({ page, context }) => {
  let sid = '', csrf = '';
  page.on('response', async response => {
    if (response.url().endsWith('/v1/dev-login')) csrf = (await response.json()).csrf_token;
    if (response.url().endsWith('/v1/sessions') && response.status() === 202) sid = (await response.json()).id;
  });
  try {
    await page.goto('/'); await page.getByRole('button', { name: 'Start conversation' }).click();
    await expect(page.getByText('Listening', { exact: true })).toBeVisible({ timeout: 45000 });
    const device = await page.locator('#microphone option').evaluateAll(options => (options as HTMLOptionElement[]).find(o => o.value !== '')?.value);
    expect(device).toBeTruthy(); await page.getByLabel('Microphone', { exact: true }).selectOption(device!);
    await expect(page.getByText('Device changed. Start a new conversation.', { exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Start conversation' })).toBeEnabled();
  } finally {
    if (sid && csrf) await context.request.post(`/v1/sessions/${sid}/close`, { data: {}, headers: { Origin: 'http://127.0.0.1:18700', 'X-CSRF-Token': csrf } });
  }
});

test('connection can be cancelled while providers prepare', async ({ page }) => {
  await page.route('**/v1/sessions', async route => { await new Promise(resolve => setTimeout(resolve, 300)); await route.continue(); });
  await page.goto('/'); await page.getByRole('button', { name: 'Use typed messages instead' }).click();
  await page.getByRole('button', { name: 'Cancel connection' }).click();
  await expect(page.getByRole('button', { name: 'Start conversation' })).toBeEnabled({ timeout: 15000 });
  await expect.poll(() => page.locator('audio').evaluate(e => (e as HTMLAudioElement).srcObject === null)).toBe(true);
});
