import { test, expect } from '@playwright/test';
import { mkdir, writeFile, appendFile } from 'node:fs/promises';
import { resolve } from 'node:path';

test('Claire typed session sustains completion, interruption and fresh playback', async ({ page, context, baseURL, browser }) => {
  const seconds = Number(process.env.THA_CLAIRE_SOAK_SECONDS ?? '60');
  const cohort = process.env.THA_CLAIRE_COHORT;
  if (![60, 1800].includes(seconds) || !cohort || !/^[a-z0-9-]{1,64}$/.test(cohort)) throw new Error('Explicit bounded soak cohort required');
  const parent = resolve('../var/p06-claire/soak');
  await mkdir(parent, { recursive: true, mode: 0o700 });
  const directory = resolve(parent, cohort);
  await mkdir(directory, { mode: 0o700 });
  const progress = resolve(directory, 'turns.jsonl');
  await writeFile(progress, '', { flag: 'wx', mode: 0o600 });
  await page.addInitScript(() => {
    const state = (window as any).__claireSoak = { events: [] as any[], overflow: 0, recovery: 0 };
    window.addEventListener('tha-observation', event => {
      const row = (event as CustomEvent).detail;
      if (['receiver_recovery_required', 'reconnecting'].includes(row.kind)) state.recovery++;
      if (state.events.length >= 256) state.overflow++;
      else state.events.push(row);
    });
  });
  let sid = '', csrf = '', started = 0, passed = false, closeConfirmed = false;
  let completed = 0, interrupted = 0;
  const rows: any[] = [];
  let attempt: Record<string, unknown> | undefined;
  const controls: unknown[] = [];
  page.on('response', async response => {
    if (response.url().endsWith('/v1/dev-login')) csrf = (await response.json()).csrf_token;
    if (response.url().endsWith('/v1/sessions') && response.status() === 202) sid = (await response.json()).id;
  });
  const snapshot = () => page.evaluate(() => {
    const state = (window as any).__claireSoak;
    if (document.querySelector('[role="alert"]') || state.overflow || state.recovery) throw new Error('Soak receiver/control failure');
    return state.events;
  });
  async function control(kind: 'text' | 'interrupt', text?: string) {
    const pending = page.waitForResponse(response => response.url().endsWith('/controls') && response.request().postDataJSON()?.kind === kind);
    if (kind === 'text') {
      await page.getByLabel('Your message').fill(text!);
      await page.getByRole('button', { name: 'Send message' }).click();
    } else await page.getByRole('button', { name: 'Interrupt', exact: true }).click();
    const response = await pending;
    expect(response.status()).toBe(200);
    const body = await response.json();
    expect(Number.isSafeInteger(body.epoch)).toBe(true);
    expect(body.receiver_safe_resume).toBe(false);
    controls.push({ kind, epoch: body.epoch, sequence: body.sequence });
    return body.epoch as number;
  }
  try {
    await page.goto('/');
    await expect(page.getByText('Claire research preview', { exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Use typed messages instead' }).click();
    await expect(page.getByText('Ready for a message', { exact: true })).toBeVisible({ timeout: 45000 });
    started = performance.now();
    let previous: any;
    while (performance.now() - started < seconds * 1000) {
      if (rows.length >= 1024) throw new Error('Soak turn bound exceeded');
      await page.evaluate(() => { (window as any).__claireSoak.events = []; });
      const index = rows.length, cancel = index % 5 === 0;
      attempt = { index, intended: cancel ? 'interrupt_after_decoded_audio' : 'complete', node_started_ms: performance.now() - started };
      const epoch = await control('text', index % 2 ? 'What is two plus two?' : 'Describe a calm morning in twelve words.');
      attempt.epoch = epoch;
      await expect.poll(async () => (await snapshot()).some((e: any) => e.kind === 'playback_binding' && e.epoch === epoch), { timeout: 45000 }).toBe(true);
      const binding = (await snapshot()).find((e: any) => e.kind === 'playback_binding' && e.epoch === epoch);
      expect(binding.visual_mode).toBe('claire_vendor_research');
      if (previous) {
        expect(binding.audio_sid).not.toBe(previous.audio_sid);
        expect(binding.video_sid).not.toBe(previous.video_sid);
      }
      await expect.poll(async () => (await snapshot()).some((e: any) => e.kind === 'decoded_audio_observed' && e.epoch === epoch && e.audio_sid === binding.audio_sid), { timeout: 60000 }).toBe(true);
      await expect.poll(async () => (await snapshot()).some((e: any) => e.kind === 'video_presented_callback' && e.epoch === epoch && e.video_sid === binding.video_sid), { timeout: 30000 }).toBe(true);
      if (cancel) {
        attempt.sender_complete_before_interrupt = (await snapshot()).some((e: any) => e.kind === 'sender_complete' && e.epoch === epoch);
        attempt.interrupt_epoch = await control('interrupt');
        await expect.poll(() => page.locator('audio').evaluate((audio: HTMLAudioElement) => audio.srcObject === null || audio.muted)).toBe(true);
        interrupted++;
      } else {
        await expect.poll(async () => (await snapshot()).some((e: any) => e.kind === 'sender_complete' && e.epoch === epoch), { timeout: 60000 }).toBe(true);
        const end = (await snapshot()).filter((e: any) => e.kind === 'sender_complete' && e.epoch === epoch);
        expect(end).toHaveLength(1);
        const metrics = end[0].native_source_metrics, count = end[0].valid_audio_samples;
        expect(count).toBeGreaterThan(0);
        expect(end[0].codec_padding_samples).toBe(((-count % 480 + 480) % 480) + 960);
        expect(metrics.source_video_frames + metrics.video_skipped).toBe(Math.ceil(count / 1600));
        expect(metrics.queue_peak_samples).toBeLessThanOrEqual(48000);
        expect(metrics.motion_queue_peak_packets).toBeLessThanOrEqual(50);
        expect(metrics.pose_queue_peak_frames).toBeLessThanOrEqual(32);
        expect(end[0].presentation_ack).toBe(false);
        attempt.sender = end[0]; completed++;
      }
      attempt.status = 'passed'; attempt.node_finished_ms = performance.now() - started;
      attempt.observations = await snapshot();
      rows.push(attempt); await appendFile(progress, JSON.stringify(attempt) + '\n');
      attempt = undefined; previous = binding;
      await page.waitForTimeout(rows.length % 10 === 0 ? 2000 : 200);
    }
    expect(completed).toBeGreaterThan(0); expect(interrupted).toBeGreaterThan(0);
    passed = true;
  } finally {
    if (attempt) await appendFile(progress, JSON.stringify({ ...attempt, status: 'failed' }) + '\n');
    try {
      if (sid && csrf) {
        const closed = await context.request.post(`/v1/sessions/${sid}/close`, { data: {}, headers: { Origin: baseURL!, 'X-CSRF-Token': csrf } });
        closeConfirmed = closed.status() === 200 && (await closed.json()).status === 'CLOSED';
        expect(closeConfirmed).toBe(true);
      }
    } finally {
      await writeFile(resolve(directory, 'summary.json'), JSON.stringify({ passed: passed && closeConfirmed, close_confirmed: closeConfirmed, requested_seconds: seconds,
        measured_node_elapsed_ms: started ? performance.now() - started : null, completed, interrupted, attempts: rows.length + (attempt ? 1 : 0),
        controls, origin: baseURL, browser_version: browser.version(), microphone_input: 'none; typed closed-loop workload',
        clock: 'Node performance; browser observation clocks stored separately', physical_av_qualified: false,
        latency_or_capacity_qualified: false }, null, 2), { flag: 'wx', mode: 0o600 });
    }
  }
});
