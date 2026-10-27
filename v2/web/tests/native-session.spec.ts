import { test, expect } from '@playwright/test';
import { writeFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';

for (const microphone of [false, true]) {
  test(`native CUDA/NVENC session: ${microphone ? 'microphone ASR' : 'typed message and interruption'} to real Qwen playback`, async ({ page, context, baseURL }) => {
    await page.addInitScript(() => {
      (window as any).__nativeSessionEvidence = [];
      window.addEventListener('tha-observation', event => (window as any).__nativeSessionEvidence.push((event as CustomEvent).detail));
    });
    let sid = '', csrf = '';
    page.on('response', async response => {
      if (response.url().endsWith('/v1/dev-login')) csrf = (await response.json()).csrf_token;
      if (response.url().endsWith('/v1/sessions') && response.status() === 202) sid = (await response.json()).id;
    });
    try {
      await page.goto('/');
      await page.getByRole('button', { name: microphone ? 'Start conversation' : 'Use typed messages instead' }).click();
      await expect(page.getByText(microphone ? 'Listening' : 'Ready for a message', { exact: true })).toBeVisible({ timeout: 45000 });
      if (microphone) await expect(page.locator('.caption.user')).toContainText(/two|2/i, { timeout: 45000 });
      else {
        await page.getByLabel('Your message').fill('What is two plus two?');
        await page.getByRole('button', { name: 'Send message' }).click();
      }
      await expect.poll(() => page.locator('video').evaluate((element: HTMLVideoElement) => element.videoWidth), { timeout: 30000 }).toBe(512);
      const decoded = () => page.evaluate(() => (window as any).__nativeSessionEvidence.filter((event: any) => event.kind === 'decoded_audio_observed').length);
      await expect.poll(decoded, { timeout: 60000 }).toBeGreaterThan(0);
      if (!microphone) {
        await page.getByRole('button', { name: 'Interrupt', exact: true }).click();
        await expect.poll(() => page.locator('audio').evaluate(e => (e as HTMLAudioElement).srcObject === null || (e as HTMLAudioElement).muted)).toBe(true);
        await page.getByLabel('Your message').fill('Say hello again.');
        await page.getByRole('button', { name: 'Send message' }).click();
        await expect.poll(decoded, { timeout: 60000 }).toBeGreaterThan(1);
      }
      await expect.poll(() => page.evaluate(() => {
        const events = (window as any).__nativeSessionEvidence;
        const latest = events.filter((event: any) => event.kind === 'playback_binding').at(-1);
        return latest && events.some((event: any) => event.kind === 'sender_complete' && event.epoch === latest.epoch);
      }), { timeout: 45000 }).toBe(true);
      const observations = await page.evaluate(() => (window as any).__nativeSessionEvidence);
      const bindings = observations.filter((event: any) => event.kind === 'playback_binding');
      expect(bindings.length).toBeGreaterThan(0);
      expect(bindings.every((binding: any) => binding.transport === 'livekit_cpp_preencoded')).toBe(true);
      expect(observations.some((event: any) => event.kind === 'video_presented_callback')).toBe(true);
      const completed = observations.filter((event: any) => event.kind === 'sender_complete' && event.epoch === bindings.at(-1).epoch);
      expect(completed.length).toBe(1);
      expect(completed[0].valid_audio_samples).toBeGreaterThan(0);
      expect(completed[0].codec_padding_samples).toBeGreaterThanOrEqual(960);
      expect(completed[0].codec_padding_samples).toBeLessThan(1440);
      expect(completed[0].native_source_metrics.queue_peak_samples).toBeLessThanOrEqual(48000);
      expect(completed[0].presentation_ack).toBe(false);
      expect(observations.some((event: any) => event.kind === 'receiver_recovery_required')).toBe(false);
      if (!microphone) {
        expect(bindings[0].audio_sid).not.toBe(bindings.at(-1).audio_sid);
        expect(bindings[0].video_sid).not.toBe(bindings.at(-1).video_sid);
      }
      const directory = resolve('../var/p06/native-session-browser'); await mkdir(directory, { recursive: true, mode: 0o700 });
      const name = microphone ? 'microphone' : 'typed';
      await writeFile(resolve(directory, `${name}.json`), JSON.stringify({ origin: baseURL, observations,
        real_providers: true, microphone_input: microphone ? 'synthetic file through browser capture' : 'none', physical_av_qualified: false }, null, 2));
      await page.screenshot({ path: resolve(directory, `${name}.png`) });
      await page.getByRole('button', { name: 'End conversation' }).click();
      await expect(page.getByRole('button', { name: 'Start conversation' })).toBeVisible();
    } finally {
      if (sid && csrf) await context.request.post(`/v1/sessions/${sid}/close`, { data: {}, headers: { Origin: baseURL!, 'X-CSRF-Token': csrf } });
    }
  });
}
