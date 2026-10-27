import { test, expect } from '@playwright/test';
import { writeFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';

for (const microphone of [false, true]) {
  test(`Claire research conversation: ${microphone ? 'microphone ASR' : 'typed message and interruption'}`, async ({ page, context, baseURL }) => {
    await page.addInitScript(() => {
      (window as any).__claireEvidence = [];
      window.addEventListener('tha-observation', event => (window as any).__claireEvidence.push((event as CustomEvent).detail));
    });
    let sid = '', csrf = '';
    page.on('response', async response => {
      if (response.url().endsWith('/v1/dev-login')) csrf = (await response.json()).csrf_token;
      if (response.url().endsWith('/v1/sessions') && response.status() === 202) sid = (await response.json()).id;
    });
    try {
      await page.goto('/');
      await expect(page.getByText('Claire research preview', { exact: true })).toBeVisible();
      const avatar = page.getByLabel('Vendor research avatar');
      await expect(avatar).toBeVisible();
      expect((await avatar.boundingBox())?.width).toBeGreaterThanOrEqual(200);
      await expect(page.getByLabel('Avatar reference')).toHaveValue('claire-vendor-research');
      await expect(page.getByText('Manage avatar imports', { exact: true })).toHaveCount(0);
      await page.getByRole('button', { name: microphone ? 'Start conversation' : 'Use typed messages instead' }).click();
      await expect(page.getByText(microphone ? 'Listening' : 'Ready for a message', { exact: true })).toBeVisible({ timeout: 45000 });
      if (microphone) await expect(page.locator('.caption.user')).toContainText(/two|2/i, { timeout: 45000 });
      else {
        await page.getByLabel('Your message').fill('What is two plus two?');
        await page.getByRole('button', { name: 'Send message' }).click();
      }
      await expect.poll(() => page.locator('video').evaluate((video: HTMLVideoElement) => video.videoWidth), { timeout: 30000 }).toBe(512);
      const decoded = () => page.evaluate(() => (window as any).__claireEvidence.filter((event: any) => event.kind === 'decoded_audio_observed').length);
      await expect.poll(decoded, { timeout: 60000 }).toBeGreaterThan(0);
      if (!microphone) {
        await page.getByRole('button', { name: 'Interrupt', exact: true }).click();
        await expect.poll(() => page.locator('audio').evaluate(audio => (audio as HTMLAudioElement).srcObject === null || (audio as HTMLAudioElement).muted)).toBe(true);
        await page.getByLabel('Your message').fill('Say hello again.');
        await page.getByRole('button', { name: 'Send message' }).click();
        await expect.poll(() => page.evaluate(() => {
          const events = (window as any).__claireEvidence;
          const bindings = events.filter((event: any) => event.kind === 'playback_binding');
          const latest = bindings.at(-1);
          return bindings.length > 1 && events.some((event: any) => event.kind === 'decoded_audio_observed' && event.epoch === latest.epoch);
        }), { timeout: 60000 }).toBe(true);
      }
      await expect.poll(() => page.evaluate(() => {
        const events = (window as any).__claireEvidence;
        const latest = events.filter((event: any) => event.kind === 'playback_binding').at(-1);
        return latest && events.some((event: any) => event.kind === 'sender_complete' && event.epoch === latest.epoch);
      }), { timeout: 45000 }).toBe(true);
      const observations = await page.evaluate(() => (window as any).__claireEvidence);
      const bindings = observations.filter((event: any) => event.kind === 'playback_binding');
      expect(bindings.length).toBeGreaterThan(0);
      expect(bindings.every((event: any) => event.visual_mode === 'claire_vendor_research' && event.transport === 'livekit_cpp_preencoded')).toBe(true);
      expect(observations.some((event: any) => event.kind === 'video_presented_callback' && event.epoch === bindings.at(-1).epoch && event.visual_mode === 'claire_vendor_research')).toBe(true);
      expect(observations.some((event: any) => event.kind === 'receiver_recovery_required')).toBe(false);
      const complete = observations.filter((event: any) => event.kind === 'sender_complete' && event.epoch === bindings.at(-1).epoch);
      expect(complete).toHaveLength(1);
      const sender = complete[0];
      expect(sender.valid_audio_samples).toBeGreaterThan(0);
      expect(sender.codec_padding_samples).toBe((-sender.valid_audio_samples % 480 + 480) % 480 + 960);
      expect(sender.native_source_metrics.source_video_frames + sender.native_source_metrics.video_skipped).toBe(Math.ceil(sender.valid_audio_samples / 1600));
      expect(sender.native_source_metrics.preroll_video_frames).toBeGreaterThanOrEqual(3);
      expect(sender.native_source_metrics.pose_queue_peak_frames).toBeLessThanOrEqual(32);
      expect(sender.native_source_metrics.queue_peak_samples).toBeLessThanOrEqual(48000);
      expect(sender.presentation_ack).toBe(false);
      if (!microphone) {
        expect(bindings[0].audio_sid).not.toBe(bindings.at(-1).audio_sid);
        expect(bindings[0].video_sid).not.toBe(bindings.at(-1).video_sid);
      }
      const cohort = process.env.THA_CLAIRE_COHORT ?? test.info().project.name;
      if (!/^[a-z0-9-]{1,64}$/.test(cohort)) throw new Error('Bounded private cohort name required.');
      const directory = resolve('../var/p06-claire/browser', cohort);
      await mkdir(directory, { recursive: true, mode: 0o700 });
      const name = microphone ? 'microphone' : 'typed';
      await writeFile(resolve(directory, `${name}.json`), JSON.stringify({ origin: baseURL, observations, browser_version: context.browser()?.version(),
        real_providers: true, source_avatar: 'vendor_research_sample', voice_kind: 'synthetic_fixture',
        microphone_input: microphone ? 'synthetic file through browser capture' : 'none', physical_av_qualified: false }, null, 2), { mode: 0o600 });
      await page.screenshot({ path: resolve(directory, `${name}.png`) });
      await page.getByRole('button', { name: 'End conversation' }).click();
      await expect(page.getByRole('button', { name: 'Start conversation' })).toBeVisible();
    } finally {
      if (sid && csrf) await context.request.post(`/v1/sessions/${sid}/close`, { data: {}, headers: { Origin: baseURL!, 'X-CSRF-Token': csrf } });
    }
  });
}
