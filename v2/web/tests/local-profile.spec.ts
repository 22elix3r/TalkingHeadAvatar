import { test, expect } from '@playwright/test';
import { writeFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';

for (const microphone of [false, true]) {
  test(`fresh offline-provisioned profile: ${microphone ? 'microphone ASR' : 'typed message and interruption'} to real Qwen playback`, async ({ page, context, baseURL }) => {
    await page.addInitScript(() => {
      (window as any).__localEvidence = [];
      window.addEventListener('tha-observation', event => (window as any).__localEvidence.push((event as CustomEvent).detail));
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
      const decoded = () => page.evaluate(() => (window as any).__localEvidence.filter((event: any) => event.kind === 'decoded_audio_observed').length);
      await expect.poll(decoded, { timeout: 60000 }).toBeGreaterThan(0);
      if (!microphone) {
        await page.getByRole('button', { name: 'Interrupt', exact: true }).click();
        await expect.poll(() => page.locator('audio').evaluate(e => (e as HTMLAudioElement).srcObject === null || (e as HTMLAudioElement).muted)).toBe(true);
        await page.getByLabel('Your message').fill('Say hello again.');
        await page.getByRole('button', { name: 'Send message' }).click();
        await expect.poll(decoded, { timeout: 60000 }).toBeGreaterThan(1);
      }
      await expect.poll(() => page.evaluate(() => (window as any).__localEvidence.some((event: any) => event.kind === 'sender_complete')), { timeout: 45000 }).toBe(true);
      const observations = await page.evaluate(() => (window as any).__localEvidence);
      const bindings = observations.filter((event: any) => event.kind === 'playback_binding');
      if (!microphone) expect(bindings[0].audio_sid).not.toBe(bindings.at(-1).audio_sid);
      const directory = resolve('../var/p06/local-browser'); await mkdir(directory, { recursive: true, mode: 0o700 });
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
