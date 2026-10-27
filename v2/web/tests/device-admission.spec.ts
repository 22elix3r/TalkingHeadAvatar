import { test, expect, type Page } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { readFile, writeFile, unlink } from 'node:fs/promises';
import { resolve } from 'node:path';

async function observe(page: Page) {
  await page.addInitScript(() => { (window as any).__admission = [];
    window.addEventListener('tha-observation', event => (window as any).__admission.push((event as CustomEvent).detail)); });
}
async function answer(page: Page) {
  await page.getByLabel('Your message').fill('What is two plus two?');
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect.poll(() => page.evaluate(() => (window as any).__admission.some((event: any) => event.kind === 'decoded_audio_observed')), { timeout: 60000 }).toBe(true);
}

test('real Qwen lease protects active session from another profile and standalone worker, then permits handoff', async ({ page, browser }) => {
  const second = await browser.newContext({ baseURL: 'http://127.0.0.1:18710' });
  const other = await second.newPage(); await observe(page); await observe(other);
  const root = resolve('..'), probe = resolve('../var/p06/device-competitor.json');
  await unlink(probe).catch(error => { if (error.code !== 'ENOENT') throw error; });
  try {
    await page.goto('/'); await page.getByRole('button', { name: 'Use typed messages instead' }).click();
    await expect(page.getByText('Ready for a message', { exact: true })).toBeVisible({ timeout: 45000 });
    execFileSync(resolve('../.venv/bin/python'), [resolve('../scripts/check_device_admission.py'), '--runtime',
      resolve('../var/native-runtime-serial.json'), '--output', probe], { cwd: root, timeout: 20000,
      env: { ...process.env, HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1' }, stdio: 'pipe' });
    await other.goto('/'); await other.getByRole('button', { name: 'Use typed messages instead' }).click();
    await expect(other.getByRole('alert')).toContainText('Another local profile or voice task is using the GPU', { timeout: 15000 });
    await answer(page); // The rejected competitor must not interrupt the admitted real worker.
    await page.getByRole('button', { name: 'End conversation' }).click();
    await expect(page.getByRole('button', { name: 'Start conversation' })).toBeVisible();
    await other.goto('/'); await other.getByRole('button', { name: 'Use typed messages instead' }).click();
    await expect(other.getByText('Ready for a message', { exact: true })).toBeVisible({ timeout: 45000 });
    await answer(other);
    const evidence = { primary: await page.evaluate(() => (window as any).__admission),
      secondary: await other.evaluate(() => (window as any).__admission),
      competitor: JSON.parse(await readFile(probe, 'utf8')), cross_profile_busy_visible: true,
      actual_qwen_after_rejection: true, actual_qwen_after_handoff: true, multi_session_capacity_qualified: false };
    await writeFile(resolve('../var/p06/device-browser.json'), JSON.stringify(evidence, null, 2), { mode: 0o600 });
    await other.getByRole('button', { name: 'End conversation' }).click();
    await expect(other.getByRole('button', { name: 'Start conversation' })).toBeVisible();
  } finally {
    for (const target of [page, other]) if (!target.isClosed()) {
      const end = target.getByRole('button', { name: 'End conversation' });
      if (await end.isVisible().catch(() => false)) await end.click().catch(() => {});
    }
    await second.close();
  }
});
