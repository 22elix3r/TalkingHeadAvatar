import { test, expect } from '@playwright/test';
import { resolve } from 'node:path';

test.use({ launchOptions: { executablePath: '/opt/google/chrome/chrome', args: ['--no-sandbox', '--use-fake-device-for-media-stream'] } });

test('native microphone denial keeps typed recovery available', async ({ page, context }) => {
  const protocol = await context.newCDPSession(page);
  await protocol.send('Browser.setPermission', { permission: { name: 'microphone' }, setting: 'denied', origin: 'http://127.0.0.1:18700' });
  await page.goto('/'); await page.getByRole('button', { name: 'Start conversation' }).click();
  await expect(page.getByRole('alert')).toContainText(/permission was denied/i);
  await expect(page.getByRole('button', { name: 'Use typed messages instead' })).toBeEnabled();
  await page.screenshot({ path: resolve('../var/p04/browser-permission.png'), fullPage: true });
});
