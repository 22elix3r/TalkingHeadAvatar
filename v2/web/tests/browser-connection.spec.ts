import { test, expect } from '@playwright/test';

test('failed media connection releases admission and can be retried', async ({ page, context }) => {
  let sid = '', csrf = '';
  page.on('response', async response => {
    if (response.url().endsWith('/v1/dev-login')) csrf = (await response.json()).csrf_token;
    if (response.url().endsWith('/v1/sessions') && response.status() === 202) sid = (await response.json()).id;
  });
  // Explicit transport fault, retaining real preparation and the durable session.
  await page.route('**/join-grant', async route => {
    const response = await route.fetch();
    const grant = await response.json();
    await route.fulfill({ response, json: { ...grant, url: 'ws://127.0.0.1:1' } });
  }, { times: 1 });
  try {
    await page.goto('http://localhost:18700/');
    await expect(page).toHaveURL('http://127.0.0.1:18700/');
    await page.getByRole('button', { name: 'Use typed messages instead' }).click();
    await expect(page.getByRole('alert')).toContainText('The audio connection could not be established', { timeout: 45000 });
    await expect(page.getByRole('button', { name: 'Start conversation' })).toBeEnabled({ timeout: 10000 });
    expect(await page.locator('audio').evaluate(e => (e as HTMLAudioElement).srcObject)).toBeNull();
    const failed = await context.request.get(`/v1/sessions/${sid}`);
    expect((await failed.json()).status).toBe('CLOSED');
    await page.getByRole('button', { name: 'Use typed messages instead' }).click();
    await expect(page.getByText('Ready for a message', { exact: true })).toBeVisible({ timeout: 45000 });
    await page.getByRole('button', { name: 'End conversation' }).click();
  } finally {
    if (sid && csrf) await context.request.post(`/v1/sessions/${sid}/close`, { data: {}, headers: {
      Origin: 'http://127.0.0.1:18700', 'X-CSRF-Token': csrf,
    } });
  }
});
