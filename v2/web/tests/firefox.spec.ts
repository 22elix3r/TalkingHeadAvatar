import { test, expect } from '@playwright/test';

test('Firefox connects and receives real Qwen audio with loopback ICE disabled', async ({ page, context, browser }) => {
  let sid = '', csrf = '';
  page.on('response', async response => {
    if (response.url().endsWith('/v1/dev-login')) csrf = (await response.json()).csrf_token;
    if (response.url().endsWith('/v1/sessions') && response.status() === 202) sid = (await response.json()).id;
  });
  await page.addInitScript(() => {
    const target = window as unknown as { __observations: Record<string, unknown>[]; __peers: RTCPeerConnection[] };
    target.__observations = [];
    target.__peers = [];
    const Original = window.RTCPeerConnection;
    window.RTCPeerConnection = class extends Original {
      constructor(configuration?: RTCConfiguration) { super(configuration); target.__peers.push(this); }
    };
    window.addEventListener('tha-observation', e => target.__observations.push((e as CustomEvent).detail));
  });
  try {
    await test.info().attach('browser-version', { body: browser.version(), contentType: 'text/plain' });
    await page.goto('/');
    await page.getByRole('button', { name: 'Use typed messages instead' }).click();
    await expect(page.getByText('Ready for a message', { exact: true })).toBeVisible({ timeout: 45000 });
    await page.getByLabel('Your message').fill('What is two plus two?');
    await page.getByRole('button', { name: 'Send message' }).click();
    await expect.poll(() => page.evaluate(() => (window as unknown as { __observations: { kind: string }[] })
      .__observations.some(e => e.kind === 'decoded_audio_observed')), { timeout: 60000 }).toBe(true);
    await page.getByRole('button', { name: 'Interrupt', exact: true }).click();
    await expect.poll(() => page.locator('audio').evaluate(e => (e as HTMLAudioElement).muted || (e as HTMLAudioElement).srcObject === null)).toBe(true);
    await page.getByRole('button', { name: 'End conversation' }).click();
    await expect(page.getByRole('button', { name: 'Start conversation' })).toBeEnabled();
  } finally {
    await test.info().attach('browser-state', { body: JSON.stringify(await page.evaluate(async () => {
      const target = window as unknown as { __observations: unknown[]; __peers: RTCPeerConnection[] };
      return { observations: target.__observations, alert: document.querySelector('[role=alert]')?.textContent,
        peers: await Promise.all((target.__peers ?? []).map(async peer => ({ state: peer.connectionState,
          stats: await Promise.race([peer.getStats().then(stats => [...stats.values()].filter(row => ['candidate-pair', 'local-candidate', 'remote-candidate'].includes(row.type))),
            new Promise(resolve => setTimeout(() => resolve('stats_timeout'), 500))]),
        }))) };
    })), contentType: 'application/json' });
    if (sid && csrf) await context.request.post(`/v1/sessions/${sid}/close`, { timeout: 10000, data: {}, headers: {
      Origin: 'http://127.0.0.1:18700', 'X-CSRF-Token': csrf,
    } });
  }
});
