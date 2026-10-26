import { test, expect } from '@playwright/test';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

async function fixtureBundle() {
  const root = resolve('../src/talkinghead_v2/fixtures/p03-avatar');
  return Buffer.from(JSON.stringify({ schema_version: 1,
    manifest: await readFile(resolve(root, 'avatar.json'), 'utf8'), mesh: await readFile(resolve(root, 'mesh.json'), 'utf8') }));
}

test('real import, review, select, Qwen conversation and active revocation', async ({ page, context }) => {
  const bundle = await fixtureBundle();
  let assetId = '', csrf = '', sid = '';
  let grant: { url: string; token: string } | undefined;
  await page.addInitScript(() => {
    const target = window as unknown as { __observations: Record<string, unknown>[] };
    target.__observations = [];
    window.addEventListener('tha-observation', e => target.__observations.push((e as CustomEvent).detail));
  });
  page.on('response', async response => {
    if (response.url().endsWith('/v1/dev-login')) csrf = (await response.json()).csrf_token;
    if (response.url().endsWith('/v1/enrollments') && response.status() === 202) assetId = (await response.json()).asset_id;
    if (response.url().endsWith('/v1/sessions') && response.status() === 202) sid = (await response.json()).id;
    if (response.url().endsWith('/join-grant') && response.ok()) grant = await response.json();
  });
  try {
    await page.goto('/');
    await page.getByText('Manage avatar imports', { exact: true }).click();
    await page.getByLabel('Asset name', { exact: true }).fill('P05 timing fixture');
    await page.getByLabel('Authorization record reference').fill('first-party-mathematical-fixture-v1');
    await page.getByLabel('Canonical mesh JSON bundle (up to 4 MiB)').setInputFiles({ name: 'timing-bundle.json', mimeType: 'application/json', buffer: bundle });
    await page.getByLabel('I own or have documented permission').check();
    await page.getByRole('button', { name: 'Import for review', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Review P05 timing fixture', exact: true })).toBeVisible({ timeout: 20000 });
    await page.reload(); // The tenant/library survives a fresh module and CSRF setup.
    await page.getByText('Manage avatar imports', { exact: true }).click();
    await page.getByRole('button', { name: 'Review P05 timing fixture', exact: true }).click();
    await expect(page.getByAltText('Neutral static preview of P05 timing fixture')).toBeVisible();
    await page.getByLabel('Approve this exact revision for static reference playback only.').check();
    await page.getByRole('button', { name: 'Approve reference', exact: true }).click();
    await expect(page.getByLabel('Avatar reference', { exact: true })).toContainText('P05 timing fixture');
    await page.getByLabel('Avatar reference', { exact: true }).selectOption(assetId);
    await page.getByRole('button', { name: 'Use typed messages instead' }).click();
    await expect(page.getByText('Ready for a message', { exact: true })).toBeVisible({ timeout: 45000 });
    await page.getByLabel('Your message').fill('Count from one to twenty.');
    await page.getByRole('button', { name: 'Send message' }).click();
    await expect.poll(() => page.evaluate(() => (window as unknown as { __observations: { kind: string }[] })
      .__observations.some(e => e.kind === 'decoded_audio_observed')), { timeout: 60000 }).toBe(true);
    await page.screenshot({ path: resolve('../var/p05/import-conversation.png'), fullPage: true });
    await page.getByRole('button', { name: 'Revoke P05 timing fixture', exact: true }).click();
    await expect(page.getByRole('alert').filter({ hasText: 'Avatar permission was revoked' })).toBeVisible({ timeout: 15000 });
    await expect.poll(() => page.locator('audio').evaluate(e => (e as HTMLAudioElement).srcObject === null)).toBe(true);
    await expect.poll(async () => {
      const r = await context.request.get(`/v1/assets/${assetId}/deletion`); return (await r.json()).status;
    }, { timeout: 15000 }).toBe('complete');
    const denied = await context.request.post(`/v1/sessions/${sid}/join-grant`, { data: {}, headers: {
      Origin: 'http://127.0.0.1:18700', 'X-CSRF-Token': csrf,
    } });
    expect([403, 409]).toContain(denied.status());
    // Self-hosted token TTL is not active token revocation: a still-valid grant
    // may rejoin an empty room. Prove that it cannot recover the stopped source.
    await page.addScriptTag({ path: resolve('node_modules/livekit-client/dist/livekit-client.umd.js') });
    const rejoin = await page.evaluate(async credentials => {
      const target = window as unknown as { LivekitClient: { Room: new () => {
        connect: (url: string, token: string) => Promise<void>; disconnect: () => Promise<void>;
        remoteParticipants: Map<string, { trackPublications: Map<string, unknown> }> } } };
      const room = new target.LivekitClient.Room();
      try {
        await room.connect(credentials!.url, credentials!.token);
        await new Promise(resolve => setTimeout(resolve, 1500));
        return { connected: true, remoteTracks: [...room.remoteParticipants.values()].reduce((n, p) => n + p.trackPublications.size, 0) };
      } catch { return { connected: false, remoteTracks: 0 }; }
      finally { await room.disconnect(); }
    }, grant);
    expect(rejoin.remoteTracks).toBe(0);
    await writeFile(resolve('../var/p05/browser-lifecycle-evidence.json'), JSON.stringify({
      fixture: 'first-party procedural mesh; existing synthetic Qwen voice', import_review_select: true,
      real_qwen_received: true, revocation_gated_receiver: true, deletion_receipt: 'complete',
      old_join_grant_denied: true, still_valid_sfu_token: rejoin,
      instantaneous_token_invalidation_claimed: false, human_quality_qualified: false,
      observations: await page.evaluate(() => (window as unknown as { __observations: unknown[] }).__observations),
    }, null, 2));
  } finally {
    if (sid && csrf) await context.request.post(`/v1/sessions/${sid}/close`, { data: {}, headers: { Origin: 'http://127.0.0.1:18700', 'X-CSRF-Token': csrf } });
    if (assetId && csrf) await context.request.post(`/v1/assets/${assetId}/revoke`, { data: {}, headers: { Origin: 'http://127.0.0.1:18700', 'X-CSRF-Token': csrf } });
  }
});

test('lost enrollment and approval responses retry without duplicate resources', async ({ page, context }) => {
  let csrf = '', assetId = '';
  page.on('response', async response => {
    if (response.url().endsWith('/v1/dev-login')) csrf = (await response.json()).csrf_token;
    if (response.url().endsWith('/v1/enrollments') && response.status() === 202) assetId = (await response.json()).asset_id;
  });
  await page.route('**/v1/enrollments', async route => { await route.fetch(); await route.abort('failed'); }, { times: 1 });
  await page.route('**/approve', async route => { await route.fetch(); await route.abort('failed'); }, { times: 1 });
  try {
    await page.goto('/'); await page.getByText('Manage avatar imports', { exact: true }).click();
    await page.getByLabel('Asset name', { exact: true }).fill('Retry fixture');
    await page.getByLabel('Authorization record reference').fill('first-party-mathematical-fixture-v1');
    await page.getByLabel('Canonical mesh JSON bundle (up to 4 MiB)').setInputFiles({ name: 'timing-bundle.json', mimeType: 'application/json', buffer: await fixtureBundle() });
    await page.getByLabel('I own or have documented permission').check();
    await page.getByRole('button', { name: 'Import for review', exact: true }).click();
    await expect(page.getByRole('alert')).toBeVisible();
    await page.getByRole('button', { name: 'Import for review', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Review Retry fixture', exact: true })).toBeVisible({ timeout: 20000 });
    const jobs = await context.request.get('/v1/jobs'); expect((await jobs.json()).items).toHaveLength(1);
    await page.getByRole('button', { name: 'Review Retry fixture', exact: true }).click();
    await page.getByLabel('Approve this exact revision for static reference playback only.').check();
    await page.getByRole('button', { name: 'Approve reference', exact: true }).click();
    await expect(page.getByRole('alert')).toBeVisible();
    await page.getByRole('button', { name: 'Approve reference', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Approve reference', exact: true })).toHaveCount(0);
    const assets = await context.request.get('/v1/assets');
    expect((await assets.json()).items.filter((row: { name: string }) => row.name === 'Retry fixture')).toHaveLength(1);
  } finally {
    if (assetId && csrf) await context.request.post(`/v1/assets/${assetId}/revoke`, { data: {}, headers: { Origin: 'http://127.0.0.1:18700', 'X-CSRF-Token': csrf } });
  }
});
