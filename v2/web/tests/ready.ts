import type { FullConfig } from '@playwright/test';

export default async function ready(config: FullConfig) {
  const baseURL = config.projects[0].use.baseURL;
  if (!baseURL || !/^http:\/\/127\.0\.0\.1:\d+$/.test(baseURL)) throw new Error('A loopback test origin is required.');
  const deadline = Date.now() + 60000;
  while (Date.now() < deadline) {
    try {
      const result = await fetch(`${baseURL}/health/ready`, { signal: AbortSignal.timeout(1000) });
      if (result.ok && (await result.json()).ready) return;
    } catch { /* A cold process has not opened its listener yet. */ }
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  throw new Error('The real P04 services did not become ready; no mocked provider substituted.');
}
