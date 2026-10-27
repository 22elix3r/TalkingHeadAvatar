import { defineConfig } from '@playwright/test';

// Disposable Playwright Firefox profile. Keep the normal loopback prohibition:
// enabling it would conceal this bug. check_p04_firefox.py tests installed Firefox.
export default defineConfig({
  testDir: './tests', testMatch: '**/firefox.spec.ts', workers: 1, retries: 0,
  globalSetup: './tests/ready.ts', timeout: 120000,
  outputDir: '../var/p04/firefox-results',
  use: {
    browserName: 'firefox', headless: true,
    baseURL: 'http://127.0.0.1:18700', viewport: { width: 1280, height: 1000 },
    launchOptions: { firefoxUserPrefs: {
      'media.peerconnection.ice.loopback': false,
      // Only the isolated test profile gets a synthetic microphone and permission.
      'media.navigator.streams.fake': true,
      'media.navigator.permission.disabled': true,
    } },
  },
});
