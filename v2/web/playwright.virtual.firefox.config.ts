import { defineConfig } from '@playwright/test';

export default defineConfig({ testDir: './tests', testMatch: '**/virtual-device.spec.ts', workers: 1, retries: 0,
  globalSetup: './tests/ready.ts', timeout: 120000, outputDir: '../var/p06/virtual-firefox-results',
  use: { browserName: 'firefox', headless: true, baseURL: 'http://127.0.0.1:18700', viewport: { width: 1280, height: 1000 },
    launchOptions: { firefoxUserPrefs: {
      'media.peerconnection.ice.loopback': false,
      'media.navigator.streams.fake': false,
      'media.navigator.permission.disabled': true,
      // Test-only enumeration lets us select exact virtual IDs before capture,
      // avoiding any temporary opening of the user's default physical input.
      'media.devices.enumerate.legacy.enabled': true,
    } } } });
