import { defineConfig } from '@playwright/test';
import { resolve } from 'node:path';

export default defineConfig({
  testDir: './tests', testMatch: '**/browser*.spec.ts', workers: 1, retries: 0,
  globalSetup: './tests/ready.ts',
  timeout: 120000, outputDir: '../var/p04/browser-results',
  reporter: [['list']],
  use: { baseURL: 'http://127.0.0.1:18700', headless: true,
    viewport: { width: 1280, height: 1000 },
    launchOptions: { executablePath: '/opt/google/chrome/chrome', args: [
      '--no-sandbox', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
      `--use-file-for-fake-audio-capture=${resolve('../var/p04/microphone.wav')}`,
    ] },
  },
});
