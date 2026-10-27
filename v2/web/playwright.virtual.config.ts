import { defineConfig } from '@playwright/test';
import reference from './playwright.config';

export default defineConfig(reference, { testMatch: '**/virtual-device.spec.ts', outputDir: '../var/p06/virtual-results',
  use: { ...reference.use, permissions: ['camera', 'microphone'], launchOptions: {
    executablePath: '/opt/google/chrome/chrome', args: ['--no-sandbox', '--use-fake-ui-for-media-stream'],
  } } });
