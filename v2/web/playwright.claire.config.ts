import { defineConfig } from '@playwright/test';
import reference from './playwright.config';
export default defineConfig(reference, {
  testMatch: '**/claire-session.spec.ts', outputDir: '../var/p06-claire/browser-results',
  projects: [{ name: 'chromium' }],
  use: { ...reference.use, baseURL: 'http://127.0.0.1:18730' },
});
