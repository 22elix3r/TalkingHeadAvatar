import { defineConfig } from '@playwright/test';
import reference from './playwright.firefox.config';
export default defineConfig(reference, {
  testMatch: '**/claire-session.spec.ts', grep: /typed message/, outputDir: '../var/p06-claire/firefox-results',
  projects: [{ name: 'firefox' }],
  use: { ...reference.use, baseURL: 'http://127.0.0.1:18730' },
});
