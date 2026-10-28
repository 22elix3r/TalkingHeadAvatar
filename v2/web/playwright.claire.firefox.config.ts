import { defineConfig } from '@playwright/test';
import reference from './playwright.firefox.config';
import { claireOrigin } from './playwright.claire-profile';
export default defineConfig(reference, {
  testMatch: '**/claire-session.spec.ts', grep: /typed message/, outputDir: '../var/p06-claire/firefox-results',
  projects: [{ name: 'firefox' }],
  use: { ...reference.use, baseURL: claireOrigin() },
});
