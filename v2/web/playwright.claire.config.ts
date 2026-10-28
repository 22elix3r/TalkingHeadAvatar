import { defineConfig } from '@playwright/test';
import reference from './playwright.config';
import { claireOrigin } from './playwright.claire-profile';
export default defineConfig(reference, {
  testMatch: '**/claire-session.spec.ts', outputDir: '../var/p06-claire/browser-results',
  projects: [{ name: 'chromium' }],
  use: { ...reference.use, baseURL: claireOrigin() },
});
