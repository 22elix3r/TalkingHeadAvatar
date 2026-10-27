import { defineConfig } from '@playwright/test';
import reference from './playwright.config';

export default defineConfig(reference, {
  testMatch: '**/lifecycle.spec.ts', outputDir: '../var/p05/browser-results',
});
