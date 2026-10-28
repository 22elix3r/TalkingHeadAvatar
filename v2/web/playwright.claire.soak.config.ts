import { defineConfig } from '@playwright/test';
import reference from './playwright.claire.config';

const seconds = Number(process.env.THA_CLAIRE_SOAK_SECONDS ?? '60');
if (![60, 1800].includes(seconds)) throw new Error('Choose the explicit 60-second smoke or 1800-second cohort');
export default defineConfig(reference, {
  testMatch: '**/claire-soak.spec.ts', timeout: (seconds + 180) * 1000,
  outputDir: '../var/p06-claire/soak-browser-results',
});
