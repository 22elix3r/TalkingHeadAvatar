import { defineConfig } from '@playwright/test';
import inbound from './playwright.inbound.config';
import firefox from './playwright.firefox.config';

export default defineConfig(inbound, {
  grep: /typed interrupt|Firefox synthetic microphone/,
  outputDir: '../var/p06-claire/inbound-firefox-results',
  use: { ...firefox.use, baseURL: inbound.use?.baseURL },
});
