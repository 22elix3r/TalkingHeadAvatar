import { defineConfig } from '@playwright/test';
import reference from './playwright.firefox.config';
export default defineConfig(reference, { testMatch: '**/native-session.spec.ts', grep: /typed message/,
  outputDir: '../var/p06/native-session-firefox-results',
  use: { ...reference.use, baseURL: 'http://127.0.0.1:18720' } });
