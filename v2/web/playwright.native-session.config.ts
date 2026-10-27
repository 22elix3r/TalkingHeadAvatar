import { defineConfig } from '@playwright/test';
import reference from './playwright.config';
export default defineConfig(reference, { testMatch: '**/native-session.spec.ts',
  outputDir: '../var/p06/native-session-browser-results',
  use: { ...reference.use, baseURL: 'http://127.0.0.1:18720' } });
