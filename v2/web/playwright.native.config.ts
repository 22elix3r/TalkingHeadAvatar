import { defineConfig } from '@playwright/test';
import reference from './playwright.config';

export default defineConfig(reference, { testMatch: '**/native-media.spec.ts', outputDir: '../var/p06/browser-results' });
