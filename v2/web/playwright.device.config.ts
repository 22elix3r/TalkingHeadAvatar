import { defineConfig } from '@playwright/test';
import reference from './playwright.config';
export default defineConfig(reference, { testMatch: '**/device-admission.spec.ts', outputDir: '../var/p06/device-browser-results' });
