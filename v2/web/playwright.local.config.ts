import { defineConfig } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import reference from './playwright.config';

const profile = process.env.THA_LOCAL_PROFILE ?? 'p06-check';
if (!/^[a-z][a-z0-9-]{1,23}$/.test(profile)) throw new Error('Invalid local profile name');
const settings = JSON.parse(readFileSync(resolve(`../var/local/${profile}/runtime.json`), 'utf8'));
if (!/^http:\/\/127\.0\.0\.1:\d+$/.test(settings.origin)) throw new Error('Loopback profile required');
export default defineConfig(reference, { testMatch: '**/local-profile.spec.ts',
  outputDir: '../var/p06/local-browser-results', use: { ...reference.use, baseURL: settings.origin } });
