import { test as base, expect } from '@playwright/test';
import { mkdir, mkdtemp, rm } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';

// Firefox's automation permission API cannot grant camera/microphone. Seed only
// a newly-created disposable profile with localhost permissions before launch.
// No user's profile is read or modified, and no physical input is opened merely
// to unlock device labels.
export const test = base.extend({ context: async ({ playwright, browserName, baseURL, headless }, use) => {
  if (browserName === 'firefox') {
    await mkdir(resolve('../var/p06'), { recursive: true, mode: 0o700 });
    const directory = await mkdtemp(resolve('../var/p06/virtual-firefox-profile-'));
    try {
      execFileSync(resolve('../.venv/bin/python'), ['-c', `
import sqlite3,sys,time
from pathlib import Path
p=Path(sys.argv[1])/'permissions.sqlite'
with sqlite3.connect(p) as db:
 db.execute('CREATE TABLE moz_perms (id INTEGER PRIMARY KEY, origin TEXT, type TEXT, permission INTEGER, expireType INTEGER, expireTime INTEGER, modificationTime INTEGER)')
 db.execute('CREATE TABLE moz_hosts (id INTEGER PRIMARY KEY, host TEXT, type TEXT, permission INTEGER, expireType INTEGER, expireTime INTEGER, modificationTime INTEGER, isInBrowserElement INTEGER)')
 db.execute('PRAGMA user_version=12')
 for permission in ['microphone','camera']:
  db.execute('INSERT INTO moz_perms (origin,type,permission,expireType,expireTime,modificationTime) VALUES (?,?,1,0,0,?)',(sys.argv[2],permission,int(time.time()*1000)))
p.chmod(0o600)
`, directory, baseURL!], { timeout: 5000 });
      const context = await playwright.firefox.launchPersistentContext(directory, { headless, baseURL,
        viewport: { width: 1280, height: 1000 }, firefoxUserPrefs: {
          'media.peerconnection.ice.loopback': false, 'media.navigator.streams.fake': false,
          'media.navigator.permission.disabled': true, 'media.devices.enumerate.legacy.enabled': true,
        } });
      try { await use(context); } finally { await context.close(); }
    } finally { await rm(directory, { recursive: true, force: true }); }
  } else {
    const browser = await playwright.chromium.launch({ headless, executablePath: '/opt/google/chrome/chrome',
      args: ['--no-sandbox', '--use-fake-ui-for-media-stream'] });
    try {
      const context = await browser.newContext({ baseURL, viewport: { width: 1280, height: 1000 }, permissions: ['camera', 'microphone'] });
      try { await use(context); } finally { await context.close(); }
    } finally { await browser.close(); }
  }
} });
export { expect };
