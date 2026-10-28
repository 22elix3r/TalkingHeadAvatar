import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

export function claireOrigin(): string {
  const profile = process.env.THA_CLAIRE_PROFILE;
  if (!profile) return 'http://127.0.0.1:18730';
  if (!/^[a-z][a-z0-9-]{1,23}$/.test(profile)) throw new Error('Invalid Claire local profile');
  const settings = JSON.parse(readFileSync(resolve(`../var/local/${profile}/runtime.json`), 'utf8'));
  if (settings.media_backend !== 'native_claire_research' || !settings.native_media_runtime ||
      !/^http:\/\/127\.0\.0\.1:\d+$/.test(settings.origin)) throw new Error('Pinned loopback Claire profile required');
  return settings.origin;
}
