export function csrfToken() {
  const name = import.meta.env.VITE_CSRF_COOKIE_NAME || 'tha_csrf';
  return document.cookie.split('; ').find(row => row.startsWith(`${name}=`))?.slice(name.length + 1) ?? '';
}
let pending: Promise<{ csrf_token: string; tenant_id: string; profile?: string; default_asset_id?: string }> | undefined;

export function identity() {
  if (!pending) pending = fetch('/v1/dev-login', { method: 'POST', credentials: 'same-origin',
    signal: AbortSignal.timeout(10000),
    headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': csrfToken() }, body: '{}' })
    .then(async response => {
      if (!response.ok) throw new Error('The local library could not connect. Please reload.');
      return response.json();
    }).catch(error => { pending = undefined; throw error; });
  return pending;
}

export async function api<T>(path: string, body?: unknown, options: { method?: string; token?: string; raw?: string; key?: string } = {}): Promise<T> {
  await identity();
  const method = options.method ?? (body === undefined ? 'GET' : 'POST');
  const response = await fetch(path, { method, credentials: 'same-origin',
    signal: AbortSignal.timeout(15000),
    headers: { ...(method === 'GET' ? {} : { 'Content-Type': 'application/json', 'X-CSRF-Token': csrfToken() }),
      ...(options.key ? { 'Idempotency-Key': options.key } : {}), ...(options.token ? { 'X-Upload-Token': options.token } : {}) },
    body: options.raw ?? (body === undefined ? undefined : JSON.stringify(body)) });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const messages: Record<string, string> = { authorization_unavailable: 'Permission for this asset is no longer available.',
      upload_size_or_hash_mismatch: 'The file did not match its declared size or fingerprint. Choose it again.',
      review_revision_changed: 'The asset changed. Reload its review before approving.', invalid_import: 'The file is not a compatible mesh import.',
      research_reference_only: 'This preview uses the vendor research avatar. Imported identities need retargeting first.' };
    throw new Error(messages[data.detail] ?? 'That library action could not finish. Please retry.');
  }
  return response.json();
}
