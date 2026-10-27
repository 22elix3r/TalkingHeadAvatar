import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from './api';

type Asset = { id: string; name: string; status: string; revision?: string; consent_id?: string; builtin?: boolean;
  evaluation?: { vertices: number; faces: number; structural_compatibility: boolean } };
type Job = { id: string; asset_id: string; status: string; stage: string; attempt: number; error_code?: string };

export function AssetLibrary({ selected, onSelect, disabled }: { selected: string; onSelect: (id: string) => void; disabled: boolean }) {
  const [assets, setAssets] = useState<Asset[]>([]), [job, setJob] = useState<Job>();
  const [review, setReview] = useState<Asset>(), [accepted, setAccepted] = useState(false);
  const [name, setName] = useState(''), [reference, setReference] = useState('');
  const [kind, setKind] = useState('synthetic_fixture'), [attested, setAttested] = useState(false);
  const [file, setFile] = useState<File>(), [busy, setBusy] = useState(false), [error, setError] = useState('');
  const [cursor, setCursor] = useState<string | null>(null);
  const importKeys = useRef<{ signature: string; consent: string; upload: string; enrollment: string } | undefined>(undefined);
  const approvalKeys = useRef(new Map<string, string>());
  const research = import.meta.env.VITE_REFERENCE_PROFILE === 'claire_vendor_research' || assets.some(row => row.id === 'claire-vendor-research');
  const selectedReference = research && selected === 'synthetic-audio-reference' ? 'claire-vendor-research' : selected;
  const refresh = useCallback(async () => { const data = await api<{ items: Asset[]; next_cursor: string | null }>('/v1/assets'); setAssets(data.items); setCursor(data.next_cursor); }, []);
  useEffect(() => { let mounted = true;
    void Promise.all([api<{ items: Asset[]; next_cursor: string | null }>('/v1/assets'), api<{ items: Job[] }>('/v1/jobs')]).then(([data, jobs]) => {
      if (mounted) { setAssets(data.items); setCursor(data.next_cursor); setJob(jobs.items.find(row => ['queued', 'running', 'review'].includes(row.status))); }
    })
      .catch(() => { if (mounted) setError('The avatar library is unavailable.'); });
    return () => { mounted = false; };
  }, []);
  useEffect(() => {
    if (!job || !['queued', 'running'].includes(job.status)) return;
    let mounted = true;
    const timer = setInterval(() => { void api<Job>(`/v1/jobs/${job.id}`).then(row => {
      if (!mounted) return; setJob(row);
      if (!['queued', 'running'].includes(row.status)) void refresh().catch(() => { if (mounted) setError('Could not refresh the library.'); });
    }).catch(() => { if (mounted) setError('Import status could not be refreshed.'); }); }, 750);
    return () => { mounted = false; clearInterval(timer); };
  }, [job?.id, job?.status, refresh]);

  async function act(operation: () => Promise<void>) {
    setBusy(true); setError('');
    try { await operation(); } catch (e) { setError(e instanceof Error ? e.message : 'The library action failed.'); }
    finally { setBusy(false); }
  }
  async function submit() {
    if (!file || !attested || !name.trim() || !reference.trim()) return;
    await act(async () => {
      if (file.size > 4 * 1024 * 1024) throw new Error('Choose a JSON bundle smaller than 4 MiB.');
      const raw = await file.text();
      const bytes = new TextEncoder().encode(raw);
      const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(v => v.toString(16).padStart(2, '0')).join('');
      const signature = JSON.stringify({ name: name.trim(), reference: reference.trim(), kind, digest });
      if (importKeys.current?.signature !== signature) importKeys.current = { signature,
        consent: crypto.randomUUID(), upload: crypto.randomUUID(), enrollment: crypto.randomUUID() };
      const keys = importKeys.current;
      const consent = await api<{ id: string }>('/v1/consents', { kind, reference: reference.trim(),
        allowed_uses: [kind === 'synthetic_fixture' ? 'timing_fixture_only' : 'research_avatar'], attested: true }, { key: keys.consent });
      const upload = await api<{ id: string; status: string; write_token: string; content_url: string }>('/v1/uploads', {
        consent_id: consent.id, size: bytes.length, sha256: digest, purpose: 'avatar_import', content_type: 'application/json',
      }, { key: keys.upload });
      if (upload.status !== 'consumed') {
        await api(upload.content_url, undefined, { method: 'PUT', token: upload.write_token, raw });
        await api(`/v1/uploads/${upload.id}/complete`, {});
      }
      setJob(await api<Job>('/v1/enrollments', { upload_id: upload.id, name: name.trim() }, { key: keys.enrollment }));
      await refresh();
    });
  }
  async function approve() {
    if (!review || !accepted) return;
    await act(async () => {
      const operation = `${review.id}:${review.revision}`;
      if (!approvalKeys.current.has(operation)) approvalKeys.current.set(operation, crypto.randomUUID());
      await api(`/v1/assets/${review.id}/approve`, { revision: review.revision, accept_static_reference_limitations: true }, { key: approvalKeys.current.get(operation) });
      setJob(previous => previous?.asset_id === review.id ? { ...previous, status: 'complete', stage: 'published' } : previous);
      setReview(undefined); setAccepted(false); await refresh();
    });
  }
  return <section className="asset-library" aria-label="Avatar library">
    <label htmlFor="avatar">Avatar reference</label>
    <select id="avatar" value={selectedReference} disabled={disabled || busy} onChange={e => onSelect(e.target.value)}>
      {assets.filter(row => row.status === 'approved').map(row => <option key={row.id} value={row.id}>{row.name}</option>)}
      {!assets.length && <option value="synthetic-audio-reference">Synthetic demo voice</option>}
    </select>
    {research ? <p>Vendor research sample with a synthetic voice. Imported identities require retargeting.</p> : <details><summary>Manage avatar imports</summary>
      <p>Import an authorized canonical mesh for static reference playback with the demo voice. Facial animation and human likeness quality remain unqualified.</p>
      <form onSubmit={e => { e.preventDefault(); void submit(); }}>
        <label htmlFor="asset-name">Asset name</label><input id="asset-name" value={name} maxLength={80} required onChange={e => setName(e.target.value)} />
        <label htmlFor="asset-kind">Source authorization</label><select id="asset-kind" value={kind} onChange={e => { setKind(e.target.value); setAttested(false); }}>
          <option value="synthetic_fixture">Synthetic geometry fixture</option><option value="human_consented">My consented likeness</option><option value="research_authorized">Research-authorized asset</option>
        </select>
        <label htmlFor="consent-reference">Authorization record reference</label><input id="consent-reference" value={reference} maxLength={300} required onChange={e => setReference(e.target.value)} />
        <label htmlFor="mesh-bundle">Canonical mesh JSON bundle (up to 4 MiB)</label><input id="mesh-bundle" type="file" accept="application/json,.json" required onChange={e => { setFile(e.target.files?.[0]); importKeys.current = undefined; }} />
        <label className="check"><input type="checkbox" checked={attested} onChange={e => setAttested(e.target.checked)} />I own or have documented permission for this research use, including this likeness if present.</label>
        <button type="submit" disabled={busy || !attested || !file}>Import for review</button>
      </form>
      {job && <div role="status"><p>Import: {job.stage} · {job.status}{job.error_code ? ' · The file could not be validated.' : ''}</p>
        {['queued', 'running', 'review'].includes(job.status) && <button disabled={busy} onClick={() => void act(async () => { setJob(await api(`/v1/jobs/${job.id}/cancel`, {})); await refresh(); })}>Cancel import</button>}
      </div>}
      <ul>{assets.filter(row => !row.builtin && row.id !== 'synthetic-audio-reference').map(row => <li key={row.id}>
        <span>{row.name} · {row.status}</span>
        {row.status === 'review' && <button disabled={busy} onClick={() => void act(async () => { setReview(await api(`/v1/assets/${row.id}`)); setAccepted(false); })}>Review {row.name}</button>}
        {row.status !== 'revoked' && <button disabled={busy} onClick={() => void act(async () => {
          await api(`/v1/assets/${row.id}/revoke`, {}); if (selected === row.id) onSelect('synthetic-audio-reference');
          if (review?.id === row.id) setReview(undefined); await refresh();
        })}>Revoke {row.name}</button>}
        {row.status === 'revoked' && <span>Unavailable · deletion requested</span>}
      </li>)}</ul>
      {cursor && <button disabled={busy} onClick={() => void act(async () => {
        const data = await api<{ items: Asset[]; next_cursor: string | null }>(`/v1/assets?after=${cursor}`);
        setAssets(previous => [...previous, ...data.items.filter(row => !previous.some(old => old.id === row.id))]); setCursor(data.next_cursor);
      })}>More assets</button>}
      {review && <div className="asset-review" aria-label="Asset review">
        <h3>Review {review.name}</h3><img src={`/v1/assets/${review.id}/preview`} alt={`Neutral static preview of ${review.name}`} width={128} height={128} />
        <p>{review.evaluation?.vertices} vertices · {review.evaluation?.faces} triangles · compatibility checked. This does not qualify speaking motion or identity quality.</p>
        <label className="check"><input type="checkbox" checked={accepted} onChange={e => setAccepted(e.target.checked)} />Approve this exact revision for static reference playback only.</label>
        <button disabled={busy || !accepted} onClick={() => void approve()}>Approve reference</button>
      </div>}
    </details>}
    {error && <p role="alert">{error}</p>}
  </section>;
}
