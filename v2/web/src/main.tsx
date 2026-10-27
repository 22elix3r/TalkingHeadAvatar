import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import type { ConversationClient, ServerEvent } from './client';
import './style.css';
import { AssetLibrary } from './AssetLibrary';

type Caption = { id: string; speaker: string; text: string; final: boolean };

function App() {
  const audio = useRef<HTMLAudioElement>(null), video = useRef<HTMLVideoElement>(null);
  const client = useRef<ConversationClient | null>(null);
  const [active, setActive] = useState(false), [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('Ready when you are'), [error, setError] = useState('');
  const [blocked, setBlocked] = useState(false), [muted, setMuted] = useState(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]), [device, setDevice] = useState('');
  const [captions, setCaptions] = useState<Caption[]>([]), [text, setText] = useState('');
  const [assetId, setAssetId] = useState('synthetic-audio-reference');
  const researchAvatar = import.meta.env.VITE_REFERENCE_PROFILE === 'claire_vendor_research' || assetId === 'claire-vendor-research';
  function event(e: ServerEvent) {
    if (e.kind === 'caption' && e.text) {
      const id = e.speaker === 'user' ? `u-${e.utterance}` : `a-${e.epoch}-${e.segment}`;
      setCaptions(previous => [...previous.filter(c => c.id !== id), { id, speaker: e.speaker ?? 'assistant',
        text: e.text!, final: e.final ?? false }].slice(-40));
    }
    if (e.kind === 'closed') setActive(false);
  }
  useEffect(() => () => { void client.current?.stop(); }, []);
  async function start(typed = false) {
    setBusy(true); setError(''); setMuted(false);
    if (client.current) await client.current.stop();
    const { ConversationClient } = await import('./client');
    const next = new ConversationClient({ event, status: setStatus, error: setError, audioBlocked: setBlocked, devices: setDevices }, audio.current!, video.current!);
    client.current = next;
    try { if (typed) await next.startTyped(assetId); else await next.start(device, assetId); setActive(true); }
    catch (e) {
      if (!(e instanceof DOMException && e.name === 'AbortError')) setError(e instanceof Error ? e.message : 'Unable to start. Please retry.');
      await next.stop();
    }
    finally { setBusy(false); }
  }
  async function end() { setBusy(true); await client.current?.stop(); setActive(false); setBusy(false); }
  async function send(e: React.FormEvent) {
    e.preventDefault(); if (!text.trim() || !active) return;
    const message = text.trim(); setText(''); setError('');
    setCaptions(previous => [...previous, { id: crypto.randomUUID(), speaker: 'user', text: message, final: true }].slice(-40));
    setStatus('Thinking');
    try { await client.current?.control('text', message); } catch (e) { setError((e as Error).message); }
  }
  return <main>
    <header><a className="brand" href="/" aria-label="TalkingHead home"><span className="brand-mark">t.</span> talkinghead<span className="research">RESEARCH</span></a><span className="local-badge">● Local preview</span></header>
    <section className="intro"><p className="eyebrow">A LITTLE MORE HUMAN</p><h1>Make room for a conversation.</h1><p>Speak naturally, pause, and pick up where you left off.</p></section>
    <AssetLibrary selected={assetId} onSelect={setAssetId} disabled={busy || active} />
    <div className="workspace">
      <section className="voice-card" aria-label="Voice conversation">
        <div className="card-top"><span className="pill">{researchAvatar ? 'RESEARCH AVATAR' : 'AUDIO REFERENCE'}</span><span className={active ? 'connection online' : 'connection'}>{active ? 'Connected' : 'Not connected'}</span></div>
        <div className={'presence' + (researchAvatar ? ' presence-avatar' : '')}>
          <div className={'orb ' + (status === 'Speaking' ? 'speaking' : '')} hidden={researchAvatar}><span>◎</span></div>
          <video ref={video} muted playsInline aria-label={researchAvatar ? 'Vendor research avatar' : 'Static timing reference; speaking avatar unavailable'} />
          <h2>{researchAvatar ? 'Claire research preview' : 'Synthetic demo voice'}</h2><p>{researchAvatar ? 'Vendor sample with a synthetic voice.' : 'A generated voice for this local research preview.'}</p>
        </div>
        <div className="status" role="status" aria-live="polite"><span className={active ? 'dot live' : 'dot'} />{status}</div>
        <div className="avatar-note"><div><strong>{researchAvatar ? 'Research reference' : 'Speaking avatar coming next'}</strong><p>{researchAvatar ? 'Visual quality and audio/video alignment are still being evaluated.' : 'This preview uses audio and a static reference. Facial animation has not been qualified.'}</p></div></div>
        <label className="device-label" htmlFor="microphone">Microphone</label><select id="microphone" value={device} disabled={busy} onChange={async e => { setDevice(e.target.value); if (active) { await end(); setStatus('Device changed. Start a new conversation.'); } }}><option value="">System default</option>{devices.map(d => <option key={d.deviceId} value={d.deviceId}>{d.label || 'Microphone'}</option>)}</select>
        <div className="actions">{active ? <><button className="primary" disabled={busy} onClick={() => void end()}>End conversation</button><button className="secondary" onClick={async () => { try { await client.current?.control('interrupt'); setStatus('Listening'); } catch (e) { setError((e as Error).message); } }}>Interrupt</button></> : <button className="primary" disabled={busy} onClick={() => void start()}>{busy ? 'Connecting…' : 'Start conversation'}<span aria-hidden="true"> ↗</span></button>}</div>
        {busy && !active ? <button className="text-button" onClick={() => void end()}>Cancel connection</button> : null}
        {active ? <label className="mute"><input type="checkbox" checked={muted} onChange={async e => { const value = e.target.checked; setMuted(value); try { await client.current?.muteMicrophone(value); } catch (e) { setError((e as Error).message); } }} />Mute microphone</label> : <button className="text-button" disabled={busy} onClick={() => void start(true)}>Use typed messages instead</button>}
        {blocked ? <button className="audio-unlock" onClick={() => void client.current?.enableAudio()}>Enable audio playback</button> : null}
        {error ? <div className="error" role="alert">{error}</div> : null}
      </section>
      <section className="transcript-card" aria-label="Conversation captions"><div className="transcript-title"><h2>Conversation</h2><span>CAPTIONS</span></div>
        <div className="captions" aria-live="polite" aria-relevant="additions text">{captions.length === 0 ? <div className="empty"><span aria-hidden="true">〰</span><h3>Your words belong here.</h3><p>Start with a question, or use a typed message.<br/>Live captions appear as you speak.</p></div> : captions.map(c => <article className={'caption ' + c.speaker} key={c.id}><span>{c.speaker === 'user' ? 'YOU' : 'VOICE'}{c.speaker === 'user' && !c.final ? ' · listening' : ''}</span><p>{c.text}</p></article>)}</div>
        <form onSubmit={send}><label className="sr-only" htmlFor="message">Your message</label><input id="message" value={text} onChange={e => setText(e.target.value)} maxLength={1500} placeholder={active ? 'Or type a message…' : 'Connect to send a message'} disabled={!active} /><button className="send" aria-label="Send message" disabled={!active || !text.trim()}>↑</button></form>
        <p className="privacy">Local research mode · Recording is off · Captions stay in memory</p>
      </section>
    </div><footer><span>{researchAvatar ? 'Research avatar · synthetic voice' : 'A voice first. An avatar when it is ready.'}</span><span>Qwen3-TTS · Synthetic reference</span></footer>
    <audio ref={audio} autoPlay />
  </main>;
}

createRoot(document.getElementById('root')!).render(<App />);
