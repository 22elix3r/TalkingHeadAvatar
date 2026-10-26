import { Room, RoomEvent, Track, createLocalAudioTrack, type LocalAudioTrack, type RemoteTrack,
  type RemoteTrackPublication, type RemoteParticipant } from 'livekit-client';
import { PlaybackGate, type Binding } from './playback';

export type ServerEvent = { kind: string; epoch: number; text?: string; speaker?: string; final?: boolean;
  utterance?: number; revision?: number; segment?: number; code?: string; [key: string]: unknown };
type Session = { id: string; status: string; epoch: number; error_code?: string };
type Hooks = { event: (e: ServerEvent) => void; status: (s: string) => void; error: (e: string) => void;
  audioBlocked: (v: boolean) => void; devices: (devices: MediaDeviceInfo[]) => void };

function observation(kind: string, data: Record<string, unknown> = {}) {
  window.dispatchEvent(new CustomEvent('tha-observation', { detail: { kind, clock_id: 'browser_performance',
    time_ms: performance.now(), ...data } }));
}

const friendly: Record<string, string> = {
  capacity_exhausted: 'Another local conversation is active. End it before starting a new one.',
  providers_not_ready: 'The voice service is starting. Please try again shortly.',
  session_closed: 'This conversation ended. Start a new one to reconnect.',
  preparation_failed: 'The voice could not start. Please retry.',
  microphone_discontinuity: 'Microphone audio was interrupted. Start a new conversation.',
  lease_lost: 'The local service lost its connection. Start again when it is available.',
};

export class ConversationClient {
  private room = new Room({ adaptiveStream: false, dynacast: false });
  private microphone?: LocalAudioTrack;
  private session?: Session;
  private csrf = '';
  private sequence = 0;
  private controls: Promise<unknown> = Promise.resolve();
  private sse?: EventSource;
  private active = false;
  private stopping = false;
  private recoveryRequired = false;
  private gate = new PlaybackGate();
  private tracks = new Map<string, { track: RemoteTrack; publisher: string }>();
  private monitor?: AudioContext;
  private monitorTimer?: ReturnType<typeof setInterval>;
  private heartbeatTimer?: ReturnType<typeof setInterval>;
  private playingSid?: string;
  private onOffline = () => this.requireReconnect('Network connection interrupted. Start again after reconnecting.');

  constructor(private hooks: Hooks, private audio: HTMLAudioElement, private video: HTMLVideoElement) {
    window.addEventListener('offline', this.onOffline);
    this.room.on(RoomEvent.TrackPublished, (publication, participant) => {
      publication.setSubscribed(this.gate.accepts(publication.trackSid, participant.identity));
    });
    this.room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
      this.tracks.set(publication.trackSid, { track, publisher: participant.identity });
      this.attach(publication.trackSid);
    });
    this.room.on(RoomEvent.TrackUnsubscribed, (track, publication) => {
      track.detach(); this.tracks.delete(publication.trackSid);
    });
    this.room.on(RoomEvent.Reconnecting, () => {
      observation('reconnecting');
      this.requireReconnect('Connection interrupted. Start a new conversation to reconnect safely.');
    });
    this.room.on(RoomEvent.Disconnected, () => {
      this.gate.block(); this.detach(); this.microphone?.stop();
      if (!this.stopping) {
        this.hooks.error('Connection closed. Start a new conversation to reconnect.');
        this.hooks.status('Disconnected'); this.active = false;
      }
    });
  }

  private async request<T>(path: string, body?: unknown, headers: Record<string, string> = {}): Promise<T> {
    const response = await fetch(path, { method: body === undefined ? 'GET' : 'POST', credentials: 'same-origin',
      headers: { ...(body === undefined ? {} : { 'Content-Type': 'application/json', 'X-CSRF-Token': this.csrf }), ...headers },
      body: body === undefined ? undefined : JSON.stringify(body) });
    if (!response.ok) {
      const data = await response.json().catch(() => ({ detail: '' }));
      throw new Error(friendly[data.detail] ?? 'The local service could not complete that action. Please retry.');
    }
    return response.json();
  }

  async start(deviceId: string) {
    if (!window.isSecureContext || !navigator.mediaDevices) throw new Error('Microphone access needs a secure local page. Open http://127.0.0.1:18700.');
    this.hooks.status('Checking microphone');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: deviceId ? { deviceId: { exact: deviceId } } : true });
      stream.getTracks().forEach(track => track.stop());
      this.hooks.devices((await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'audioinput'));
    } catch (error) {
      if (error instanceof DOMException && error.name === 'NotAllowedError') throw new Error('Microphone permission was denied. Allow it in your browser, or use typed messages.');
      throw new Error('No working microphone was found. Select another device or use typed messages.');
    }
    await this.connect(deviceId, true);
  }

  async startTyped() { await this.connect('', false); }

  private async connect(deviceId: string, microphone: boolean) {
    if (this.stopping) throw new DOMException('Connection cancelled', 'AbortError');
    this.hooks.status('Preparing voice');
    const login = await this.request<{ csrf_token: string }>('/v1/dev-login', {});
    this.csrf = login.csrf_token;
    this.session = await this.request<Session>('/v1/sessions', { profile: 'audio_reference', asset_id: 'synthetic-audio-reference' },
      { 'Idempotency-Key': crypto.randomUUID() });
    if (this.stopping) { await this.stop(); throw new DOMException('Connection cancelled', 'AbortError'); }
    const deadline = performance.now() + 45000;
    while (this.session.status === 'PREPARING' && performance.now() < deadline) {
      if (this.stopping) throw new DOMException('Connection cancelled', 'AbortError');
      await new Promise(resolve => setTimeout(resolve, 250));
      this.session = await this.request<Session>(`/v1/sessions/${this.session.id}`);
    }
    if (!['READY', 'ACTIVE'].includes(this.session.status)) throw new Error(friendly[this.session.error_code ?? ''] ?? 'The voice is not ready. End this conversation and retry.');
    const join = await this.request<{ url: string; token: string; epoch: number; publisher: string }>(`/v1/sessions/${this.session.id}/join-grant`, {});
    if (this.stopping) throw new DOMException('Connection cancelled', 'AbortError');
    this.gate.advance(join.epoch);
    this.sse = new EventSource(`/v1/sessions/${this.session.id}/events`);
    this.sse.onmessage = e => this.onEvent(JSON.parse(e.data) as ServerEvent);
    this.sse.onerror = () => {
      this.requireReconnect('Control connection interrupted. Restart to reconnect safely.');
    };
    await this.room.connect(join.url, join.token, { autoSubscribe: false });
    this.active = true;
    this.heartbeatTimer = setInterval(() => {
      if (this.session && this.active) void this.request(`/v1/sessions/${this.session.id}/heartbeat`, {}).catch(() => {});
    }, 5000);
    await this.enableAudio();
    if (microphone) {
      this.microphone = await createLocalAudioTrack({ deviceId: deviceId || undefined,
        echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 });
      await this.room.localParticipant.publishTrack(this.microphone, { source: Track.Source.Microphone });
      observation('microphone_published');
    }
    this.hooks.status(microphone ? 'Listening' : 'Ready for a message');
    observation('connected', { microphone });
  }

  private onEvent(event: ServerEvent) {
    if (this.recoveryRequired) return;
    if (!this.gate.advance(event.epoch)) return;
    if (event.kind === 'interrupted' || event.kind === 'closed' || event.kind === 'error') {
      this.gate.cancelTo(event.epoch); this.detach();
      this.updateSubscriptions();
    }
    if (event.kind === 'playback_binding') {
      const binding = event as unknown as Binding;
      if (typeof binding.audio_sid !== 'string' || typeof binding.video_sid !== 'string' || typeof binding.publisher !== 'string') return;
      if (!this.gate.bind(binding)) return;
      this.updateSubscriptions();
      this.attach(binding.audio_sid); this.attach(binding.video_sid);
      observation('playback_binding', { epoch: event.epoch, audio_sid: binding.audio_sid, video_sid: binding.video_sid });
    }
    if (event.kind === 'caption' && event.speaker === 'user' && event.final) this.hooks.status('Thinking');
    if (event.kind === 'sender_complete') {
      this.hooks.status('Listening');
      observation('sender_complete', { epoch: event.epoch, valid_audio_samples: event.valid_audio_samples, presentation_ack: false });
    }
    if (event.kind === 'error') this.hooks.error(friendly[event.code ?? ''] ?? 'The answer was interrupted. Please retry.');
    if (event.kind === 'closed') {
      this.hooks.status('Conversation ended'); this.active = false; this.microphone?.stop();
      this.sse?.close(); this.stopping = true; void this.room.disconnect();
    }
    this.hooks.event(event);
  }

  private updateSubscriptions() {
    this.room.remoteParticipants.forEach(participant => participant.trackPublications.forEach(publication => {
      publication.setSubscribed(this.gate.accepts(publication.trackSid, participant.identity));
    }));
  }

  private attach(sid: string) {
    const entry = this.tracks.get(sid);
    if (!entry || !this.gate.accepts(sid, entry.publisher)) return;
    if (entry.track.kind === Track.Kind.Audio && this.playingSid !== sid) {
      this.detachAudio();
      entry.track.attach(this.audio); this.playingSid = sid;
      this.audio.dataset.trackSid = sid; this.audio.dataset.epoch = String(this.gate.epoch);
      this.audio.muted = false;
      void this.audio.play().catch(() => this.hooks.audioBlocked(true));
      this.monitorAudio(entry.track, sid);
    } else if (entry.track.kind === Track.Kind.Video) {
      entry.track.attach(this.video);
      this.video.dataset.trackSid = sid;
      void this.video.play().catch(() => {});
      if ('requestVideoFrameCallback' in this.video) this.video.requestVideoFrameCallback(() => {
        if (this.gate.accepts(sid, entry.publisher)) observation('video_presented_callback', { epoch: this.gate.epoch, video_sid: sid, visual_mode: 'static_reference' });
      });
    }
  }

  private monitorAudio(track: RemoteTrack, sid: string) {
    this.monitor = new AudioContext();
    const source = this.monitor.createMediaStreamSource(new MediaStream([track.mediaStreamTrack]));
    const analyser = this.monitor.createAnalyser(); analyser.fftSize = 1024; source.connect(analyser);
    const samples = new Float32Array(1024);
    let first = true;
    this.monitorTimer = setInterval(() => {
      if (this.playingSid !== sid || !this.gate.binding || this.gate.binding.audio_sid !== sid) return;
      analyser.getFloatTimeDomainData(samples);
      if (first && samples.some(sample => Math.abs(sample) > .001)) {
        first = false; this.hooks.status('Speaking');
        observation('decoded_audio_observed', { epoch: this.gate.epoch, audio_sid: sid, boundary: 'browser WebAudio graph; physical output unqualified' });
      }
    }, 20);
    void this.monitor.resume().catch(() => this.hooks.audioBlocked(true));
  }

  private detachAudio() {
    this.audio.muted = true; this.audio.pause(); this.audio.srcObject = null;
    this.audio.removeAttribute('src'); this.playingSid = undefined;
    this.tracks.forEach(entry => { if (entry.track.kind === Track.Kind.Audio) entry.track.detach(this.audio); });
    if (this.monitorTimer) clearInterval(this.monitorTimer);
    if (this.monitor) void this.monitor.close().catch(() => {});
    this.monitor = undefined; this.monitorTimer = undefined;
  }
  private detach() {
    this.detachAudio(); this.video.pause(); this.video.srcObject = null;
    this.tracks.forEach(entry => { if (entry.track.kind === Track.Kind.Video) entry.track.detach(this.video); });
  }
  private requireReconnect(message: string) {
    if (this.recoveryRequired || this.stopping) return;
    this.recoveryRequired = true; this.gate.block(); this.detach(); this.microphone?.stop();
    this.hooks.error(message); this.hooks.event({ kind: 'closed', epoch: this.gate.epoch });
    observation('receiver_recovery_required');
    void this.stop();
  }
  async enableAudio() {
    try { await this.room.startAudio(); if (this.playingSid) await this.audio.play(); this.hooks.audioBlocked(false); }
    catch { this.hooks.audioBlocked(true); }
  }
  control(kind: 'interrupt' | 'text', text?: string): Promise<void> {
    if (!this.session || !this.active) return Promise.resolve();
    if (kind === 'interrupt' || kind === 'text') {
      this.gate.block(); this.detach(); observation('local_output_muted', { epoch: this.gate.epoch });
    }
    const operation = async () => {
      if (!this.session || !this.active) return;
      const result = await this.request<{ epoch: number }>(`/v1/sessions/${this.session.id}/controls`, {
        sequence: ++this.sequence, expected_epoch: this.gate.epoch, kind, ...(text ? { text } : {}),
      });
      this.gate.advance(result.epoch);
    };
    const pending = this.controls.then(operation, operation);
    this.controls = pending;
    return pending;
  }
  async muteMicrophone(muted: boolean) {
    if (muted) { await this.microphone?.mute(); await this.control('interrupt'); }
    else await this.microphone?.unmute();
  }
  async stop() {
    this.stopping = true; this.active = false; this.gate.block(); this.detach();
    window.removeEventListener('offline', this.onOffline);
    if (this.heartbeatTimer) clearInterval(this.heartbeatTimer);
    this.microphone?.stop(); this.sse?.close();
    if (this.session) await this.request(`/v1/sessions/${this.session.id}/close`, {}).catch(() => {});
    await this.room.disconnect(); this.tracks.clear();
    this.hooks.status('Conversation ended'); observation('stopped');
  }
}
