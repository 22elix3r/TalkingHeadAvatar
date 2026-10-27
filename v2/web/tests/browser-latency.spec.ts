import { test, expect } from '@playwright/test';
import { resolve } from 'node:path';
import { writeFile } from 'node:fs/promises';

test('controlled client graph endpoint to received audio latency', async ({ page, context }) => {
  let sid = '', csrf = '';
  page.on('response', async response => {
    if (response.url().endsWith('/v1/dev-login')) csrf = (await response.json()).csrf_token;
    if (response.url().endsWith('/v1/sessions') && response.status() === 202) sid = (await response.json()).id;
  });
  await page.addInitScript(() => {
    type Input = { first_signal_ms: number | null; last_signal_ms: number | null; poll_ms: number; fft_samples: number; sample_rate: number };
    const input: Input[] = [], output: { kind: string; time_ms: number }[] = [];
    const state = window as unknown as { __graphTiming: { input: Input[]; output: { kind: string; time_ms: number }[] } };
    state.__graphTiming = { input, output };
    window.addEventListener('tha-observation', e => output.push((e as CustomEvent).detail));
    const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async constraints => {
      const stream = await original(constraints);
      if (stream.getAudioTracks().length) {
        const clock = new AudioContext(); const source = clock.createMediaStreamSource(stream);
        const analyser = clock.createAnalyser(); analyser.fftSize = 1024; source.connect(analyser);
        const samples = new Float32Array(1024);
        const row: Input = { first_signal_ms: null, last_signal_ms: null, poll_ms: 20, fft_samples: 1024, sample_rate: clock.sampleRate };
        input.push(row); void clock.resume();
        const timer = setInterval(() => {
          if (stream.getAudioTracks().every(track => track.readyState === 'ended')) {
            clearInterval(timer); void clock.close(); return;
          }
          analyser.getFloatTimeDomainData(samples);
          if (samples.some(sample => Math.abs(sample) > .001)) {
            row.first_signal_ms ??= performance.now(); row.last_signal_ms = performance.now();
          }
        }, 20);
      }
      return stream;
    };
  });
  try {
    await page.goto('/'); await page.getByRole('button', { name: 'Start conversation' }).click();
    await expect.poll(() => page.evaluate(() => (window as unknown as { __graphTiming: { output: { kind: string }[] } }).__graphTiming.output.some(e => e.kind === 'decoded_audio_observed')), { timeout: 90000 }).toBe(true);
    const data = await page.evaluate(() => (window as unknown as { __graphTiming: { input: { last_signal_ms: number | null }[]; output: { kind: string; time_ms: number }[] } }).__graphTiming);
    const incoming = data.input.filter(row => row.last_signal_ms !== null).at(-1)!;
    const outgoing = data.output.find(row => row.kind === 'decoded_audio_observed')!;
    const latency = outgoing.time_ms - incoming.last_signal_ms!;
    expect(latency).toBeGreaterThan(1200);
    await writeFile(resolve('../var/p04/client-graph-latency.json'), JSON.stringify({ schema_version: 1,
      evidence_category: 'local_observation', clock_id: 'browser_performance', sample_count: 1,
      fixture: 'synthetic microphone: What is two plus two?', input: incoming, output: outgoing,
      observed_ms: latency, boundary: 'last nonzero microphone WebAudio observation to first nonzero received WebAudio observation',
      uncertainty: '20 ms polling on each graph, FFT window; driver, graph buffering and acoustic offset uncalibrated',
      client_annotated_acoustic_endpoint: false, meaningful_word_onset_annotated: false,
      audiovisual_slo_qualified: false, human_quality_qualified: false }, null, 2));
  } finally {
    if (sid && csrf) await context.request.post(`/v1/sessions/${sid}/close`, { data: {}, headers: { Origin: 'http://127.0.0.1:18700', 'X-CSRF-Token': csrf } });
  }
});
