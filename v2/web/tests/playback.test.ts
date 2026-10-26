import { describe, expect, it } from 'vitest';
import { PlaybackGate } from '../src/playback';

describe('receiver track replacement', () => {
  it('does not resume from a cancel acknowledgment or late old callbacks', () => {
    const gate = new PlaybackGate();
    gate.bind({ epoch: 1, audio_sid: 'old-audio', video_sid: 'old-video', publisher: 'media' });
    expect(gate.accepts('old-audio', 'media')).toBe(true);
    gate.block();
    expect(gate.accepts('old-audio', 'media')).toBe(false);
    expect(gate.bind({ epoch: 1, audio_sid: 'old-audio', video_sid: 'old-video', publisher: 'media' })).toBe(false);
    gate.advance(2);
    expect(gate.binding).toBe(null);
    gate.bind({ epoch: 3, audio_sid: 'fresh-audio', video_sid: 'fresh-video', publisher: 'media' });
    expect(gate.accepts('old-audio', 'media')).toBe(false);
    expect(gate.accepts('fresh-audio', 'other-session')).toBe(false);
    expect(gate.bind({ epoch: 1, audio_sid: 'old-audio', video_sid: 'old-video', publisher: 'media' })).toBe(false);
    expect(gate.accepts('fresh-audio', 'media')).toBe(true);
  });
  it('allows only a fresh epoch binding after local mute and server cancellation', () => {
    const gate = new PlaybackGate();
    gate.bind({ epoch: 1, audio_sid: 'a', video_sid: 'v', publisher: 'm' });
    gate.block(); gate.cancelTo(2);
    expect(gate.binding).toBe(null);
    expect(gate.bind({ epoch: 2, audio_sid: 'new-a', video_sid: 'new-v', publisher: 'm' })).toBe(true);
  });
});
