export type Binding = { epoch: number; audio_sid: string; video_sid: string; publisher: string };

/** Receiver authorization is an exact new track identity, never a server cancel ack. */
export class PlaybackGate {
  epoch = 0;
  binding: Binding | null = null;
  private minimumResumeEpoch = 0;
  advance(epoch: number) {
    if (epoch < this.epoch) return false;
    if (epoch > this.epoch) this.binding = null;
    this.epoch = epoch;
    return true;
  }
  block() { this.binding = null; this.minimumResumeEpoch = Math.max(this.minimumResumeEpoch, this.epoch + 1); }
  cancelTo(epoch: number) {
    this.advance(epoch); this.binding = null;
    this.minimumResumeEpoch = Math.max(this.minimumResumeEpoch, epoch);
  }
  bind(value: Binding) {
    if (value.epoch < this.minimumResumeEpoch) return false;
    if (!this.advance(value.epoch)) return false;
    this.binding = value;
    return true;
  }
  accepts(sid: string, publisher: string) {
    return this.binding !== null && publisher === this.binding.publisher &&
      (sid === this.binding.audio_sid || sid === this.binding.video_sid);
  }
}
