"""One generation of Qwen PCM -> paired playback/motion packets; no devices."""
from dataclasses import dataclass
import struct

from talkinghead_v2.audio.clock import SampleLedger
from talkinghead_v2.audio.resample import RationalResampler
from talkinghead_v2.contracts import AudioChunk, AudioEOF, ContractError, GenerationKey


@dataclass(frozen=True)
class PairedAudioPacket:
    key: GenerationKey
    sequence: int
    start_sample: int  # 48 kHz playback domain; motion starts at start_sample/3.
    audio: tuple[int, ...]
    motion: tuple[float, ...]
    eof: bool

    def packed(self) -> bytes:
        return (struct.pack('<cQQQII', b'F' if self.eof else b'D', self.key.epoch, self.sequence,
                            self.start_sample, len(self.audio), len(self.motion)) +
                struct.pack(f'<{len(self.audio)}h', *self.audio) +
                struct.pack(f'<{len(self.motion)}f', *self.motion))


class ClaireInputMux:
    """Persistent FIR histories and exact paired EOF, bounded to 100 ms input.

    Each result packet holds at most 20 ms. The caller must await sink
    backpressure before accepting the next provider event. Stale generations
    and cancellation cannot flush a filter tail into the active stream.
    """
    def __init__(self, key: GenerationKey):
        self.key = key
        self.ledger = SampleLedger(key, 24000)
        self.playback = RationalResampler(24000, 48000)
        self.motion = RationalResampler(24000, 16000)
        self._audio_pending = ()
        self._motion_pending = ()
        self.sequence = self.sent_samples = self.sent_motion_samples = 0
        self.cancelled = False

    def accept(self, event: AudioChunk | AudioEOF) -> tuple[PairedAudioPacket, ...]:
        if not isinstance(event, (AudioChunk, AudioEOF)):
            raise ContractError('paired Claire input requires audio or explicit EOF')
        if self.cancelled or event.envelope.key != self.key:
            return ()
        if isinstance(event, AudioChunk):
            if len(event.samples) > 2400 or event.end_sample > 24000 * 120:
                raise ContractError('paired Claire input exceeds 100-ms chunk / 120-second turn bound')
        elif event.valid_end_sample != event.end_sample or not event.end_sample:
            raise ContractError('paired Claire EOF requires nonempty whole source samples')
        self.ledger.accept(event)
        if isinstance(event, AudioChunk):
            audio = self.playback.push(event.samples)
            motion = self.motion.push(event.samples)
            final = False
        else:
            audio, motion, final = self.playback.finish(), self.motion.finish(), True
        self._audio_pending += audio
        self._motion_pending += motion
        packets = []
        while len(self._audio_pending) >= 960 and len(self._motion_pending) >= 320:
            packets.append(self._packet(960, 320, False))
        if final:
            if (len(self._audio_pending) >= 960 or
                    len(self._motion_pending) != (len(self._audio_pending) + 2) // 3):
                raise ContractError('paired Claire resampler EOF conservation failed')
            packets.append(self._packet(len(self._audio_pending), len(self._motion_pending), True))
        return tuple(packets)

    def _packet(self, audio_count, motion_count, eof):
        motion = self._motion_pending[:motion_count]
        if any(abs(value) > 4 for value in motion):
            self.cancel()
            raise ContractError('paired Claire motion amplitude exceeds qualified bound')
        audio = tuple(int(max(-1., min(1., value)) * 32767) for value in self._audio_pending[:audio_count])
        packet = PairedAudioPacket(self.key, self.sequence, self.sent_samples, audio, motion, eof)
        self.sequence += 1
        self.sent_samples += audio_count
        self.sent_motion_samples += motion_count
        self._audio_pending = self._audio_pending[audio_count:]
        self._motion_pending = self._motion_pending[motion_count:]
        return packet

    def cancel(self):
        self.cancelled = True
        self.playback.cancel()
        self.motion.cancel()
        self._audio_pending = self._motion_pending = ()
