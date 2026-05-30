import time

import numpy as np

from tts_engine.audio_bridge import TTSAudioBridge


def test_tts_audio_bridge_emits_fixed_hubert_chunks():
    consumed = []

    def consumer(chunk):
        consumed.append(chunk.copy())

    bridge = TTSAudioBridge(audio_consumer=consumer, hubert_chunk_size=4, queue_maxsize=4)
    bridge.start()
    try:
        bridge.put_chunk(np.array([1.0, 2.0], dtype=np.float32))
        bridge.put_chunk(np.array([3.0, 4.0, 5.0], dtype=np.float32))
        time.sleep(0.2)
    finally:
        bridge.stop()

    assert len(consumed) == 1
    assert np.allclose(consumed[0], np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

