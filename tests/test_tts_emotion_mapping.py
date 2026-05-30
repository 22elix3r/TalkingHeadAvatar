from tts_engine.synthesizer import ChatterboxEngine


def test_emotion_mode_mapping_is_stable():
    assert ChatterboxEngine._emotion_to_exaggeration("neutral") == 0.45
    assert ChatterboxEngine._emotion_to_exaggeration("engaged") == 0.65
    assert ChatterboxEngine._emotion_to_exaggeration("emphatic") == 0.9
    assert ChatterboxEngine._emotion_to_exaggeration("concerned") == 0.6
    assert ChatterboxEngine._emotion_to_exaggeration("unknown") == 0.45

