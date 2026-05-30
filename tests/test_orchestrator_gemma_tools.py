from orchestrator.gemma_loader import (
    clean_avatar_speech_text,
    process_gemma_output,
    resolve_gemma_gguf_path,
    split_first_sentence,
)


def test_split_first_sentence_uses_punctuation_boundary():
    sentence, remainder = split_first_sentence("Hello world. Another line")
    assert sentence == "Hello world."
    assert remainder.strip() == "Another line"


def test_process_gemma_output_parses_tool_call_tags():
    text = (
        "<tool_call>"
        '{"name":"avatar_speak","parameters":{"text":"Test reply","emotion_mode":"engaged"}}'
        "</tool_call>"
    )
    payload = process_gemma_output(text)
    assert payload == {"text": "Test reply", "emotion": "engaged"}


def test_resolve_gemma_gguf_path_selects_preferred_file_from_directory(tmp_path):
    (tmp_path / "gemma-4-E4B-it-Q8_0.gguf").write_bytes(b"q8")
    preferred = tmp_path / "gemma-4-E4B-it-UD-Q4_K_XL.gguf"
    preferred.write_bytes(b"q4")

    assert resolve_gemma_gguf_path(tmp_path) == preferred


def test_process_gemma_output_parses_native_tool_call():
    text = '<|tool_call>call:avatar_speak{text:<|"|>Only this should be spoken.<|"|>,emotion_mode:<|"|>engaged<|"|>}<tool_call|>'

    assert process_gemma_output(text) == {
        "text": "Only this should be spoken.",
        "emotion": "engaged",
    }


def test_clean_avatar_speech_text_strips_verbalized_tool_prefix():
    text = 'name avatar_speak parameters text "Only this should be spoken." emotion_mode neutral'

    assert clean_avatar_speech_text(text) == "Only this should be spoken."
