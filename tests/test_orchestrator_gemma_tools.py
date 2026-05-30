from orchestrator.gemma_loader import process_gemma_output, split_first_sentence


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

