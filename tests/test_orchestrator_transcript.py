from orchestrator.transcript import MeetingTranscript


class DummyTokenizer:
    @staticmethod
    def encode(text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return text.split()


def test_meeting_transcript_truncates_oldest_entries():
    transcript = MeetingTranscript(max_tokens=6)
    transcript.add("A", "one two")
    transcript.add("B", "three four")
    transcript.add("C", "five six")

    context = transcript.to_context(DummyTokenizer())
    assert "[A]: one two" not in context
    assert "[B]: three four" in context
    assert "[C]: five six" in context


def test_meeting_transcript_truncate_to_last_minutes():
    transcript = MeetingTranscript(max_tokens=100)
    transcript.add("A", "old", timestamp=10.0)
    transcript.add("B", "recent", timestamp=200.0)
    transcript.truncate_to_last_n_minutes(minutes=2, now=260.0)
    assert len(transcript.entries) == 1
    assert transcript.entries[0].text == "recent"
