from talkinghead_v2.providers.tts.quality import word_errors, words


def test_strict_wer_counts_insertions_deletions_substitutions_and_unicode():
    assert word_errors("one two three", "one four three extra") == {
        "reference_words": 3, "substitutions": 1, "deletions": 0, "insertions": 1, "errors": 2, "wer": 2/3}
    assert word_errors("one two three", "one three")["deletions"] == 1
    assert words("CAFÉ — yes!") == ["café", "yes"]
    assert word_errors("$12.50", "twelve fifty")["errors"] == 2
    assert word_errors("", "extra")["wer"] is None
