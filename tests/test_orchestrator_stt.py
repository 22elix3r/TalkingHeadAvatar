from orchestrator.stt import RollingTextDeduplicator, normalize_transcript_text


def test_normalize_transcript_text_strips_case_and_punctuation():
    text = "Hello,   Team! Let's ship."
    assert normalize_transcript_text(text) == "hello team lets ship"


def test_deduplicator_drops_exact_repetition():
    dedupe = RollingTextDeduplicator()
    assert dedupe.filter("We should ship this today.") == "We should ship this today."
    assert dedupe.filter("We should ship this today.") == ""


def test_deduplicator_trims_overlapping_prefix():
    dedupe = RollingTextDeduplicator(min_overlap_words=2)
    assert dedupe.filter("We should ship this today") == "We should ship this today"
    assert (
        dedupe.filter("ship this today after lunch")
        == "after lunch"
    )


def test_deduplicator_drops_subset_repeat():
    dedupe = RollingTextDeduplicator(min_overlap_words=2)
    assert dedupe.filter("I already covered the migration plan in detail") == (
        "I already covered the migration plan in detail"
    )
    assert dedupe.filter("migration plan in detail") == ""

