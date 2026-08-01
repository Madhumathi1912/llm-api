from app.services.chunking import Chunker


def test_short_text_returns_single_chunk():
    """Text shorter than chunk_size should come back as exactly one chunk."""
    chunker = Chunker(chunk_size=100, overlap=10)
    result = chunker.chunk("This is a short sentence.")
    assert len(result) == 1
    assert result[0] == "This is a short sentence."


def test_long_text_splits_into_multiple_chunks():
    """Text longer than chunk_size should be split into more than one chunk."""
    chunker = Chunker(chunk_size=50, overlap=10)
    long_text = "A" * 200
    result = chunker.chunk(long_text)
    assert len(result) > 1


def test_chunks_overlap_correctly():
    """Consecutive chunks should share the configured overlap amount of content."""
    chunker = Chunker(chunk_size=20, overlap=5)
    text = "0123456789" * 5 # 50 characters total, deterministic content
    chunks = chunker.chunk(text)

    # The end of chunk[0] should reappear at the start of chunk[1],
    # proving the overlap window actually works.
    overlap_from_chunk_0 = chunks[0][-5:]
    start_of_chunk_1 = chunks[1][:5]
    assert overlap_from_chunk_0 == start_of_chunk_1


def test_invalid_chunk_size_raises():
    """chunk_size must be greater than overlap — this should be enforced at construction."""
    try:
        Chunker(chunk_size=10, overlap=10)
        assert False, "Expected ValueError but none was raised"
    except ValueError:
        pass


def test_empty_text_returns_empty_value():
    chunker = Chunker(chunk_size=500, overlap=10)
    result = chunker.chunk("")
    assert result == []