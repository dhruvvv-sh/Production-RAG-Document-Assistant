from app.services.chunking import TokenChunker
from app.services.pdf import PdfPage


def test_long_page_is_chunked_with_limits_and_page_number():
    chunker = TokenChunker(min_tokens=20, max_tokens=30, overlap_tokens=5)
    text = " ".join(f"Sentence number {index} is useful." for index in range(80))

    chunks = chunker.chunk_pages([PdfPage(number=7, text=text)])

    assert len(chunks) > 1
    assert all(chunk.page_number == 7 for chunk in chunks)
    assert all(chunk.token_count <= 30 for chunk in chunks)
    assert all(chunk.token_count >= 20 for chunk in chunks[:-1])


def test_short_page_stays_in_one_chunk():
    chunker = TokenChunker(min_tokens=20, max_tokens=30, overlap_tokens=5)

    chunks = chunker.chunk_pages([PdfPage(number=2, text="A short page.")])

    assert len(chunks) == 1
    assert chunks[0].content == "A short page."
