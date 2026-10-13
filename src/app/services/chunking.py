import re
from dataclasses import dataclass

import tiktoken

from app.services.pdf import PdfPage

_BOUNDARY_PATTERN = re.compile(r"\n\s*\n|(?<=[.!?])\s+")


@dataclass(frozen=True)
class TextChunk:
    page_number: int
    content: str
    token_count: int


class TokenChunker:
    def __init__(self, min_tokens: int = 500, max_tokens: int = 800, overlap_tokens: int = 80):
        if not 0 <= overlap_tokens < min_tokens <= max_tokens:
            raise ValueError("Expected 0 <= overlap_tokens < min_tokens <= max_tokens")
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def chunk_pages(self, pages: list[PdfPage]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for page in pages:
            if page.text.strip():
                chunks.extend(self._chunk_page(page))
        return chunks

    def _chunk_page(self, page: PdfPage) -> list[TextChunk]:
        token_ids = self.encoding.encode(page.text)
        if len(token_ids) <= self.max_tokens:
            return [TextChunk(page.number, page.text.strip(), len(token_ids))]

        results: list[TextChunk] = []
        start = 0
        while start < len(token_ids):
            hard_end = min(start + self.max_tokens, len(token_ids))
            end = hard_end
            if hard_end < len(token_ids):
                candidate = self.encoding.decode(token_ids[start:hard_end])
                minimum_character = len(
                    self.encoding.decode(token_ids[start : start + self.min_tokens])
                )
                boundaries = [match.end() for match in _BOUNDARY_PATTERN.finditer(candidate)]
                valid_boundaries = [
                    position for position in boundaries if position >= minimum_character
                ]
                if valid_boundaries:
                    boundary_tokens = self.encoding.encode(candidate[: valid_boundaries[-1]])
                    end = start + len(boundary_tokens)

            content = self.encoding.decode(token_ids[start:end]).strip()
            if content:
                results.append(TextChunk(page.number, content, len(self.encoding.encode(content))))
            if end == len(token_ids):
                break
            start = max(start + 1, end - self.overlap_tokens)
        return results
