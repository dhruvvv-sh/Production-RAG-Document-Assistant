import math
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.casefold())


@dataclass(frozen=True)
class BM25Hit:
    item_index: int
    score: float


class BM25Index:
    """Small in-memory BM25 index rebuilt from the current filtered corpus."""

    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents = [tokenize(document) for document in documents]
        self.term_frequencies = [Counter(document) for document in self.documents]
        self.document_lengths = [len(document) for document in self.documents]
        self.average_length = (
            sum(self.document_lengths) / len(self.document_lengths) if self.documents else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for document in self.documents:
            document_frequency.update(set(document))
        document_count = len(self.documents)
        self.idf = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, limit: int) -> list[BM25Hit]:
        query_terms = tokenize(query)
        if not query_terms or not self.documents:
            return []

        hits: list[BM25Hit] = []
        for index, frequencies in enumerate(self.term_frequencies):
            length = self.document_lengths[index]
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                length_ratio = length / self.average_length if self.average_length else 0.0
                denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                score += self.idf.get(term, 0.0) * (frequency * (self.k1 + 1)) / denominator
            if score > 0:
                hits.append(BM25Hit(item_index=index, score=score))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]
