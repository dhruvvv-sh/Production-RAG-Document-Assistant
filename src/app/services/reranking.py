from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from sentence_transformers import CrossEncoder

from app.config import get_settings
from app.services.retrieval import RetrievalHit


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Lazy cross-encoder adapter for query-passage relevance scoring."""

    def __init__(self, model_name: str, batch_size: int) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []

        tokenizer = self.model.tokenizer
        query_tokens = tokenizer.encode(query, add_special_tokens=False)
        special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
        window_size = max(1, self.model.max_seq_length - len(query_tokens) - special_tokens)
        overlap = min(64, window_size // 4)
        step = max(1, window_size - overlap)
        pairs: list[tuple[str, str]] = []
        owners: list[int] = []
        for owner, passage in enumerate(passages):
            passage_tokens = tokenizer.encode(passage, add_special_tokens=False)
            if not passage_tokens:
                pairs.append((query, passage))
                owners.append(owner)
                continue
            for start in range(0, len(passage_tokens), step):
                window = passage_tokens[start : start + window_size]
                pairs.append((query, tokenizer.decode(window, skip_special_tokens=True)))
                owners.append(owner)
                if start + window_size >= len(passage_tokens):
                    break

        window_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        scores = [float("-inf")] * len(passages)
        for owner, score in zip(owners, window_scores, strict=True):
            scores[owner] = max(scores[owner], float(score))
        return scores


@dataclass(frozen=True)
class RerankedHit:
    hit: RetrievalHit
    retrieval_rank: int
    rerank_score: float


def rerank_hits(
    query: str,
    hits: list[RetrievalHit],
    reranker: Reranker,
    *,
    top_k: int,
) -> list[RerankedHit]:
    scores = reranker.score(query, [hit.content for hit in hits])
    if len(scores) != len(hits):
        raise ValueError("Reranker returned an unexpected number of scores")
    ranked = [
        RerankedHit(hit=hit, retrieval_rank=index, rerank_score=score)
        for index, (hit, score) in enumerate(zip(hits, scores, strict=True), start=1)
    ]
    return sorted(ranked, key=lambda item: item.rerank_score, reverse=True)[:top_k]


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    settings = get_settings()
    return CrossEncoderReranker(settings.reranker_model, settings.reranker_batch_size)
