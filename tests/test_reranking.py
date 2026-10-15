from dataclasses import replace

from app.services.reranking import rerank_hits
from app.services.retrieval import RetrievalHit


class OrderedReranker:
    def score(self, query: str, passages: list[str]) -> list[float]:
        return [0.1, 0.9, 0.2]


def test_reranker_reorders_candidates_and_preserves_retrieval_rank():
    base = RetrievalHit(
        chunk_id="first",
        document_id="document",
        filename="document.pdf",
        page_number=1,
        content="first passage",
        dense_score=0.9,
        bm25_score=1.0,
        hybrid_score=1.0,
    )
    hits = [
        base,
        replace(base, chunk_id="second", content="second passage", hybrid_score=0.8),
        replace(base, chunk_id="third", content="third passage", hybrid_score=0.7),
    ]

    reranked = rerank_hits("query", hits, OrderedReranker(), top_k=2)

    assert [item.hit.chunk_id for item in reranked] == ["second", "third"]
    assert [item.retrieval_rank for item in reranked] == [2, 3]
    assert [item.rerank_score for item in reranked] == [0.9, 0.2]
