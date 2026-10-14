import math
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk, Document, DocumentStatus
from app.services.bm25 import BM25Index
from app.services.embeddings import Embedder


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    content: str
    dense_score: float | None
    bm25_score: float | None
    hybrid_score: float


@dataclass(frozen=True)
class _Candidate:
    chunk: Chunk
    filename: str


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if math.isclose(low, high):
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _base_statement(document_ids: list[str] | None):
    statement = (
        select(Chunk, Document.filename)
        .join(Document, Document.id == Chunk.document_id)
        .where(Document.status == DocumentStatus.READY)
    )
    if document_ids:
        statement = statement.where(Document.id.in_(document_ids))
    return statement


def _dense_scores(
    db: Session,
    query_vector: list[float],
    candidates: list[_Candidate],
    document_ids: list[str] | None,
    limit: int,
) -> dict[str, float]:
    dialect = db.get_bind().dialect.name
    if dialect == "postgresql":
        similarity = (1 - Chunk.embedding.cosine_distance(query_vector)).label("similarity")
        statement = (
            select(Chunk.id, similarity)
            .join(Document, Document.id == Chunk.document_id)
            .where(Document.status == DocumentStatus.READY, Chunk.embedding.is_not(None))
            .order_by(Chunk.embedding.cosine_distance(query_vector))
            .limit(limit)
        )
        if document_ids:
            statement = statement.where(Document.id.in_(document_ids))
        return {chunk_id: float(score) for chunk_id, score in db.execute(statement)}

    scored = {
        candidate.chunk.id: _cosine_similarity(query_vector, list(candidate.chunk.embedding))
        for candidate in candidates
        if candidate.chunk.embedding is not None
    }
    return dict(sorted(scored.items(), key=lambda item: item[1], reverse=True)[:limit])


def hybrid_search(
    db: Session,
    *,
    query: str,
    embedder: Embedder,
    top_k: int,
    dense_weight: float,
    candidate_limit: int,
    document_ids: list[str] | None = None,
) -> list[RetrievalHit]:
    rows = db.execute(_base_statement(document_ids)).all()
    candidates = [_Candidate(chunk=row[0], filename=row[1]) for row in rows]
    if not candidates:
        return []

    by_id = {candidate.chunk.id: candidate for candidate in candidates}
    query_vector = embedder.embed_query(query)
    dense_scores = _dense_scores(db, query_vector, candidates, document_ids, candidate_limit)

    bm25 = BM25Index([candidate.chunk.content for candidate in candidates])
    keyword_hits = bm25.search(query, candidate_limit)
    bm25_scores = {candidates[hit.item_index].chunk.id: hit.score for hit in keyword_hits}

    normalized_dense = _normalize(dense_scores)
    normalized_bm25 = _normalize(bm25_scores)
    candidate_ids = set(dense_scores) | set(bm25_scores)
    ranked = sorted(
        candidate_ids,
        key=lambda chunk_id: (
            dense_weight * normalized_dense.get(chunk_id, 0.0)
            + (1 - dense_weight) * normalized_bm25.get(chunk_id, 0.0)
        ),
        reverse=True,
    )[:top_k]

    results: list[RetrievalHit] = []
    for chunk_id in ranked:
        candidate = by_id[chunk_id]
        hybrid_score = dense_weight * normalized_dense.get(chunk_id, 0.0) + (
            1 - dense_weight
        ) * normalized_bm25.get(chunk_id, 0.0)
        results.append(
            RetrievalHit(
                chunk_id=chunk_id,
                document_id=candidate.chunk.document_id,
                filename=candidate.filename,
                page_number=candidate.chunk.page_number,
                content=candidate.chunk.content,
                dense_score=dense_scores.get(chunk_id),
                bm25_score=bm25_scores.get(chunk_id),
                hybrid_score=hybrid_score,
            )
        )
    return results
