import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from app.config import get_settings
from app.services.answering import GroundedAnswer
from app.services.bm25 import BM25Index
from app.services.embeddings import Embedder, get_embedder
from app.services.reranking import Reranker, get_reranker
from app.services.retrieval import _cosine_similarity, _normalize

DEFAULT_DATASET = Path(__file__).with_name("data") / "retrieval_evaluation.json"


@dataclass(frozen=True)
class EvaluationReport:
    query_count: int
    corpus_size: int
    k: int
    dense_recall: float
    hybrid_recall: float
    improvement_percentage_points: float


@dataclass(frozen=True)
class RerankingEvaluationReport:
    query_count: int
    candidate_k: int
    result_k: int
    hybrid_mrr: float
    reranked_mrr: float
    mrr_improvement: float
    mrr_relative_improvement_percent: float
    hybrid_precision_at_k: float
    reranked_precision_at_k: float
    precision_improvement_percentage_points: float
    precision_relative_improvement_percent: float
    average_retrieval_ms: float
    average_reranking_ms: float
    average_total_ms: float


@dataclass(frozen=True)
class GroundingEvaluationReport:
    answer_count: int
    answered_count: int
    citation_count: int
    citation_correctness: float
    answer_faithfulness: float
    insufficient_evidence_rate: float


def _recall(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved) & relevant) / len(relevant)


def _reciprocal_rank(ranking: list[str], relevant: set[str]) -> float:
    for rank, chunk_id in enumerate(ranking, start=1):
        if chunk_id in relevant:
            return 1 / rank
    return 0.0


def _precision_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranking[:k]) & relevant) / k


def evaluate_grounding(answers: list[GroundedAnswer]) -> GroundingEvaluationReport:
    citation_count = sum(answer.total_citation_count for answer in answers)
    correct_citations = sum(answer.correct_citation_count for answer in answers)
    answered = [answer for answer in answers if not answer.insufficient_evidence]
    return GroundingEvaluationReport(
        answer_count=len(answers),
        answered_count=len(answered),
        citation_count=citation_count,
        citation_correctness=(correct_citations / citation_count if citation_count else 0.0),
        answer_faithfulness=(
            sum(answer.faithfulness_score for answer in answered) / len(answered)
            if answered
            else 0.0
        ),
        insufficient_evidence_rate=(
            sum(answer.insufficient_evidence for answer in answers) / len(answers)
            if answers
            else 0.0
        ),
    )


def evaluate_dataset(
    dataset: dict[str, Any],
    embedder: Embedder,
    *,
    k: int = 5,
    dense_weight: float = 0.55,
) -> EvaluationReport:
    corpus = dataset["corpus"]
    queries = dataset["queries"]
    chunk_ids = [item["id"] for item in corpus]
    texts = [item["text"] for item in corpus]
    corpus_vectors = embedder.embed_documents(texts)
    bm25 = BM25Index(texts)

    dense_recalls: list[float] = []
    hybrid_recalls: list[float] = []
    for item in queries:
        query = item["query"]
        relevant = set(item["relevant_chunk_ids"])
        query_vector = embedder.embed_query(query)
        dense_scores = {
            chunk_id: _cosine_similarity(query_vector, vector)
            for chunk_id, vector in zip(chunk_ids, corpus_vectors, strict=True)
        }
        dense_ranking = sorted(dense_scores, key=dense_scores.get, reverse=True)

        bm25_scores = {
            chunk_ids[hit.item_index]: hit.score for hit in bm25.search(query, limit=len(corpus))
        }
        normalized_dense = _normalize(dense_scores)
        normalized_bm25 = _normalize(bm25_scores)
        hybrid_scores = {
            chunk_id: dense_weight * normalized_dense.get(chunk_id, 0.0)
            + (1 - dense_weight) * normalized_bm25.get(chunk_id, 0.0)
            for chunk_id in chunk_ids
        }
        hybrid_ranking = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)
        dense_recalls.append(_recall(dense_ranking[:k], relevant))
        hybrid_recalls.append(_recall(hybrid_ranking[:k], relevant))

    dense_recall = sum(dense_recalls) / len(dense_recalls) if dense_recalls else 0.0
    hybrid_recall = sum(hybrid_recalls) / len(hybrid_recalls) if hybrid_recalls else 0.0
    return EvaluationReport(
        query_count=len(queries),
        corpus_size=len(corpus),
        k=k,
        dense_recall=dense_recall,
        hybrid_recall=hybrid_recall,
        improvement_percentage_points=(hybrid_recall - dense_recall) * 100,
    )


def evaluate_reranking(
    dataset: dict[str, Any],
    embedder: Embedder,
    reranker: Reranker,
    *,
    result_k: int = 5,
    candidate_k: int = 20,
    dense_weight: float = 0.55,
) -> RerankingEvaluationReport:
    corpus = dataset["corpus"]
    queries = dataset["queries"]
    chunk_ids = [item["id"] for item in corpus]
    texts = [item["text"] for item in corpus]
    text_by_id = dict(zip(chunk_ids, texts, strict=True))
    corpus_vectors = embedder.embed_documents(texts)
    bm25 = BM25Index(texts)

    hybrid_rr: list[float] = []
    reranked_rr: list[float] = []
    hybrid_precision: list[float] = []
    reranked_precision: list[float] = []
    retrieval_times: list[float] = []
    reranking_times: list[float] = []

    for item in queries:
        retrieval_started = perf_counter()
        query = item["query"]
        relevant = set(item["relevant_chunk_ids"])
        query_vector = embedder.embed_query(query)
        dense_scores = {
            chunk_id: _cosine_similarity(query_vector, vector)
            for chunk_id, vector in zip(chunk_ids, corpus_vectors, strict=True)
        }
        bm25_scores = {
            chunk_ids[hit.item_index]: hit.score for hit in bm25.search(query, limit=len(corpus))
        }
        normalized_dense = _normalize(dense_scores)
        normalized_bm25 = _normalize(bm25_scores)
        hybrid_scores = {
            chunk_id: dense_weight * normalized_dense.get(chunk_id, 0.0)
            + (1 - dense_weight) * normalized_bm25.get(chunk_id, 0.0)
            for chunk_id in chunk_ids
        }
        hybrid_ranking = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)
        candidates = hybrid_ranking[:candidate_k]
        retrieval_finished = perf_counter()

        rerank_scores = reranker.score(query, [text_by_id[chunk_id] for chunk_id in candidates])
        if len(rerank_scores) != len(candidates):
            raise ValueError("Reranker returned an unexpected number of scores")
        reranked = [
            chunk_id
            for chunk_id, _ in sorted(
                zip(candidates, rerank_scores, strict=True),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        reranking_finished = perf_counter()

        hybrid_rr.append(_reciprocal_rank(hybrid_ranking, relevant))
        reranked_rr.append(_reciprocal_rank(reranked, relevant))
        hybrid_precision.append(_precision_at_k(hybrid_ranking, relevant, result_k))
        reranked_precision.append(_precision_at_k(reranked, relevant, result_k))
        retrieval_times.append((retrieval_finished - retrieval_started) * 1000)
        reranking_times.append((reranking_finished - retrieval_finished) * 1000)

    count = len(queries)
    hybrid_mrr = sum(hybrid_rr) / count if count else 0.0
    reranked_mrr_value = sum(reranked_rr) / count if count else 0.0
    hybrid_precision_value = sum(hybrid_precision) / count if count else 0.0
    reranked_precision_value = sum(reranked_precision) / count if count else 0.0
    average_retrieval = sum(retrieval_times) / count if count else 0.0
    average_reranking = sum(reranking_times) / count if count else 0.0
    return RerankingEvaluationReport(
        query_count=count,
        candidate_k=candidate_k,
        result_k=result_k,
        hybrid_mrr=hybrid_mrr,
        reranked_mrr=reranked_mrr_value,
        mrr_improvement=reranked_mrr_value - hybrid_mrr,
        mrr_relative_improvement_percent=(
            ((reranked_mrr_value - hybrid_mrr) / hybrid_mrr) * 100 if hybrid_mrr else 0.0
        ),
        hybrid_precision_at_k=hybrid_precision_value,
        reranked_precision_at_k=reranked_precision_value,
        precision_improvement_percentage_points=(reranked_precision_value - hybrid_precision_value)
        * 100,
        precision_relative_improvement_percent=(
            ((reranked_precision_value - hybrid_precision_value) / hybrid_precision_value) * 100
            if hybrid_precision_value
            else 0.0
        ),
        average_retrieval_ms=average_retrieval,
        average_reranking_ms=average_reranking,
        average_total_ms=average_retrieval + average_reranking,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dense and hybrid retrieval Recall@k")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--dense-weight", type=float, default=None)
    parser.add_argument("--candidate-k", type=int, default=None)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")
    if args.dense_weight is not None and not 0 <= args.dense_weight <= 1:
        parser.error("--dense-weight must be between 0 and 1")
    if args.candidate_k is not None and args.candidate_k < args.k:
        parser.error("--candidate-k must be at least --k")

    settings = get_settings()
    dense_weight = (
        args.dense_weight if args.dense_weight is not None else settings.retrieval_dense_weight
    )
    dataset = json.loads(args.dataset.read_text())
    report = evaluate_dataset(dataset, get_embedder(), k=args.k, dense_weight=dense_weight)
    reranking_report = evaluate_reranking(
        dataset,
        get_embedder(),
        get_reranker(),
        result_k=args.k,
        candidate_k=args.candidate_k or settings.reranker_candidate_limit,
        dense_weight=dense_weight,
    )
    print(
        json.dumps(
            {"retrieval": asdict(report), "reranking": asdict(reranking_report)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
