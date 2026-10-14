import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.services.bm25 import BM25Index
from app.services.embeddings import Embedder, get_embedder
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


def _recall(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved) & relevant) / len(relevant)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dense and hybrid retrieval Recall@k")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--dense-weight", type=float, default=None)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")
    if args.dense_weight is not None and not 0 <= args.dense_weight <= 1:
        parser.error("--dense-weight must be between 0 and 1")

    settings = get_settings()
    dense_weight = (
        args.dense_weight if args.dense_weight is not None else settings.retrieval_dense_weight
    )
    dataset = json.loads(args.dataset.read_text())
    report = evaluate_dataset(dataset, get_embedder(), k=args.k, dense_weight=dense_weight)
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
