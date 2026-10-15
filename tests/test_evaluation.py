from app.evaluation import evaluate_dataset, evaluate_reranking


def test_evaluation_calculates_dense_and_hybrid_recall():
    class ExactEmbedder:
        dimension = 2

        def embed_documents(self, texts):
            return [[1.0, 0.0] if "leave" in text else [0.0, 1.0] for text in texts]

        def embed_query(self, text):
            return [1.0, 0.0]

    dataset = {
        "corpus": [
            {"id": "leave", "text": "annual leave policy"},
            {"id": "backup", "text": "nightly database backup"},
        ],
        "queries": [{"query": "vacation allowance", "relevant_chunk_ids": ["leave"]}],
    }

    report = evaluate_dataset(dataset, ExactEmbedder(), k=1)

    assert report.dense_recall == 1.0
    assert report.hybrid_recall == 1.0


def test_reranking_evaluation_calculates_mrr_precision_and_latency():
    class FlatEmbedder:
        dimension = 2

        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0]

    class RelevantFirstReranker:
        def score(self, query, passages):
            return [1.0 if "answer" in passage else 0.0 for passage in passages]

    dataset = {
        "corpus": [
            {"id": "distractor", "text": "general policy"},
            {"id": "relevant", "text": "the correct answer"},
        ],
        "queries": [{"query": "question", "relevant_chunk_ids": ["relevant"]}],
    }

    report = evaluate_reranking(
        dataset,
        FlatEmbedder(),
        RelevantFirstReranker(),
        result_k=1,
        candidate_k=2,
    )

    assert report.hybrid_mrr == 0.5
    assert report.reranked_mrr == 1.0
    assert report.hybrid_precision_at_k == 0.0
    assert report.reranked_precision_at_k == 1.0
    assert report.average_total_ms >= 0
