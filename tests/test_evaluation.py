from app.evaluation import evaluate_dataset, evaluate_grounding, evaluate_reranking
from app.services.answering import validate_and_build_answer
from app.services.llm import GeneratedAnswer, GeneratedClaim
from app.services.reranking import RerankedHit
from app.services.retrieval import RetrievalHit


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


def test_grounding_evaluation_aggregates_citation_and_faithfulness_metrics():
    class SupportReranker:
        def score(self, query, passages):
            return [0.99 for _ in passages]

    hit = RerankedHit(
        hit=RetrievalHit(
            chunk_id="chunk",
            document_id="document",
            filename="policy.pdf",
            page_number=1,
            content="The policy starts in January.",
            dense_score=1.0,
            bm25_score=1.0,
            hybrid_score=1.0,
        ),
        retrieval_rank=1,
        rerank_score=4.0,
    )
    grounded = validate_and_build_answer(
        GeneratedAnswer(
            insufficient_evidence=False,
            claims=[GeneratedClaim(text="The policy starts in January", evidence_ids=["E1"])],
        ),
        [hit],
        SupportReranker(),
        support_threshold=0.5,
    )

    report = evaluate_grounding([grounded])

    assert report.answer_count == 1
    assert report.citation_correctness == 1.0
    assert report.answer_faithfulness > 0.9
