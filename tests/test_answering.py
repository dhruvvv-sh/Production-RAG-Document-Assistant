from app.services.answering import validate_and_build_answer
from app.services.llm import GeneratedAnswer, GeneratedClaim
from app.services.reranking import RerankedHit
from app.services.retrieval import RetrievalHit


class FixedSupportValidator:
    def __init__(self, score: float):
        self.value = score

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [self.value for _ in passages]


def evidence_hit() -> RerankedHit:
    return RerankedHit(
        hit=RetrievalHit(
            chunk_id="chunk-1",
            document_id="document-1",
            filename="policy.pdf",
            page_number=12,
            content="The policy starts on 1 January 2026.",
            dense_score=0.8,
            bm25_score=2.0,
            hybrid_score=0.9,
        ),
        retrieval_rank=1,
        rerank_score=3.0,
    )


def test_supported_claim_gets_server_constructed_citation():
    generated = GeneratedAnswer(
        insufficient_evidence=False,
        claims=[
            GeneratedClaim(
                text="The policy starts on 1 January 2026.",
                evidence_ids=["E1"],
            )
        ],
    )

    result = validate_and_build_answer(
        generated,
        [evidence_hit()],
        FixedSupportValidator(0.99),
        support_threshold=0.5,
    )

    assert result.insufficient_evidence is False
    assert result.citation_correctness is True
    assert result.citation_correctness_score == 1.0
    assert result.citations[0].page_number == 12
    assert result.citations[0].chunk_id == "chunk-1"
    assert result.answer.endswith("[1].")


def test_unknown_citation_is_rejected_and_claim_removed():
    generated = GeneratedAnswer(
        insufficient_evidence=False,
        claims=[GeneratedClaim(text="Unsupported statement", evidence_ids=["E99"])],
    )

    result = validate_and_build_answer(
        generated,
        [evidence_hit()],
        FixedSupportValidator(0.99),
        support_threshold=0.5,
    )

    assert result.insufficient_evidence is True
    assert result.citation_correctness is False
    assert result.citation_correctness_score == 0.0
    assert result.citations == []
    assert result.unsupported_claims[0].text == "Unsupported statement"


def test_low_support_claim_is_removed():
    generated = GeneratedAnswer(
        insufficient_evidence=False,
        claims=[GeneratedClaim(text="The policy was cancelled", evidence_ids=["E1"])],
    )

    result = validate_and_build_answer(
        generated,
        [evidence_hit()],
        FixedSupportValidator(0.01),
        support_threshold=0.5,
    )

    assert result.insufficient_evidence is True
    assert result.citation_correctness is False
    assert result.unsupported_claims
    assert result.faithfulness_score == 0.0
