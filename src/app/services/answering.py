import math
from dataclasses import dataclass

from app.services.faithfulness import SupportValidator
from app.services.llm import EvidencePromptItem, GeneratedAnswer
from app.services.reranking import RerankedHit

INSUFFICIENT_EVIDENCE_ANSWER = (
    "I don't have enough evidence in the selected documents to answer that question."
)


@dataclass(frozen=True)
class AnswerCitation:
    citation_number: int
    evidence_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_id: str


@dataclass(frozen=True)
class UnsupportedClaim:
    text: str
    reason: str


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    citations: list[AnswerCitation]
    unsupported_claims: list[UnsupportedClaim]
    citation_correctness: bool
    citation_correctness_score: float
    correct_citation_count: int
    total_citation_count: int
    faithfulness_score: float
    confidence: float
    insufficient_evidence: bool


def evidence_prompt_items(hits: list[RerankedHit]) -> list[EvidencePromptItem]:
    return [
        EvidencePromptItem(
            evidence_id=f"E{index}",
            filename=item.hit.filename,
            page_number=item.hit.page_number,
            content=item.hit.content,
        )
        for index, item in enumerate(hits, start=1)
    ]


def _probability(score: float) -> float:
    if score >= 0:
        return 1 / (1 + math.exp(-score))
    exponent = math.exp(score)
    return exponent / (1 + exponent)


def validate_and_build_answer(
    generated: GeneratedAnswer,
    hits: list[RerankedHit],
    support_validator: SupportValidator,
    *,
    support_threshold: float,
) -> GroundedAnswer:
    evidence_by_id = {f"E{index}": item for index, item in enumerate(hits, start=1)}
    supported: list[tuple[str, list[str], float]] = []
    unsupported: list[UnsupportedClaim] = []
    correct_citation_count = 0
    total_citation_count = 0

    for claim in generated.claims:
        if not claim.text.strip():
            total_citation_count += 1
            unsupported.append(UnsupportedClaim(text=claim.text, reason="Claim text is empty"))
            continue
        citation_ids = list(dict.fromkeys(claim.evidence_ids))
        if not citation_ids:
            total_citation_count += 1
            unsupported.append(UnsupportedClaim(text=claim.text, reason="Claim has no citation"))
            continue

        total_citation_count += len(citation_ids)
        valid_ids = [evidence_id for evidence_id in citation_ids if evidence_id in evidence_by_id]
        invalid_ids = [
            evidence_id for evidence_id in citation_ids if evidence_id not in evidence_by_id
        ]
        if not valid_ids:
            unsupported.append(
                UnsupportedClaim(
                    text=claim.text,
                    reason=f"Unknown evidence reference: {', '.join(invalid_ids)}",
                )
            )
            continue

        passages = [evidence_by_id[evidence_id].hit.content for evidence_id in valid_ids]
        scores = support_validator.score(claim.text, passages)
        if len(scores) != len(passages):
            raise ValueError("Reranker returned an unexpected number of support scores")
        probabilities = scores
        supported_ids = [
            evidence_id
            for evidence_id, probability in zip(valid_ids, probabilities, strict=True)
            if probability >= support_threshold
        ]
        correct_citation_count += len(supported_ids)
        support = max(probabilities, default=0.0)
        if not supported_ids:
            unsupported.append(
                UnsupportedClaim(
                    text=claim.text,
                    reason=f"Evidence support score {support:.3f} is below the threshold",
                )
            )
            continue
        supported.append((claim.text.strip(), supported_ids, support))

    citation_correctness_score = (
        correct_citation_count / total_citation_count if total_citation_count else 1.0
    )
    all_citations_valid = correct_citation_count == total_citation_count

    if generated.insufficient_evidence or not supported:
        faithfulness = (
            sum(item[2] for item in supported) / len(generated.claims) if generated.claims else 0.0
        )
        return GroundedAnswer(
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            citations=[],
            unsupported_claims=unsupported,
            citation_correctness=all_citations_valid,
            citation_correctness_score=citation_correctness_score,
            correct_citation_count=correct_citation_count,
            total_citation_count=total_citation_count,
            faithfulness_score=faithfulness,
            confidence=0.0,
            insufficient_evidence=True,
        )

    used_ids = list(
        dict.fromkeys(
            evidence_id for _, citation_ids, _ in supported for evidence_id in citation_ids
        )
    )
    citation_number = {evidence_id: index for index, evidence_id in enumerate(used_ids, start=1)}
    answer_parts = []
    for text, citation_ids, _ in supported:
        markers = "".join(f"[{citation_number[evidence_id]}]" for evidence_id in citation_ids)
        answer_parts.append(f"{text.rstrip('.')} {markers}.")

    citations = []
    for evidence_id in used_ids:
        item = evidence_by_id[evidence_id]
        citations.append(
            AnswerCitation(
                citation_number=citation_number[evidence_id],
                evidence_id=evidence_id,
                document_id=item.hit.document_id,
                filename=item.hit.filename,
                page_number=item.hit.page_number,
                chunk_id=item.hit.chunk_id,
            )
        )

    faithfulness = sum(item[2] for item in supported) / len(generated.claims)
    retrieval_confidence = sum(
        _probability(evidence_by_id[item].rerank_score) for item in used_ids
    ) / len(used_ids)
    confidence = min(faithfulness, retrieval_confidence)
    return GroundedAnswer(
        answer=" ".join(answer_parts),
        citations=citations,
        unsupported_claims=unsupported,
        citation_correctness=all_citations_valid,
        citation_correctness_score=citation_correctness_score,
        correct_citation_count=correct_citation_count,
        total_citation_count=total_citation_count,
        faithfulness_score=faithfulness,
        confidence=confidence,
        insufficient_evidence=False,
    )
