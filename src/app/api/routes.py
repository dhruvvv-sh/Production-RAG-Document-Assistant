from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.models import Chunk, Document
from app.schemas import (
    CitationResponse,
    DocumentResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    QueryTiming,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResult,
    RetrievalTiming,
    UnsupportedClaimResponse,
)
from app.services.answering import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    evidence_prompt_items,
    validate_and_build_answer,
)
from app.services.chunking import TokenChunker
from app.services.embeddings import Embedder, get_embedder
from app.services.faithfulness import SupportValidator, get_support_validator
from app.services.ingestion import ingest_pdf
from app.services.llm import LLMProvider, LLMProviderError, get_llm_provider
from app.services.pdf import InvalidPdfError
from app.services.reranking import Reranker, get_reranker, rerank_hits
from app.services.retrieval import hybrid_search

router = APIRouter()


def to_response(db: Session, document: Document) -> DocumentResponse:
    chunk_count = (
        db.scalar(select(func.count(Chunk.id)).where(Chunk.document_id == document.id)) or 0
    )
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        status=document.status,
        page_count=document.page_count,
        chunk_count=chunk_count,
        size_bytes=document.size_bytes,
        created_at=document.created_at,
    )


@router.get("/health/live", response_model=HealthResponse, tags=["health"])
def liveness() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=HealthResponse, tags=["health"])
def readiness(db: Annotated[Session, Depends(get_db)]) -> HealthResponse:
    try:
        db.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Database is unavailable") from exc
    return HealthResponse(status="ok")


@router.post(
    "/api/v1/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
async def upload_document(
    file: Annotated[UploadFile, File(description="A text-based PDF document")],
    response: Response,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> DocumentResponse:
    filename = file.filename or "document.pdf"
    if file.content_type != "application/pdf" and not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported")

    content = await file.read(settings.max_upload_size_bytes + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail="The uploaded file exceeds the size limit")

    chunker = TokenChunker(
        min_tokens=settings.chunk_min_tokens,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    try:
        document, created = ingest_pdf(
            db,
            filename=filename,
            content_type="application/pdf",
            content=content,
            chunker=chunker,
            embedder=embedder,
        )
    except InvalidPdfError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not created:
        response.status_code = status.HTTP_200_OK
    return to_response(db, document)


@router.get("/api/v1/documents/{document_id}", response_model=DocumentResponse, tags=["documents"])
def get_document(document_id: str, db: Annotated[Session, Depends(get_db)]) -> DocumentResponse:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return to_response(db, document)


@router.post(
    "/api/v1/retrieval/search",
    response_model=RetrievalResponse,
    tags=["retrieval"],
)
def search_documents(
    request: RetrievalRequest,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    reranker: Annotated[Reranker, Depends(get_reranker)],
) -> RetrievalResponse:
    started_at = perf_counter()
    dense_weight = (
        request.dense_weight
        if request.dense_weight is not None
        else settings.retrieval_dense_weight
    )
    candidate_k = max(
        request.top_k,
        request.candidate_k or settings.reranker_candidate_limit,
    )
    candidates = hybrid_search(
        db,
        query=request.query,
        embedder=embedder,
        top_k=candidate_k,
        dense_weight=dense_weight,
        candidate_limit=max(settings.retrieval_candidate_limit, candidate_k),
        document_ids=request.document_ids,
    )
    retrieved_at = perf_counter()
    if request.rerank:
        reranked = rerank_hits(
            request.query,
            candidates,
            reranker,
            top_k=request.top_k,
        )
        results = [
            RetrievalResult(
                **vars(item.hit),
                retrieval_rank=item.retrieval_rank,
                rerank_score=item.rerank_score,
                final_rank=rank,
            )
            for rank, item in enumerate(reranked, start=1)
        ]
    else:
        results = [
            RetrievalResult(
                **vars(hit),
                retrieval_rank=rank,
                rerank_score=None,
                final_rank=rank,
            )
            for rank, hit in enumerate(candidates[: request.top_k], start=1)
        ]
    completed_at = perf_counter()
    return RetrievalResponse(
        query=request.query,
        candidate_count=len(candidates),
        timing=RetrievalTiming(
            retrieval_ms=round((retrieved_at - started_at) * 1000, 3),
            reranking_ms=round((completed_at - retrieved_at) * 1000, 3),
            total_ms=round((completed_at - started_at) * 1000, 3),
        ),
        results=results,
    )


@router.post(
    "/api/v1/query",
    response_model=QueryResponse,
    tags=["answers"],
)
def answer_question(
    request: QueryRequest,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    reranker: Annotated[Reranker, Depends(get_reranker)],
    llm_provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    support_validator: Annotated[SupportValidator, Depends(get_support_validator)],
) -> QueryResponse:
    started_at = perf_counter()
    dense_weight = (
        request.dense_weight
        if request.dense_weight is not None
        else settings.retrieval_dense_weight
    )
    candidate_k = max(
        request.top_k,
        request.candidate_k or settings.reranker_candidate_limit,
    )
    candidates = hybrid_search(
        db,
        query=request.question,
        embedder=embedder,
        top_k=candidate_k,
        dense_weight=dense_weight,
        candidate_limit=max(settings.retrieval_candidate_limit, candidate_k),
        document_ids=request.document_ids,
    )
    retrieved_at = perf_counter()
    reranked = rerank_hits(
        request.question,
        candidates,
        reranker,
        top_k=request.top_k,
    )
    reranked_at = perf_counter()
    evidence = [
        RetrievalResult(
            **vars(item.hit),
            retrieval_rank=item.retrieval_rank,
            rerank_score=item.rerank_score,
            final_rank=rank,
        )
        for rank, item in enumerate(reranked, start=1)
    ]

    if not reranked:
        completed_at = perf_counter()
        return QueryResponse(
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            citations=[],
            evidence=[],
            unsupported_claims=[],
            citation_correctness=True,
            citation_correctness_score=1.0,
            faithfulness_score=0.0,
            confidence=0.0,
            insufficient_evidence=True,
            timing=QueryTiming(
                retrieval_ms=round((retrieved_at - started_at) * 1000, 3),
                reranking_ms=round((reranked_at - retrieved_at) * 1000, 3),
                generation_ms=0.0,
                validation_ms=0.0,
                total_ms=round((completed_at - started_at) * 1000, 3),
            ),
        )

    try:
        generated = llm_provider.generate(
            request.question,
            evidence_prompt_items(reranked),
        )
    except LLMProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    generated_at = perf_counter()
    grounded = validate_and_build_answer(
        generated,
        reranked,
        support_validator,
        support_threshold=settings.claim_support_threshold,
    )
    completed_at = perf_counter()
    return QueryResponse(
        answer=grounded.answer,
        citations=[CitationResponse(**vars(citation)) for citation in grounded.citations],
        evidence=evidence,
        unsupported_claims=[
            UnsupportedClaimResponse(**vars(claim)) for claim in grounded.unsupported_claims
        ],
        citation_correctness=grounded.citation_correctness,
        citation_correctness_score=grounded.citation_correctness_score,
        faithfulness_score=grounded.faithfulness_score,
        confidence=grounded.confidence,
        insufficient_evidence=grounded.insufficient_evidence,
        timing=QueryTiming(
            retrieval_ms=round((retrieved_at - started_at) * 1000, 3),
            reranking_ms=round((reranked_at - retrieved_at) * 1000, 3),
            generation_ms=round((generated_at - reranked_at) * 1000, 3),
            validation_ms=round((completed_at - generated_at) * 1000, 3),
            total_ms=round((completed_at - started_at) * 1000, 3),
        ),
    )
