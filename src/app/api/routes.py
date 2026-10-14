from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db
from app.models import Chunk, Document
from app.schemas import (
    DocumentResponse,
    HealthResponse,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResult,
)
from app.services.chunking import TokenChunker
from app.services.embeddings import Embedder, get_embedder
from app.services.ingestion import ingest_pdf
from app.services.pdf import InvalidPdfError
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
) -> RetrievalResponse:
    dense_weight = (
        request.dense_weight
        if request.dense_weight is not None
        else settings.retrieval_dense_weight
    )
    hits = hybrid_search(
        db,
        query=request.query,
        embedder=embedder,
        top_k=request.top_k,
        dense_weight=dense_weight,
        candidate_limit=max(settings.retrieval_candidate_limit, request.top_k),
        document_ids=request.document_ids,
    )
    return RetrievalResponse(
        query=request.query,
        results=[RetrievalResult(**vars(hit)) for hit in hits],
    )
