from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models import DocumentStatus


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    status: DocumentStatus
    page_count: int
    chunk_count: int
    size_bytes: int
    created_at: datetime


class HealthResponse(BaseModel):
    status: str


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    document_ids: list[str] | None = Field(default=None, max_length=100)
    top_k: int = Field(default=5, ge=1, le=50)
    candidate_k: int | None = Field(default=None, ge=1, le=100)
    dense_weight: float | None = Field(default=None, ge=0, le=1)
    rerank: bool = True


class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    content: str
    dense_score: float | None
    bm25_score: float | None
    hybrid_score: float
    retrieval_rank: int
    rerank_score: float | None
    final_rank: int


class RetrievalTiming(BaseModel):
    retrieval_ms: float
    reranking_ms: float
    total_ms: float


class RetrievalResponse(BaseModel):
    query: str
    candidate_count: int
    timing: RetrievalTiming
    results: list[RetrievalResult]
