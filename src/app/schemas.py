from datetime import datetime

from pydantic import BaseModel, ConfigDict

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
