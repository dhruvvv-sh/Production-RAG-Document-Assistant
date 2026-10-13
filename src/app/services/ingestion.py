import hashlib

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk, Document, DocumentStatus
from app.services.chunking import TokenChunker
from app.services.pdf import extract_pages


def ingest_pdf(
    db: Session,
    *,
    filename: str,
    content_type: str,
    content: bytes,
    chunker: TokenChunker,
) -> tuple[Document, bool]:
    digest = hashlib.sha256(content).hexdigest()
    existing = db.scalar(select(Document).where(Document.sha256 == digest))
    if existing is not None:
        return existing, False

    pages = extract_pages(content)
    text_chunks = chunker.chunk_pages(pages)
    document = Document(
        filename=filename,
        content_type=content_type,
        sha256=digest,
        size_bytes=len(content),
        page_count=len(pages),
        status=DocumentStatus.READY,
    )
    document.chunks = [
        Chunk(
            chunk_index=index,
            page_number=chunk.page_number,
            content=chunk.content,
            token_count=chunk.token_count,
        )
        for index, chunk in enumerate(text_chunks)
    ]
    db.add(document)
    db.commit()
    db.refresh(document)
    return document, True
