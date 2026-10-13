# Production RAG Document Assistant

A production-oriented PDF question-answering service built incrementally with
FastAPI, PostgreSQL, BM25, sentence-transformer embeddings, and cross-encoder
reranking.

## Delivery roadmap

1. **Foundation and PDF ingestion (implemented)** — service configuration,
   PostgreSQL persistence, PDF validation/extraction, page-aware 500–800 token
   chunking, upload/status APIs, migrations, and tests.
2. **Hybrid retrieval** — BM25 and sentence-transformer indexes, score
   normalization/fusion, and a retrieval API.
3. **Cross-encoder reranking** — rerank candidate passages and measure Recall@k,
   MRR, and evidence precision.
4. **Grounded answer generation** — prompt construction, model adapter,
   page-level citations, and evidence validation.
5. **Evaluation and production hardening** — evaluation datasets, faithfulness
   and citation metrics, background jobs, observability, caching, and deployment.

Only Stage 1 is included at this point.

## Stage 1 architecture

The upload endpoint validates a PDF, extracts text page by page, chunks it with
the `cl100k_base` tokenizer, and stores the document and chunks transactionally.
Chunk boundaries prefer paragraph and sentence endings while enforcing a hard
maximum of 800 tokens. Page numbers are retained for later citations.

## Quick start

Requirements: Docker (recommended), or Python 3.11–3.13 and an existing
PostgreSQL instance.

```bash
cp .env.example .env
docker compose up --build
```

The API is then available at `http://localhost:8000`; interactive docs are at
`http://localhost:8000/docs`.

```bash
curl http://localhost:8000/health/live
curl -F "file=@document.pdf" http://localhost:8000/api/v1/documents
curl http://localhost:8000/api/v1/documents/<document-id>
```

The upload limit defaults to 25 MiB. Duplicate content returns the existing
document rather than storing it twice.

## Local development

```bash
uv sync --extra dev
cp .env.example .env
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
uv run pytest
```

Tests use in-memory SQLite and do not require a running PostgreSQL server.
