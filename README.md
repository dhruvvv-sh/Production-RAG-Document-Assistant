# Production RAG Document Assistant

A production-oriented PDF question-answering service built incrementally with
FastAPI, PostgreSQL, BM25, sentence-transformer embeddings, and cross-encoder
reranking.

## Delivery roadmap

1. **Foundation and PDF ingestion (implemented)** — service configuration,
   PostgreSQL persistence, PDF validation/extraction, page-aware 500–800 token
   chunking, upload/status APIs, migrations, and tests.
2. **Hybrid retrieval (implemented)** — BM25 and sentence-transformer indexes,
   pgvector cosine search, score normalization/fusion, retrieval API, and
   Recall@k evaluation.
3. **Cross-encoder reranking** — rerank candidate passages and measure Recall@k,
   MRR, and evidence precision.
4. **Grounded answer generation** — prompt construction, model adapter,
   page-level citations, and evidence validation.
5. **Evaluation and production hardening** — evaluation datasets, faithfulness
   and citation metrics, background jobs, observability, caching, and deployment.

Stages 1 and 2 are included at this point.

## Stage 1 architecture

The upload endpoint validates a PDF, extracts text page by page, chunks it with
the `cl100k_base` tokenizer, and stores the document and chunks transactionally.
Chunk boundaries prefer paragraph and sentence endings while enforcing a hard
maximum of 800 tokens. Page numbers are retained for later citations.

## Stage 2 architecture

Ingestion embeds every chunk with `sentence-transformers/all-MiniLM-L6-v2` and
stores its normalized 384-dimensional vector in PostgreSQL. Dense retrieval uses
pgvector cosine distance and an HNSW index. Keyword retrieval builds a BM25 index
over the requested document scope. The service independently min-max normalizes
both result sets and combines them with a configurable dense weight.

Chunks longer than MiniLM's input limit are embedded in overlapping model-token
windows and mean-pooled, preventing silent truncation of 500–800 token chunks.

Every result includes its document, filename, PDF page, chunk text, dense score,
BM25 score, and combined score. These identifiers and page numbers form the
evidence references used by the citation stage.

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
curl -X POST http://localhost:8000/api/v1/retrieval/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the annual leave allowance?","top_k":5}'
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

## Retrieval evaluation

Run the bundled labeled corpus against the configured embedding model:

```bash
uv run rag-evaluate
uv run rag-evaluate --dataset path/to/labeled-dataset.json --k 5
```

The bundled 24-chunk smoke dataset currently scores `1.00` dense Recall@5 and
`1.00` hybrid Recall@5. It verifies the evaluation pipeline but is too small and
easy to support a production-quality improvement claim. The target improvement
from 78.4% to 89.7% must be measured on a larger, representative labeled dataset;
the evaluator reports both scores and the percentage-point difference without
hard-coding them.
