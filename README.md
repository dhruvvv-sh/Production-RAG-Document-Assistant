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
3. **Cross-encoder reranking (implemented)** — retrieve 20 candidates, rerank
   query-passage pairs, return the best five, and measure MRR, evidence precision,
   and latency.
4. **Grounded answer generation (implemented)** — configurable model provider,
   structured claim generation, page-level citations, support validation, and
   safe insufficient-evidence responses.
5. **Evaluation and production hardening** — evaluation datasets, faithfulness
   and citation metrics, background jobs, observability, caching, and deployment.

Stages 1 through 4 are included at this point.

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

## Stage 3 architecture

Hybrid retrieval supplies 20 candidates to
`cross-encoder/ms-marco-MiniLM-L-6-v2`. The cross-encoder scores each
query-passage pair and the API returns the five highest-scoring evidence chunks.
Long passages are evaluated in overlapping windows and use their best window
score. The response includes original retrieval rank, reranker score, final rank,
candidate count, and retrieval/reranking/total latency.

## Stage 4 architecture

The query endpoint retrieves and reranks evidence before sending an allow-listed
set of evidence IDs and text to the configured language model. The model returns
structured atomic claims rather than final citation markup. The server then:

1. Rejects evidence IDs that were not retrieved.
2. Scores every claim-citation pair with the configurable NLI entailment model.
3. Removes citations and claims below the configured support threshold.
4. Constructs the final answer and numbered page-level citations itself.
5. Returns a safe insufficient-evidence response when no supported claims remain.

The default provider uses the OpenAI Responses API with structured output and
`store=false`. Set `OPENAI_API_KEY` in `.env`; `LLM_PROVIDER`, `LLM_MODEL`, and
`FAITHFULNESS_MODEL`, and `CLAIM_SUPPORT_THRESHOLD` are configurable. Provider
failures return `503` without exposing a fabricated answer.

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
  -d '{"query":"What is the annual leave allowance?","candidate_k":20,"top_k":5}'
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the annual leave allowance?"}'
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

The bundled 24-chunk smoke dataset currently scores `1.00` dense Recall@5,
`1.00` hybrid Recall@5, `0.875` hybrid MRR, and `1.00` reranked MRR. Precision@5
is `0.20` before and after reranking because each smoke query has exactly one
labeled relevant chunk, making `0.20` its maximum. The target production metrics
must be measured on a larger dataset with complete relevance judgments; the
evaluator calculates the scores and latency without hard-coding outcomes.

Grounded responses include citation correctness, faithfulness, confidence,
unsupported claims, all retrieved evidence, and stage-by-stage latency. The
evaluation module aggregates citation correctness and answer faithfulness across
a result set. The 92% citation-correctness and 87% faithfulness targets remain
targets until measured with representative labeled questions and an enabled LLM
provider.
