# CortexOS

Memory system for AI agents: extraction, graph, hybrid retrieval, MVN ranking, consolidation, and feedback.

## Setup

### 1. Python

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Postgres + pgvector

- Install [Postgres](https://www.postgresql.org/) and [pgvector](https://github.com/pgvector/pgvector).
- Create DB: `createdb cortexos`
- Run schema: `psql cortexos -f cortex/memory/db_schema.sql` (run `CREATE EXTENSION vector;` first if needed).

### 3. Environment

Copy `.env.example` to `.env` and set your values. Required: `CORTEX_DATABASE_URL`. Optional: Redis, Neo4j, `OPENAI_API_KEY` for ingestion and consolidation.

### 4. Run API

```bash
python run.py
# or: uvicorn cortex.api.server:app --reload --host 0.0.0.0 --port 8000
```

## APIs

- `POST /memory/add` – add memory (body: summary, entities, importance, type, etc.; ?user=)
- `GET /memory/query?q=...&user=...&k=...` – hybrid search + MVN + reranker
- `GET /memory/search` – alias
- `POST /memory/feedback` – body: used_memory_ids, reward (0–1)
- `POST /memory/ingest?user=...` – body: messages or content (extract + store + graph)
- `GET /memory/timeline?user=...` – timeline by period
- `GET /memory/graph?node=...&depth=2` – memory ids near entity
- `PATCH /memory/{id}` – update summary/importance
- `DELETE /memory/{id}` – delete memory
- `POST /consolidate/run?user=...` – run consolidation (sleep) for user
- `GET /health` – health check

## Project layout

```
cortex/
  api/          – FastAPI server, routes (memory, consolidate)
  ingestion/   – parser, extractor, prompts, entity_parser, normalize, pipeline
  memory/      – schema, store, vector_index, timeline, db_schema.sql
  graph/       – graph_store, graph_builder, schema
  retrieval/   – hybrid_search, bm25_index, candidate_builder, intent, retrieval_pipeline, basic_retrieval
  ranking/     – mvn_model, mvn_inference, mvn_features, reranker
  consolidation/ – clustering, summarizer, decay, sleep_worker
  training/    – mvn_dataset, mvn_train, synthetic_data, benchmark
  utils/       – embeddings, config, logger, observability
```
