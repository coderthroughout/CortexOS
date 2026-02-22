# CortexOS

Memory system for AI agents: extraction, graph, hybrid retrieval, MVN ranking, consolidation, and feedback.

**Demo (Vercel):** The `Demo_Project/` folder is a static frontend (chat + API playground) you can deploy to Vercel with **Root Directory** = `Demo_Project`. See `Demo_Project/README.md` and `Demo_Project/DEPLOY_VERCEL.md`.

**More docs:** `docs/` (architecture, setup, testing, deployment).

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

### MVN training from feedback

After collecting feedback via `POST /memory/feedback` (with optional `query`, `retrieved_memory_ids`, `used_memory_ids`), train the Memory Value Network:

```bash
python scripts/train_mvn.py [--limit 5000] [--save checkpoints/mvn.pt] [--epochs 10]
```

Then set `CORTEX_MVN_CHECKPOINT=checkpoints/mvn.pt` (or your path) in `.env` and restart the API to use the trained model for ranking.

### Evaluation

- **Retrieval metrics (Recall@K, MRR):**  
  `python scripts/eval_retrieval.py scripts/regression_queries.json`  
  Use `--regression --min-recall 0.5 --min-mrr 0.3` to fail the run if below thresholds (e.g. in CI).
- **Downstream judge (LLM):**  
  `python scripts/eval_downstream_judge.py scripts/regression_queries.json`  
  Requires `OPENAI_API_KEY`; scores how well answers use the retrieved memories.

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
