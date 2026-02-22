# CortexOS Demo

A standalone demo that showcases **CortexOS** as a memory layer via URL and API. **Main showcase:** a **memory-powered chat** where every message is ingested, retrieved (hybrid search), and used to build replies—with visible “Retrieved by CortexOS” blocks and feedback for MVN. **API Playground:** raw add, ingest, query, timeline, graph, feedback, consolidate.

## What is CortexOS?

CortexOS is a **memory layer for AI agents**: it stores atomic memories (episodic/semantic/procedural), links them in a graph (Neo4j), and retrieves them using vector search + BM25 + graph expansion, with optional MVN ranking and reranking. This demo talks to the CortexOS API only; you run CortexOS separately (locally or deployed).

## Setup

1. **CortexOS must be running** and reachable at a URL (e.g. `http://localhost:8000` or your deployed API).
2. **Open the demo**:
   - **From this folder:** `python -m http.server 3000` then open **http://localhost:3000/chat.html** (or **http://localhost:3000/** for index).
   - **Or** open `chat.html` / `index.html` directly in the browser (file://).
   - **API Playground:** **http://localhost:3000/playground.html** (or open `playground.html`).
   - If you run the server from the **repo root** instead, use **http://localhost:3000/Demo_Project/chat.html**.

3. **Configure in the UI:**
   - **API URL:** Base URL of CortexOS (e.g. `http://localhost:8000` or `https://your-ec2-or-api.com`). Stored in `localStorage`.
   - **User ID:** UUID used for add/ingest/query/timeline/consolidate. Default: `550e8400-e29b-41d4-a716-446655440000`.

Click **Save** to persist API URL and User ID.

## Chat flow (what it proves)

1. You send a message.
2. The app **ingests** it into CortexOS (LLM extracts memories; requires `OPENAI_API_KEY` on CortexOS server).
3. It **queries** CortexOS with your message (hybrid: vector + graph + optional BM25, then reranker).
4. The reply is built from **retrieved memories** and shown with a “Retrieved by CortexOS” block (scores visible).
5. **Feedback:** thumbs up/down sends `POST /memory/feedback` for the memories used (MVN training).
6. **Timeline** and **Graph** in the sidebar show stored memories by period and by entity.

## API Playground features

| Feature | What it does |
|--------|----------------|
| **Status** | `GET /health` and `GET /status` (embedder, Postgres, Neo4j, MVN). |
| **Add memory** | `POST /memory/add` – single memory with summary, entities, importance, type. |
| **Ingest** | `POST /memory/ingest` – raw content; CortexOS extracts memories via LLM and stores them. |
| **Query / Search** | `GET /memory/query` – hybrid retrieval (vector + optional BM25 + graph), optional MVN + reranker; returns memories with `score` and `score_breakdown`. |
| **Timeline** | `GET /memory/timeline` – memories grouped by period. |
| **Graph** | `GET /memory/graph` – traverse by entity (node) and depth; returns memory IDs. |
| **Update / Delete** | `PATCH /memory/{id}` and `DELETE /memory/{id}` – correct or remove a memory. |
| **Feedback** | `POST /memory/feedback` – record used/retrieved memory IDs and reward (for MVN training). |
| **Consolidate** | `POST /consolidate/run` – run sleep worker (cluster episodic → semantic, decay). |

## Quick test (CortexOS local or deployed)

1. Start CortexOS (e.g. `python run.py` from repo root) or use deployed API (e.g. `http://3.87.235.87:8000`).
2. Open **chat.html** (or serve the folder and open `http://localhost:3000/chat.html`).
3. Set **API URL** (e.g. `http://localhost:8000` or your deployed URL) and **User ID** (any UUID); click **Save**.
4. Send a message (e.g. “I’m building a memory layer for AI. We use Postgres and Neo4j.”). You should see “Retrieved by CortexOS” and a reply based on memories (or a note if ingest didn’t run).
5. **Playground:** open `playground.html` → Check health, Add memory, Query with same user UUID to see full API.

## Deployment

When CortexOS is deployed (e.g. EC2 or any host), set the demo’s **API URL** to that base URL (e.g. `https://api.yourdomain.com`). No code changes are required; the demo is a static client.

### Deploy on Vercel

1. Push this repo (or the `Demo_Project` folder) to GitHub.
2. In [Vercel](https://vercel.com), **Add New Project** and import the repo.
3. Set **Root Directory** to `Demo_Project` if the repo root is the full CortexOS project; otherwise leave as `.`.
4. Deploy. The chat is at `https://your-project.vercel.app/chat.html` (or `/` via redirect).
5. In the chat UI, set **API URL** to your CortexOS API. CORS is already allowed on CortexOS.

## Files

- `index.html` – Redirects to `chat.html`.
- `chat.html` + `chat.js` + `chat.css` – Memory-powered chat (ingest → query → reply + feedback, timeline, graph).
- `playground.html` – Form-based API tester (add, ingest, query, timeline, graph, PATCH/DELETE, feedback, consolidate).
- `app.js` – Playground API client.
- `styles.css` – Shared styling (dark theme, accent).
- `config.js` – Default `apiBaseUrl` and `defaultUserId` (overridable in UI and localStorage).
- `README.md` – This file.
