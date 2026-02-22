-- CortexOS Postgres schema
-- Run with pgvector extension enabled: CREATE EXTENSION IF NOT EXISTS vector;

-- Main memories table
-- embedding dimension: 1536 (OpenAI) or match your embedding model (e.g. 384 for sentence-transformers)
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('episodic', 'semantic', 'procedural')),
    summary TEXT NOT NULL,
    raw_text TEXT,
    embedding VECTOR(384),   -- match utils/embeddings (e.g. 384 for sentence-transformers; 1536 for OpenAI)
    importance FLOAT DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),
    emotion TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used TIMESTAMP WITH TIME ZONE,
    usage_count INT DEFAULT 0,
    mvn_score FLOAT,
    entities TEXT[] DEFAULT '{}',
    source TEXT DEFAULT 'chat' CHECK (source IN ('chat', 'doc', 'tool'))
);

-- Indexes per doc
CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);

-- Vector index for similarity search (IVFFlat per doc)
-- lists = number of clusters; adjust for dataset size (e.g. rows/1000)
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Optional: entity index for filtering (GIN on array)
CREATE INDEX IF NOT EXISTS idx_memories_entities ON memories USING GIN(entities);

-- Feedback log for MVN training (query, retrieved, used, reward)
CREATE TABLE IF NOT EXISTS feedback_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    query TEXT,
    retrieved_memory_ids JSONB DEFAULT '[]',
    used_memory_ids JSONB DEFAULT '[]',
    reward FLOAT DEFAULT 0.5 CHECK (reward >= 0 AND reward <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_feedback_logs_created_at ON feedback_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_logs_user_id ON feedback_logs(user_id);

-- Graph metrics cache (PageRank, degree) for reranker; populated by background job
CREATE TABLE IF NOT EXISTS graph_metrics (
    memory_id UUID PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    pagerank FLOAT DEFAULT 0,
    degree INT DEFAULT 0,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_graph_metrics_updated_at ON graph_metrics(updated_at);
