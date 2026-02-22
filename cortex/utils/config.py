"""Config from environment. Load .env from project root before reading."""
import os
from pathlib import Path
from typing import Optional

# Load .env from project root (parent of cortex/)
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)


def env(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(key, default)


# Database
DATABASE_URL: str = env("CORTEX_DATABASE_URL") or "postgresql://localhost/cortexos"
REDIS_URL: str = env("CORTEX_REDIS_URL") or "redis://localhost:6379/0"
NEO4J_URI: str = env("CORTEX_NEO4J_URI") or "bolt://localhost:7687"
NEO4J_USER: str = env("CORTEX_NEO4J_USER") or "neo4j"
NEO4J_PASSWORD: str = env("CORTEX_NEO4J_PASSWORD") or ""

# Embeddings
EMBEDDING_MODEL: str = env("CORTEX_EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"

# API
API_HOST: str = env("CORTEX_API_HOST") or "0.0.0.0"
API_PORT: int = int(env("CORTEX_API_PORT") or "8000")
