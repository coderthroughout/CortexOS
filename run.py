"""Run CortexOS API server. Load .env before anything else."""
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass
import uvicorn
from cortex.utils.config import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run(
        "cortex.api.server:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
