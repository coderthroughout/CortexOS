"""Run CortexOS API server."""
import uvicorn
from cortex.utils.config import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run(
        "cortex.api.server:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
