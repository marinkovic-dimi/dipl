"""Entrypoint script for running the FastAPI server."""

import sys
from pathlib import Path

# Add project root to Python path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import uvicorn
from src.api.dependencies import get_api_config


def main():
    """Run the FastAPI application with uvicorn."""
    try:
        config = get_api_config()

        print(f"Starting API server on {config.host}:{config.port}")
        print(f"Documentation will be available at http://{config.host}:{config.port}/docs")

        uvicorn.run(
            "src.api.main:app",
            host=config.host,
            port=config.port,
            reload=False,  # Set to True for development (auto-reload on file changes)
            log_level="info",
            access_log=True
        )

    except Exception as e:
        print(f"Failed to start API server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
