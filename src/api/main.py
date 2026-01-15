"""Main FastAPI application for Serbian Advertisement Classifier."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.utils.logging import setup_logging, get_logger
from .routers import prediction_router
from .dependencies import get_api_config, get_prediction_service

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Serbian Ad Classifier API",
    description="""
API for classifying Serbian advertisements into categories using a transformer-based model.

## Features
- **Serbian Language Support**: Handles both Cyrillic and Latin scripts
- **Top-5 Predictions**: Returns top 5 category predictions with confidence scores
- **Automatic Preprocessing**: Transliteration, stop words removal, text normalization
- **High Performance**: Transformer-based architecture with attention mechanisms

## Usage
1. Check `/api/v1/health` to verify the API is ready
2. Send POST requests to `/api/v1/predict` with Serbian text
3. Get top-5 category predictions with confidence scores

## Example
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={"text": "Нови Самсунг телефон А50, одлично стање"}
)
print(response.json())
```
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# Configure CORS
try:
    config = get_api_config()
    if config.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, specify actual origins
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("CORS enabled for all origins")
except Exception as e:
    logger.warning(f"Failed to load CORS config: {e}")


# Include routers
app.include_router(prediction_router.router)


@app.on_event("startup")
async def startup_event():
    """Startup event handler - preloads model for faster first request."""
    logger.info("=" * 60)
    logger.info("Starting Serbian Ad Classifier API")
    logger.info("=" * 60)

    try:
        config = get_api_config()
        logger.info(f"API Configuration:")
        logger.info(f"  - Host: {config.host}")
        logger.info(f"  - Port: {config.port}")
        logger.info(f"  - Top-K: {config.top_k}")
        logger.info(f"  - Model: {config.model_checkpoint_dir}")

        # Trigger model loading (via dependency injection)
        # This preloads the model to avoid delay on first request
        logger.info("\nPreloading model...")
        service = get_prediction_service()

        stats = service.get_stats()
        logger.info(f"\nModel Statistics:")
        logger.info(f"  - Classes: {stats['num_classes']}")
        logger.info(f"  - Vocab Size: {stats['vocab_size']}")
        logger.info(f"  - Max Length: {stats['max_sequence_length']}")

        logger.info("\n" + "=" * 60)
        logger.info("✓ API Ready to serve predictions!")
        logger.info(f"✓ Documentation: http://{config.host}:{config.port}/docs")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}", exc_info=True)
        logger.error("API starting anyway, but predictions will fail!")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("Shutting down Serbian Ad Classifier API")


@app.get(
    "/",
    summary="Root endpoint",
    description="Returns API information and links to documentation."
)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Serbian Ad Classifier API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/v1/health",
        "predict": "/api/v1/predict"
    }


@app.get(
    "/health",
    summary="Basic health check",
    description="Simple health check without loading model (faster than /api/v1/health)."
)
async def health():
    """Basic health check endpoint."""
    return {"status": "ok"}


# Custom exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs for details."}
    )
