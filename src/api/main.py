from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.utils.logging import setup_logging, get_logger
from .routers import prediction_router
from .dependencies import get_api_config, get_prediction_service, get_model_info

setup_logging(log_level="INFO")
logger = get_logger(__name__)

app = FastAPI(
    title="Serbian Ad Classifier API",
    description="""
        API for classifying Serbian advertisements into categories using a transformer-based model.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

try:
    config = get_api_config()
    if config.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("CORS enabled for all origins")
except Exception as e:
    logger.warning(f"Failed to load CORS config: {e}")


app.include_router(prediction_router.router)


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("Starting Serbian Ad Classifier API")
    logger.info("=" * 60)

    try:
        api_config = get_api_config()
        logger.info(f"API Configuration:")
        logger.info(f"  - Host: {api_config.host}")
        logger.info(f"  - Port: {api_config.port}")
        logger.info(f"  - Top-K: {api_config.top_k}")
        logger.info(f"  - Model: {api_config.model_checkpoint_dir}")

        logger.info("\nPreloading model...")
        service = get_prediction_service()
        model_info = get_model_info()

        stats = service.get_stats()
        logger.info(f"\nModel Information:")
        logger.info(f"  - Experiment: {model_info.experiment_name}")
        logger.info(f"  - Timestamp: {model_info.timestamp}")
        logger.info(f"  - Classes: {stats['num_classes']}")
        logger.info(f"  - Vocab Size: {stats['vocab_size']}")
        logger.info(f"  - Max Length: {stats['max_sequence_length']}")
        if model_info.test_accuracy:
            logger.info(f"  - Test Accuracy: {model_info.test_accuracy:.2%}")
        if model_info.test_top3_accuracy:
            logger.info(f"  - Top-3 Accuracy: {model_info.test_top3_accuracy:.2%}")

        logger.info("\n" + "=" * 60)
        logger.info("✓ API Ready to serve predictions!")
        logger.info(f"✓ Documentation: http://{api_config.host}:{api_config.port}/docs")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}", exc_info=True)
        logger.error("API starting anyway, but predictions will fail!")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Serbian Ad Classifier API")


@app.get(
    "/",
    summary="Root endpoint",
    description="Returns API information and links to documentation."
)
async def root():
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
    return {"status": "ok"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs for details."}
    )
