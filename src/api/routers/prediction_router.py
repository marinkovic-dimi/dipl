"""Prediction API router with endpoints for classification."""

from fastapi import APIRouter, Depends, HTTPException, status
from src.utils.logging import get_logger

from ..schemas import PredictionRequest, PredictionResponse, HealthResponse
from ..services import PredictionService
from ..dependencies import get_prediction_service, get_model_checkpoint_dir

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["predictions"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict advertisement category",
    description="Classifies Serbian advertisement text into categories with confidence scores.",
    responses={
        200: {
            "description": "Successful prediction with top-5 categories",
            "model": PredictionResponse
        },
        400: {
            "description": "Invalid input (empty text, invalid format)",
        },
        500: {
            "description": "Internal server error during prediction",
        }
    }
)
async def predict(
    request: PredictionRequest,
    service: PredictionService = Depends(get_prediction_service)
) -> PredictionResponse:
    """
    Predicts the category for a given advertisement text.

    Returns top-5 predictions sorted by confidence (descending).

    **Input:**
    - Accepts Serbian text in Cyrillic or Latin script
    - Text length: 1-5000 characters
    - Automatically handles preprocessing (transliteration, stop words, etc.)

    **Output:**
    - Original text
    - Top-5 category predictions with confidence scores [0, 1]
    - Preprocessed text (for debugging)

    **Example:**
    ```json
    {
      "text": "Нови Самсунг телефон А50, одлично стање, 150 евра"
    }
    ```
    """
    try:
        logger.info(f"Received prediction request: {request.text[:50]}...")

        result = service.predict_single(request.text)

        logger.info(
            f"Prediction successful. Top: {result.predictions[0].category_name} "
            f"(ID: {result.predictions[0].category_id}, {result.predictions[0].confidence:.2%})"
        )

        return result

    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check if API and model are loaded and ready to serve predictions."
)
async def health_check(
    service: PredictionService = Depends(get_prediction_service),
    checkpoint_dir: str = Depends(get_model_checkpoint_dir)
) -> HealthResponse:
    """
    Health check endpoint.

    Returns:
    - API status
    - Model loading status
    - Loaded model checkpoint path

    Use this endpoint to verify the API is ready before sending prediction requests.
    """
    try:
        # If we got here, service is loaded successfully (dependency injection)
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_checkpoint=checkpoint_dir
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_checkpoint=""
        )


@router.get(
    "/stats",
    summary="Get service statistics",
    description="Returns statistics about the loaded model and service configuration."
)
async def get_stats(
    service: PredictionService = Depends(get_prediction_service)
) -> dict:
    """
    Returns prediction service statistics.

    Provides information about:
    - Number of classes
    - Vocabulary size
    - Maximum sequence length
    - Available categories
    - Top-k configuration
    """
    try:
        stats = service.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )
