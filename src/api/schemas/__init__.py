"""API schemas module."""

from .prediction_schemas import (
    PredictionRequest,
    PredictionResponse,
    CategoryPrediction,
    HealthResponse,
    ErrorResponse
)

__all__ = [
    'PredictionRequest',
    'PredictionResponse',
    'CategoryPrediction',
    'HealthResponse',
    'ErrorResponse'
]
