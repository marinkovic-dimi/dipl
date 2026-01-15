"""Pydantic schemas for prediction API requests and responses."""

from pydantic import BaseModel, Field
from typing import List


class PredictionRequest(BaseModel):
    """Request schema for single text prediction."""

    text: str = Field(
        ...,
        description="Raw advertisement text in Serbian (Cyrillic or Latin)",
        min_length=1,
        max_length=5000,
        examples=["Нови Самсунг телефон А50, одлично стање"]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Нови Самсунг телефон А50, одлично стање, 150 евра"
            }
        }


class CategoryPrediction(BaseModel):
    """Single category prediction with confidence score."""

    category: str = Field(
        ...,
        description="Predicted category name"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence score in range [0, 1]"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "category": "Mobilni telefoni",
                "confidence": 0.87
            }
        }


class PredictionResponse(BaseModel):
    """Response schema with top-k predictions."""

    text: str = Field(
        ...,
        description="Original input text"
    )
    predictions: List[CategoryPrediction] = Field(
        ...,
        description="Top-5 predictions sorted by confidence (descending)",
        min_length=1,
        max_length=5
    )
    preprocessed_text: str = Field(
        ...,
        description="Text after preprocessing (transliteration, cleaning, stop words removal)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Нови Самсунг телефон А50, одлично стање, 150 евра",
                "predictions": [
                    {"category": "Mobilni telefoni", "confidence": 0.87},
                    {"category": "Elektronika", "confidence": 0.08},
                    {"category": "Oprema", "confidence": 0.03},
                    {"category": "Tablet", "confidence": 0.01},
                    {"category": "Ostalo", "confidence": 0.01}
                ],
                "preprocessed_text": "samsung telefon a stanje evra"
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(
        ...,
        description="API status"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether model is loaded and ready"
    )
    model_checkpoint: str = Field(
        ...,
        description="Path to loaded model checkpoint"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_checkpoint": "experiments/model_wandb_20251221_153839"
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""

    detail: str = Field(
        ...,
        description="Error message"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Prediction failed: Invalid input"
            }
        }
