"""Prediction service for orchestrating the prediction pipeline."""

from typing import Dict
import numpy as np
from pathlib import Path

from src.models import AdClassifier
from src.tokenization import WordPieceTokenizer
from src.data.preprocessors import SerbianTextPreprocessor
from src.utils.logging import LoggerMixin
from src.utils.category_names import CategoryNameLoader
from ..schemas.prediction_schemas import PredictionResponse, CategoryPrediction


class PredictionService(LoggerMixin):
    """
    Orchestrates the complete prediction pipeline.

    Pipeline flow:
    1. Preprocess raw text (transliteration, cleaning, stop words removal)
    2. Tokenize preprocessed text to token IDs
    3. Run model prediction to get top-k class probabilities
    4. Map class IDs to class names
    5. Format results into response object

    Args:
        classifier: Trained AdClassifier model
        tokenizer: Trained WordPieceTokenizer
        preprocessor: SerbianTextPreprocessor for text cleaning
        class_map: Mapping from class IDs (int) to class names (str)
        top_k: Number of top predictions to return (default: 5)
    """

    def __init__(
        self,
        classifier: AdClassifier,
        tokenizer: WordPieceTokenizer,
        preprocessor: SerbianTextPreprocessor,
        class_map: Dict[int, str],
        top_k: int = 5,
        category_names_file: str = "data/category_name.json"
    ):
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.class_map = class_map
        self.top_k = top_k

        # Load category names
        self.category_loader = CategoryNameLoader(category_names_file)

        self.logger.info(f"PredictionService initialized with top_k={top_k}")
        self.logger.info(f"Model ready with {len(class_map)} classes")

    def predict_single(self, raw_text: str) -> PredictionResponse:
        """
        Predicts categories for a single text input.

        Args:
            raw_text: Raw advertisement text in Serbian (Cyrillic or Latin)

        Returns:
            PredictionResponse with top-k predictions and confidence scores

        Raises:
            ValueError: If input text is empty after preprocessing
            RuntimeError: If prediction fails
        """
        try:
            # Step 1: Preprocess text
            self.logger.debug(f"Preprocessing text: {raw_text[:100]}...")
            clean_text = self.preprocessor.preprocess_text(raw_text)

            if not clean_text or clean_text.strip() == "":
                raise ValueError(
                    "Input text is empty after preprocessing. "
                    "Please provide meaningful text content."
                )

            self.logger.debug(f"Preprocessed text: {clean_text}")

            # Step 2: Tokenize
            self.logger.debug("Tokenizing text...")
            token_ids = self.tokenizer.encode_batch(
                [clean_text],
                pad_to_max_length=True
            )
            self.logger.debug(f"Token IDs shape: {token_ids.shape}")

            # Step 3: Predict
            self.logger.debug("Running model prediction...")
            top_k_indices, top_k_probs = self.classifier.predict_top_k(
                token_ids,
                k=self.top_k
            )

            # Step 4: Map class IDs to class names and format results
            predictions = []
            for class_id, prob in zip(top_k_indices[0], top_k_probs[0]):
                class_id_int = int(class_id)
                # Get group_id from class_map (index -> group_id)
                group_id = self.class_map.get(
                    class_id_int,
                    f"Unknown_{class_id_int}"
                )

                # Get combined name (group | category) from category loader
                category_name = self.category_loader.get_name(group_id)

                predictions.append(
                    CategoryPrediction(
                        category_id=group_id,
                        category_name=category_name,
                        confidence=float(prob)
                    )
                )

            self.logger.info(
                f"Prediction complete. Top prediction: {predictions[0].category_name} "
                f"(ID: {predictions[0].category_id}, {predictions[0].confidence:.2%})"
            )

            # Step 5: Create response
            response = PredictionResponse(
                text=raw_text,
                predictions=predictions,
                preprocessed_text=clean_text
            )

            return response

        except ValueError as e:
            self.logger.warning(f"Validation error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}", exc_info=True)
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def predict_batch(self, texts: list[str]) -> list[PredictionResponse]:
        """
        Predicts categories for multiple texts (batch prediction).

        Args:
            texts: List of raw advertisement texts

        Returns:
            List of PredictionResponse objects, one per input text

        Note:
            Currently processes texts one by one. Can be optimized for
            true batch processing in the future.
        """
        self.logger.info(f"Batch prediction for {len(texts)} texts")

        results = []
        for i, text in enumerate(texts):
            try:
                result = self.predict_single(text)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to predict text {i}: {e}")
                # Could add partial results handling here if needed
                raise

        self.logger.info(f"Batch prediction complete: {len(results)} results")
        return results

    def get_stats(self) -> Dict:
        """
        Returns prediction service statistics.

        Returns:
            Dictionary with service configuration and statistics
        """
        return {
            'top_k': self.top_k,
            'num_classes': len(self.class_map),
            'vocab_size': self.tokenizer.get_vocab_size(),
            'max_sequence_length': self.tokenizer.max_length,
            'classes': list(self.class_map.values())
        }
