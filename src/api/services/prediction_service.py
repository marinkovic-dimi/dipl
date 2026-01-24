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

        self.category_loader = CategoryNameLoader(category_names_file)

        self.logger.info(f"PredictionService initialized with top_k={top_k}")
        self.logger.info(f"Model ready with {len(class_map)} classes")

    def predict_single(self, raw_text: str) -> PredictionResponse:
        try:
            self.logger.debug(f"Preprocessing text: {raw_text[:100]}...")
            clean_text = self.preprocessor.preprocess_text(raw_text)

            if not clean_text or clean_text.strip() == "":
                raise ValueError(
                    "Input text is empty after preprocessing. "
                    "Please provide meaningful text content."
                )

            self.logger.debug(f"Preprocessed text: {clean_text}")

            self.logger.debug("Tokenizing text...")
            token_ids = self.tokenizer.encode_batch(
                [clean_text],
                pad_to_max_length=True
            )
            self.logger.debug(f"Token IDs shape: {token_ids.shape}")

            self.logger.debug("Running model prediction...")
            top_k_indices, top_k_probs = self.classifier.predict_top_k(
                token_ids,
                k=self.top_k
            )

            predictions = []
            for class_id, prob in zip(top_k_indices[0], top_k_probs[0]):
                class_id_int = int(class_id)
                group_id = self.class_map.get(
                    class_id_int,
                    f"Unknown_{class_id_int}"
                )

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
        self.logger.info(f"Batch prediction for {len(texts)} texts")

        results = []
        for i, text in enumerate(texts):
            try:
                result = self.predict_single(text)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to predict text {i}: {e}")
                raise

        self.logger.info(f"Batch prediction complete: {len(results)} results")
        return results

    def get_stats(self) -> Dict:
        return {
            'top_k': self.top_k,
            'num_classes': len(self.class_map),
            'vocab_size': self.tokenizer.get_vocab_size(),
            'max_sequence_length': self.tokenizer.max_length,
            'classes': list(self.class_map.values())
        }
