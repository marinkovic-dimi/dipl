"""Script for loading trained models and making predictions."""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List

from src.models import AdClassifier
# Import to register the custom metric with Keras
from src.models.classifier import top_k_acc  # noqa: F401
from src.tokenization import WordPieceTokenizer
from src.data import SerbianTextPreprocessor
from src.utils import setup_logging, get_logger
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def load_trained_model(model_dir: str) -> Tuple[AdClassifier, WordPieceTokenizer, Dict, Dict]:
    """
    Load trained model, tokenizer, class mapping, and metadata from experiment directory.

    Args:
        model_dir: Path to the experiment directory containing model artifacts

    Returns:
        Tuple of (classifier, tokenizer, class_map, metadata)
    """
    logger = get_logger(__name__)
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Load metadata
    metadata_path = model_dir / 'metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    logger.info(f"Loaded metadata from {metadata_path}")

    # Load class mapping
    class_map_path = model_dir / 'class_map.json'
    if not class_map_path.exists():
        raise FileNotFoundError(f"Class map file not found: {class_map_path}")

    with open(class_map_path, 'r') as f:
        class_map = json.load(f)

    # Convert keys back to int
    class_map = {int(k): v for k, v in class_map.items()}
    reverse_class_map = {v: k for k, v in class_map.items()}

    logger.info(f"Loaded class mapping with {len(class_map)} classes")

    # Create classifier with saved parameters
    classifier = AdClassifier(
        vocab_size=metadata['vocab_size'],
        num_classes=metadata['num_classes'],
        max_length=metadata['max_length'],
        embed_dim=metadata.get('embed_dim', 256),
        num_heads=metadata.get('num_heads', 8),
        num_layers=metadata.get('num_layers', 4),
        ff_dim=metadata.get('ff_dim', 512),
        dropout_rate=metadata.get('dropout_rate', 0.2),
        pooling_strategy=metadata.get('pooling_strategy', 'cls')
    )

    # Load model weights
    model_path = model_dir / 'classifier.keras'
    if not model_path.exists():
        # Try checkpoint
        model_path = model_dir / 'checkpoint.keras'

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found in {model_dir}")

    classifier.load_model(str(model_path))
    logger.info(f"Loaded model from {model_path}")

    # Load tokenizer
    tokenizer_path = model_dir / 'tokenizer.json'
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    tokenizer = WordPieceTokenizer(
        vocab_size=metadata['vocab_size'],
        max_length=metadata['max_length']
    )
    tokenizer.load(str(tokenizer_path))
    logger.info(f"Loaded tokenizer from {tokenizer_path}")

    return classifier, tokenizer, reverse_class_map, metadata


def predict_text(
    text: str,
    classifier: AdClassifier,
    tokenizer: WordPieceTokenizer,
    preprocessor: SerbianTextPreprocessor,
    class_map: Dict[int, int],
    top_k: int = 3
) -> List[Tuple[int, float]]:
    """
    Make prediction on a single text.

    Args:
        text: Input text to classify
        classifier: Trained classifier
        tokenizer: Trained tokenizer
        preprocessor: Text preprocessor
        class_map: Mapping from model indices to original class IDs
        top_k: Number of top predictions to return

    Returns:
        List of (class_id, probability) tuples
    """
    # Preprocess
    clean_text = preprocessor.preprocess_text(text)

    # Tokenize
    encoded = tokenizer.encode_batch([clean_text])

    # Predict
    top_k_indices, top_k_probs = classifier.predict_top_k(encoded, k=top_k)

    # Map back to original class IDs
    results = []
    for idx, prob in zip(top_k_indices[0].numpy(), top_k_probs[0].numpy()):
        original_class_id = class_map.get(idx, idx)
        results.append((original_class_id, float(prob)))

    return results


def predict_batch(
    texts: List[str],
    classifier: AdClassifier,
    tokenizer: WordPieceTokenizer,
    preprocessor: SerbianTextPreprocessor,
    class_map: Dict[int, int]
) -> np.ndarray:
    """
    Make predictions on a batch of texts.

    Args:
        texts: List of input texts
        classifier: Trained classifier
        tokenizer: Trained tokenizer
        preprocessor: Text preprocessor
        class_map: Mapping from model indices to original class IDs

    Returns:
        Array of predicted class IDs
    """
    # Preprocess all texts
    clean_texts = [preprocessor.preprocess_text(t) for t in texts]

    # Tokenize
    encoded = tokenizer.encode_batch(clean_texts)

    # Predict
    predicted_indices = classifier.predict_classes(encoded)

    # Map back to original class IDs
    predicted_classes = np.array([class_map.get(int(idx), int(idx)) for idx in predicted_indices])

    return predicted_classes


def main():
    """Example usage of loaded model."""
    parser = argparse.ArgumentParser(description='Load trained model and make predictions')
    parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help='Path to the experiment directory containing model artifacts'
    )
    parser.add_argument(
        '--text',
        type=str,
        default=None,
        help='Text to classify (optional, uses examples if not provided)'
    )
    args = parser.parse_args()

    logger = setup_logging(log_level='INFO', experiment_name='load_model')

    logger.info("=" * 60)
    logger.info("LOADING TRAINED MODEL")
    logger.info("=" * 60)

    # Load model
    classifier, tokenizer, class_map, metadata = load_trained_model(args.model_dir)

    logger.info("\nModel Information:")
    logger.info(f"  - Classes: {metadata['num_classes']}")
    logger.info(f"  - Vocab size: {metadata['vocab_size']}")
    logger.info(f"  - Max length: {metadata['max_length']}")
    logger.info(f"  - Test accuracy: {metadata.get('test_accuracy', 'N/A'):.2%}")
    logger.info(f"  - Trained on: {metadata.get('train_samples', 'N/A')} samples")

    # Create preprocessor
    preprocessor = SerbianTextPreprocessor(
        transliterate_cyrillic=True,
        lowercase=True,
        remove_stop_words=False,
        verbose=False
    )

    logger.info("\n" + "=" * 60)
    logger.info("MAKING PREDICTIONS")
    logger.info("=" * 60)

    if args.text:
        # Predict user-provided text
        texts = [args.text]
    else:
        # Example texts
        texts = [
            "iPhone 15 Pro Max 256GB crni",
            "Samsung Galaxy S24 Ultra",
            "Xiaomi punjac 65W brzi",
            "Maska za telefon silikon",
            "Slusalice bluetooth bezicne"
        ]

    for text in texts:
        logger.info(f"\nInput: {text}")

        predictions = predict_text(
            text, classifier, tokenizer, preprocessor, class_map, top_k=3
        )

        logger.info("Predictions:")
        for i, (class_id, prob) in enumerate(predictions, 1):
            logger.info(f"  {i}. Class {class_id}: {prob:.2%}")

    logger.info("\n" + "=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
