from typing import Optional, Dict, Any
from .ad_classifier import AdClassifier


def create_classifier_model(
    vocab_size: int,
    num_classes: int,
    config: Optional[Dict[str, Any]] = None
) -> AdClassifier:
    if config is None:
        config = {}

    default_config = {
        'max_length': 100,
        'embed_dim': 1024,
        'num_heads': 8,
        'num_layers': 2,
        'ff_dim': 2048,
        'dropout_rate': 0.1,
        'pooling_strategy': 'cls',
        'activation': 'relu',
        'use_class_weights': False,
        'label_smoothing': 0.0
    }

    final_config = {**default_config, **config}

    classifier = AdClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        **final_config
    )

    return classifier
