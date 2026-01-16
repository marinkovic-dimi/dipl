"""
Model Information Dataclass

Sadrži sve model parametre učitane iz checkpoint foldera.
Kombinuje podatke iz config.yaml i metadata.json.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ModelInfo:
    """Model informacije učitane iz checkpoint-a."""

    # Tokenization parametri (iz config.yaml)
    vocab_size: int
    max_length: int

    # Model architecture parametri (iz config.yaml)
    embedding_dim: int
    num_heads: int
    num_layers: int
    ff_dim: int
    dropout_rate: float
    pooling_strategy: str
    label_smoothing: float

    # Training metadata (iz metadata.json)
    num_classes: int
    experiment_name: str
    timestamp: str

    # Putanje
    checkpoint_dir: Path

    # Optional metrics (iz metadata.json)
    test_accuracy: Optional[float] = None
    test_top3_accuracy: Optional[float] = None
