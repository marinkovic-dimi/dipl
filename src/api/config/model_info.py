from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ModelInfo:
    vocab_size: int
    max_length: int
    embedding_dim: int
    num_heads: int
    num_layers: int
    ff_dim: int
    dropout_rate: float
    pooling_strategy: str
    label_smoothing: float
    num_classes: int
    experiment_name: str
    timestamp: str
    checkpoint_dir: Path
    test_accuracy: Optional[float] = None
    test_top3_accuracy: Optional[float] = None
