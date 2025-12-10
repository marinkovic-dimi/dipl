from dataclasses import dataclass


@dataclass
class ModelConfig:
    embedding_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ff_dim: int = 512
    dropout_rate: float = 0.2
    pooling_strategy: str = "cls"
    label_smoothing: float = 0.0

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    epochs: int = 10

    top_k: int = 3
