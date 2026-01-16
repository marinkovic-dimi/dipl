from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    embedding_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ff_dim: int = 512
    dropout_rate: float = 0.2
    embedding_dropout: float = None  # If None, uses dropout_rate
    attention_dropout: float = None   # If None, uses dropout_rate
    ffn_dropout: float = None         # If None, uses dropout_rate
    dense_dropout: float = None       # If None, uses dropout_rate
    pooling_strategy: str = "cls"
    use_intermediate_dense: bool = True  # If False, skip intermediate layer
    intermediate_dim: int = None  # If None, uses embedding_dim // 2
    label_smoothing: float = 0.0

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    epochs: int = 10

    top_k: int = 3

    # Legacy parameters from old configs (not used but needed for backward compatibility)
    attention_dropout: Optional[float] = None
    dense_dropout: Optional[float] = None
    embedding_dropout: Optional[float] = None
    ffn_dropout: Optional[float] = None
    intermediate_dim: Optional[int] = None
    use_intermediate_dense: bool = False
