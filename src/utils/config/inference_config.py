from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """Configuration for model inference and API serving."""

    model_checkpoint_dir: str = "experiments/model_wandb_latest"
    host: str = "0.0.0.0"
    port: int = 8000
    top_k: int = 5
    batch_size: int = 32
    enable_cors: bool = True
