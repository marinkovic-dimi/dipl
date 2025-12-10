"""Training configuration."""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    save_dir: str = "experiments"
    model_name: str = "klasifikator"
    save_best_only: bool = True
    patience: int = 3
    monitor: str = "val_loss"
    mode: str = "min"

    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 2
    min_lr: float = 1e-6

    log_level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "logs"

    checkpoint_freq: str = "epoch"
    save_frequency: int = 1
