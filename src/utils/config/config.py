"""Main configuration class that aggregates all config components."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .data_config import DataConfig
from .tokenization_config import TokenizationConfig
from .model_config import ModelConfig
from .balancing_config import BalancingConfig
from .training_config import TrainingConfig
from .wandb_config import WandbConfig


@dataclass
class Config:
    project_name: str = "klasifikator"
    version: str = "1.0.0"
    experiment_name: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    data: DataConfig = field(default_factory=DataConfig)
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    balancing: BalancingConfig = field(default_factory=BalancingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = f"{self.project_name}_{self.timestamp}"
