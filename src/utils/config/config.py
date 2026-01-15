from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from .data_config import DataConfig
from .tokenization_config import TokenizationConfig
from .model_config import ModelConfig
from .balancing_config import BalancingConfig
from .training_config import TrainingConfig
from .wandb_config import WandbConfig
from .inference_config import InferenceConfig


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
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = f"{self.project_name}_{self.timestamp}"

    def resolve_paths(self, project_root: Union[str, Path]) -> "Config":
        root = Path(project_root)

        def resolve(path_str: str) -> str:
            p = Path(path_str)
            if not p.is_absolute():
                return str(root / p)
            return path_str

        self.data.raw_data_path = resolve(self.data.raw_data_path)
        self.data.processed_data_dir = resolve(self.data.processed_data_dir)
        self.data.ostalo_groups_file = resolve(self.data.ostalo_groups_file)

        self.tokenization.tokenizer_cache_dir = resolve(self.tokenization.tokenizer_cache_dir)

        self.training.save_dir = resolve(self.training.save_dir)
        self.training.log_dir = resolve(self.training.log_dir)

        self.inference.model_checkpoint_dir = resolve(self.inference.model_checkpoint_dir)

        return self
