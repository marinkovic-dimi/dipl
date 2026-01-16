import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .config import Config
from .data_config import DataConfig
from .tokenization_config import TokenizationConfig
from .model_config import ModelConfig
from .balancing_config import BalancingConfig
from .training_config import TrainingConfig
from .wandb_config import WandbConfig
from .inference_config import InferenceConfig


class ConfigManager:

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

    @classmethod
    def from_yaml(
        cls,
        config_path: str,
        project_root: Optional[Union[str, Path]] = None
    ) -> Config:
        config_path = Path(config_path)
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        config = cls._dict_to_config(config_dict)

        if project_root is None:
            project_root = config_path.resolve().parent.parent

        config.resolve_paths(project_root)
        return config

    @classmethod
    def _dict_to_config(cls, config_dict: Dict[str, Any]) -> Config:
        data_config = DataConfig(**config_dict.get('data', {}))
        tokenization_config = TokenizationConfig(**config_dict.get('tokenization', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        balancing_config = BalancingConfig(**config_dict.get('balancing', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        wandb_config = WandbConfig(**config_dict.get('wandb', {}))
        inference_config = InferenceConfig(**config_dict.get('inference', {}))

        main_config_dict = {k: v for k, v in config_dict.items()
                           if k not in ['data', 'tokenization', 'model', 'balancing', 'training', 'wandb', 'inference']}

        return Config(
            data=data_config,
            tokenization=tokenization_config,
            model=model_config,
            balancing=balancing_config,
            training=training_config,
            wandb=wandb_config,
            inference=inference_config,
            **main_config_dict
        )

    def save_config(self, config: Config, filename: str) -> Path:
        config_path = self.config_dir / filename
        config_dict = self._config_to_dict(config)

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

        return config_path

    @staticmethod
    def _config_to_dict(config: Config) -> Dict[str, Any]:
        return {
            'project_name': config.project_name,
            'version': config.version,
            'experiment_name': config.experiment_name,
            'timestamp': config.timestamp,
            'data': config.data.__dict__,
            'tokenization': config.tokenization.__dict__,
            'model': config.model.__dict__,
            'balancing': config.balancing.__dict__,
            'training': config.training.__dict__,
            'wandb': config.wandb.__dict__,
            'inference': config.inference.__dict__
        }

    def create_default_config(self) -> Config:
        return Config()
