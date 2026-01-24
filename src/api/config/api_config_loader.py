import yaml
from pathlib import Path
from typing import Optional
from src.utils.config import InferenceConfig


class ApiConfigLoader:
    @staticmethod
    def from_yaml(config_path: str, project_root: Optional[Path] = None) -> InferenceConfig:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        inference_dict = config_dict.get('inference', config_dict)

        api_config = InferenceConfig(**inference_dict)

        if project_root:
            if not Path(api_config.model_checkpoint_dir).is_absolute():
                api_config.model_checkpoint_dir = str(
                    project_root / api_config.model_checkpoint_dir
                )

        return api_config
