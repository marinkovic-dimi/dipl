"""
API Config Loader

Učitava samo API-specifične postavke iz YAML fajla.
Svi model parametri se učitavaju iz model checkpoint foldera.
"""

import yaml
from pathlib import Path
from typing import Optional
from src.utils.config import InferenceConfig


class ApiConfigLoader:
    """Učitava API konfiguraciju iz YAML fajla."""

    @staticmethod
    def from_yaml(config_path: str, project_root: Optional[Path] = None) -> InferenceConfig:
        """
        Učitava InferenceConfig iz YAML fajla.

        Podržava dva formata:
        1. Novi format: samo 'inference' sekcija
        2. Stari format: kompletan config sa svim sekcijama

        Args:
            config_path: Putanja do YAML config fajla
            project_root: Root direktorijum projekta za resolve relativnih putanja

        Returns:
            InferenceConfig objekat sa API postavkama

        Raises:
            FileNotFoundError: Ako config fajl ne postoji
            yaml.YAMLError: Ako YAML parsing ne uspe
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # Izvuci inference sekciju (ili koristi ceo dict ako nema sekciju)
        inference_dict = config_dict.get('inference', config_dict)

        # Kreiraj InferenceConfig
        api_config = InferenceConfig(**inference_dict)

        # Resolve putanje ako je project_root prosleđen
        if project_root:
            if not Path(api_config.model_checkpoint_dir).is_absolute():
                api_config.model_checkpoint_dir = str(
                    project_root / api_config.model_checkpoint_dir
                )

        return api_config
