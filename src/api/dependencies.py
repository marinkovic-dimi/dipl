"""FastAPI dependency injection for API services."""

import os
from pathlib import Path
from functools import lru_cache

from src.utils.config import ConfigManager
from src.data.preprocessors import SerbianTextPreprocessor
from .services import ModelLoader, PredictionService

# Get project root (3 levels up from this file: src/api/dependencies.py -> src/api -> src -> project_root)
_project_root = Path(__file__).resolve().parent.parent.parent


@lru_cache()
def get_config():
    """
    Loads and caches API configuration.

    Returns cached config on subsequent calls (singleton pattern).
    Config path can be overridden via CONFIG_PATH environment variable.

    Returns:
        Config object with inference settings
    """
    config_path = os.getenv('CONFIG_PATH', 'configs/api.yaml')

    # Make path absolute if relative
    if not Path(config_path).is_absolute():
        config_path = _project_root / config_path

    config = ConfigManager.from_yaml(str(config_path), project_root=_project_root)
    config.resolve_paths(_project_root)

    return config


@lru_cache()
def get_model_checkpoint_dir() -> str:
    """
    Gets model checkpoint directory from config or environment variable.

    Environment variable MODEL_CHECKPOINT_DIR takes precedence over config.

    Returns:
        Absolute path to model checkpoint directory
    """
    env_checkpoint = os.getenv('MODEL_CHECKPOINT_DIR')

    if env_checkpoint:
        # Environment variable provided - use it
        checkpoint_path = Path(env_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = _project_root / checkpoint_path
        return str(checkpoint_path)

    # Use config value
    config = get_config()
    return config.inference.model_checkpoint_dir


@lru_cache()
def get_prediction_service() -> PredictionService:
    """
    Creates and caches prediction service singleton.

    Loads all model artifacts on first call:
    - Model weights (classifier.keras)
    - Tokenizer (tokenizer.json)
    - Class mapping (class_map.json)
    - Preprocessor with Serbian language settings

    Subsequent calls return the cached instance.

    Returns:
        PredictionService ready for inference

    Raises:
        FileNotFoundError: If checkpoint directory or required files don't exist
        RuntimeError: If model loading fails
    """
    config = get_config()
    checkpoint_dir = Path(get_model_checkpoint_dir())

    # Load model artifacts
    loader = ModelLoader(checkpoint_dir)

    classifier = loader.load_model()
    tokenizer = loader.load_tokenizer()
    class_map = loader.load_class_map()

    # Create preprocessor with Serbian language settings
    preprocessor = SerbianTextPreprocessor(
        transliterate_cyrillic=True,
        lowercase=True,
        remove_stop_words=True,
        custom_transformations=[],
        verbose=False
    )

    # Create and return prediction service
    service = PredictionService(
        classifier=classifier,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        class_map=class_map,
        top_k=config.inference.top_k
    )

    return service


def get_api_config():
    """
    Gets API configuration settings.

    Returns:
        InferenceConfig with API settings (host, port, CORS, etc.)
    """
    config = get_config()
    return config.inference
