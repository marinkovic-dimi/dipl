"""FastAPI dependency injection for API services."""

import os
from pathlib import Path
from functools import lru_cache

from src.data.preprocessors import SerbianTextPreprocessor
from .config import ApiConfigLoader, ModelInfo
from .services import ModelLoader, PredictionService

# Get project root (3 levels up from this file: src/api/dependencies.py -> src/api -> src -> project_root)
_project_root = Path(__file__).resolve().parent.parent.parent


@lru_cache()
def get_api_config():
    """
    Učitava i kešira API konfiguraciju.

    Vraća keširan config nakon prvog poziva (singleton pattern).
    Putanja do konfiga može biti override-ovana preko CONFIG_PATH env variable.

    Returns:
        InferenceConfig sa API postavkama (host, port, model path, itd.)
    """
    config_path = os.getenv('CONFIG_PATH', 'configs/api.yaml')

    # Make path absolute if relative
    if not Path(config_path).is_absolute():
        config_path = _project_root / config_path

    api_config = ApiConfigLoader.from_yaml(str(config_path), _project_root)

    return api_config


@lru_cache()
def get_model_info() -> ModelInfo:
    """
    Učitava model informacije iz checkpoint-a.

    Učitava sve model parametre (architecture, tokenization)
    iz checkpoint foldera (config.yaml + metadata.json).

    Returns:
        ModelInfo sa kompletnim model parametrima

    Raises:
        RuntimeError: Ako checkpoint fajlovi nedostaju
    """
    api_config = get_api_config()
    checkpoint_dir = Path(api_config.model_checkpoint_dir)

    loader = ModelLoader(checkpoint_dir)
    model_info = loader.load_model_info()

    return model_info


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
    api_config = get_api_config()
    return api_config.model_checkpoint_dir


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
    api_config = get_api_config()
    model_info = get_model_info()  # Učitaj model info
    checkpoint_dir = Path(api_config.model_checkpoint_dir)

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

    # Path to category names file
    category_names_file = _project_root / "data" / "category_name.json"

    # Create and return prediction service
    service = PredictionService(
        classifier=classifier,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        class_map=class_map,
        top_k=api_config.top_k,
        category_names_file=str(category_names_file)
    )

    return service
