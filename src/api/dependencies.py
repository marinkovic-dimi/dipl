import os
from pathlib import Path
from functools import lru_cache

from src.data.preprocessors import SerbianTextPreprocessor
from .config import ApiConfigLoader, ModelInfo
from .services import ModelLoader, PredictionService

_project_root = Path(__file__).resolve().parent.parent.parent


@lru_cache()
def get_api_config():
    config_path = os.getenv('CONFIG_PATH', 'configs/api.yaml')

    if not Path(config_path).is_absolute():
        config_path = _project_root / config_path

    api_config = ApiConfigLoader.from_yaml(str(config_path), _project_root)

    return api_config


@lru_cache()
def get_model_info() -> ModelInfo:
    api_config = get_api_config()
    checkpoint_dir = Path(api_config.model_checkpoint_dir)

    loader = ModelLoader(checkpoint_dir)
    model_info = loader.load_model_info()

    return model_info


@lru_cache()
def get_model_checkpoint_dir() -> str:
    env_checkpoint = os.getenv('MODEL_CHECKPOINT_DIR')

    if env_checkpoint:
        checkpoint_path = Path(env_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = _project_root / checkpoint_path
        return str(checkpoint_path)

    api_config = get_api_config()
    return api_config.model_checkpoint_dir


@lru_cache()
def get_prediction_service() -> PredictionService:
    api_config = get_api_config()
    model_info = get_model_info()  
    checkpoint_dir = Path(api_config.model_checkpoint_dir)

    loader = ModelLoader(checkpoint_dir)

    classifier = loader.load_model()
    tokenizer = loader.load_tokenizer()
    class_map = loader.load_class_map()

    preprocessor = SerbianTextPreprocessor(
        transliterate_cyrillic=True,
        lowercase=True,
        remove_stop_words=True,
        custom_transformations=[],
        verbose=False
    )

    category_names_file = _project_root / "data" / "category_name.json"

    service = PredictionService(
        classifier=classifier,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        class_map=class_map,
        top_k=api_config.top_k,
        category_names_file=str(category_names_file)
    )

    return service
