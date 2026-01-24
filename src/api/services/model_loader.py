import json
from pathlib import Path
from typing import Dict, TYPE_CHECKING

from src.models import AdClassifier
from src.tokenization import WordPieceTokenizer
from src.utils.config import ConfigManager
from src.utils.logging import LoggerMixin

if TYPE_CHECKING:
    from ..config import ModelInfo


class ModelLoader(LoggerMixin):
    REQUIRED_FILES = ['class_map.json']  

    def __init__(
        self,
        checkpoint_dir: Path,
        tokenizer_path: str = None,
        model_filename: str = None
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.tokenizer_path = tokenizer_path  
        self.model_filename = model_filename  
        self._validate_checkpoint()
        self.logger.info(f"Initialized ModelLoader for: {self.checkpoint_dir}")

    def _validate_checkpoint(self) -> None:
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {self.checkpoint_dir}"
            )

        model_files = ['classifier.keras', 'checkpoint.keras', 'model.keras']
        model_found = False
        for model_file in model_files:
            if (self.checkpoint_dir / model_file).exists():
                if self.model_filename is None:
                    self.model_filename = model_file
                model_found = True
                break

        if not model_found:
            raise FileNotFoundError(
                f"No model file found in {self.checkpoint_dir}. "
                f"Tried: {model_files}"
            )

        if not (self.checkpoint_dir / 'class_map.json').exists():
            raise FileNotFoundError(
                f"class_map.json not found in {self.checkpoint_dir}. "
                f"Use create_class_map_from_data.py to generate it."
            )

        tokenizer_in_checkpoint = (self.checkpoint_dir / 'tokenizer.json').exists()
        if not tokenizer_in_checkpoint and self.tokenizer_path is None:
            self.logger.warning(
                "tokenizer.json not in checkpoint and no external path provided. "
                "Will try to load from cache during tokenizer loading."
            )

        self.logger.info(f"Model file: {self.model_filename}")
        self.logger.info("All required checkpoint files found")

    def load_config(self) -> Dict:
        config_path = self.checkpoint_dir / 'config.yaml'
        if not config_path.exists():
            self.logger.warning("config.yaml not found in checkpoint, using defaults")
            return {}

        try:
            config = ConfigManager.from_yaml(str(config_path))
            self.logger.info("Loaded config from checkpoint")
            return config
        except Exception as e:
            self.logger.warning(f"Failed to load config.yaml: {e}")
            return {}

    def load_model_info(self) -> 'ModelInfo':
        from ..config import ModelInfo

        config = self.load_config()
        if not config or not hasattr(config, 'model'):
            raise RuntimeError(
                f"config.yaml missing or invalid in {self.checkpoint_dir}. "
                "Cannot load model parameters."
            )

        metadata = self.load_metadata()
        if not metadata:
            self.logger.warning("metadata.json not found, using config only")
            metadata = {}

        model_info = ModelInfo(
            vocab_size=config.tokenization.vocab_size,
            max_length=config.tokenization.max_length,
            embedding_dim=config.model.embedding_dim,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            ff_dim=config.model.ff_dim,
            dropout_rate=config.model.dropout_rate,
            pooling_strategy=config.model.pooling_strategy,
            label_smoothing=config.model.label_smoothing,
            num_classes=metadata.get('num_classes', len(self.load_class_map())),
            experiment_name=metadata.get('experiment_name', config.experiment_name),
            timestamp=metadata.get('timestamp', config.timestamp),
            test_accuracy=metadata.get('test_accuracy'),
            test_top3_accuracy=metadata.get('test_top3_accuracy'),
            checkpoint_dir=self.checkpoint_dir
        )

        self.logger.info(f"Loaded model info: {model_info.experiment_name}")
        return model_info

    def load_model(self) -> AdClassifier:
        model_path = self.checkpoint_dir / self.model_filename

        try:
            config = self.load_config()

            if config and hasattr(config, 'model'):
                classifier = AdClassifier(
                    vocab_size=config.tokenization.vocab_size,
                    num_classes=1,  
                    max_length=config.tokenization.max_length,
                    embed_dim=config.model.embedding_dim,
                    num_heads=config.model.num_heads,
                    num_layers=config.model.num_layers,
                    ff_dim=config.model.ff_dim,
                    dropout_rate=config.model.dropout_rate,
                    pooling_strategy=config.model.pooling_strategy,
                    label_smoothing=config.model.label_smoothing
                )
                self.logger.info("Created classifier from checkpoint config")
            else:
                classifier = AdClassifier(vocab_size=1, num_classes=1)
                self.logger.warning("Creating classifier with minimal params (will load from file)")

            classifier.load_model(str(model_path))
            self.logger.info(f"Loaded model from: {model_path}")

            return classifier

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def load_tokenizer(self) -> WordPieceTokenizer:
        metadata = self.load_metadata()
        max_length = metadata.get('max_length', 100) if metadata else 100

        tokenizer_paths = []

        if self.tokenizer_path:
            tokenizer_paths.append(Path(self.tokenizer_path))

        tokenizer_paths.append(self.checkpoint_dir / 'tokenizer.json')

        config = self.load_config()
        if config and hasattr(config, 'tokenization'):
            cache_dir = Path(config.tokenization.tokenizer_cache_dir)
            if cache_dir.exists():
                cache_tokenizers = list(cache_dir.glob('tokenizer_*.json'))
                tokenizer_paths.extend(cache_tokenizers)

        for tokenizer_path in tokenizer_paths:
            if tokenizer_path.exists():
                try:
                    tokenizer = WordPieceTokenizer(max_length=max_length)
                    tokenizer.load(str(tokenizer_path))
                    self.logger.info(f"Loaded tokenizer from: {tokenizer_path}")
                    self.logger.info(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
                    self.logger.info(f"Tokenizer max length: {tokenizer.max_length}")

                    return tokenizer

                except Exception as e:
                    self.logger.warning(f"Failed to load tokenizer from {tokenizer_path}: {e}")
                    continue

        self.logger.error("Failed to load tokenizer from any location")
        raise RuntimeError(
            f"Tokenizer loading failed. Tried {len(tokenizer_paths)} locations. "
            f"Please provide tokenizer.json in checkpoint or cache directory."
        )

    def load_class_map(self) -> Dict[int, str]:
        class_map_path = self.checkpoint_dir / 'class_map.json'

        try:
            with open(class_map_path, 'r', encoding='utf-8') as f:
                class_map_raw = json.load(f)

            class_map = {v: k for k, v in class_map_raw.items()}
            self.logger.info(f"Loaded class map with {len(class_map)} classes")
            self.logger.debug(f"Sample classes: {list(class_map.items())[:5]}")
            return class_map

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in class_map.json: {e}")
            raise RuntimeError(f"Failed to parse class_map.json: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load class map: {e}")
            raise RuntimeError(f"Class map loading failed: {e}")

    def load_metadata(self) -> Dict:
        metadata_path = self.checkpoint_dir / 'metadata.json'

        if not metadata_path.exists():
            self.logger.warning("metadata.json not found in checkpoint")
            return {}

        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            self.logger.info("Loaded training metadata")
            if 'test_accuracy' in metadata:
                self.logger.info(f"Model test accuracy: {metadata['test_accuracy']:.4f}")

            return metadata

        except Exception as e:
            self.logger.warning(f"Failed to load metadata.json: {e}")
            return {}
