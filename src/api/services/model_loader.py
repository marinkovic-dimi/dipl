"""Model loader service for loading trained model artifacts."""

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
    """
    Handles loading of model artifacts from checkpoint directory.

    Loads and validates all required files:
    - classifier.keras: Trained model weights
    - tokenizer.json: Trained tokenizer
    - class_map.json: Class ID to class name mapping
    - config.yaml: Training configuration

    Args:
        checkpoint_dir: Path to the checkpoint directory containing artifacts

    Raises:
        FileNotFoundError: If checkpoint directory or required files don't exist
        ValueError: If loaded artifacts are invalid
    """

    REQUIRED_FILES = ['class_map.json']  # Only class_map is strictly required

    def __init__(
        self,
        checkpoint_dir: Path,
        tokenizer_path: str = None,
        model_filename: str = None
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.tokenizer_path = tokenizer_path  # Optional external tokenizer path
        self.model_filename = model_filename  # Optional custom model filename
        self._validate_checkpoint()
        self.logger.info(f"Initialized ModelLoader for: {self.checkpoint_dir}")

    def _validate_checkpoint(self) -> None:
        """Validates that checkpoint directory and required files exist."""
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {self.checkpoint_dir}"
            )

        # Check for model file - try multiple names
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

        # Check for class_map (required)
        if not (self.checkpoint_dir / 'class_map.json').exists():
            raise FileNotFoundError(
                f"class_map.json not found in {self.checkpoint_dir}. "
                f"Use create_class_map_from_data.py to generate it."
            )

        # Tokenizer can be in checkpoint or external
        tokenizer_in_checkpoint = (self.checkpoint_dir / 'tokenizer.json').exists()
        if not tokenizer_in_checkpoint and self.tokenizer_path is None:
            self.logger.warning(
                "tokenizer.json not in checkpoint and no external path provided. "
                "Will try to load from cache during tokenizer loading."
            )

        self.logger.info(f"Model file: {self.model_filename}")
        self.logger.info("All required checkpoint files found")

    def load_config(self) -> Dict:
        """
        Loads configuration from checkpoint directory.

        Returns:
            Configuration dictionary (may be None if config.yaml doesn't exist)
        """
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
        """
        Učitava kompletne model informacije iz checkpoint-a.

        Kombinuje podatke iz:
        - config.yaml (architecture parametri)
        - metadata.json (training rezultati)

        Returns:
            ModelInfo sa svim model parametrima

        Raises:
            RuntimeError: Ako potrebni fajlovi nedostaju
        """
        from ..config import ModelInfo

        # Učitaj config.yaml
        config = self.load_config()
        if not config or not hasattr(config, 'model'):
            raise RuntimeError(
                f"config.yaml missing or invalid in {self.checkpoint_dir}. "
                "Cannot load model parameters."
            )

        # Učitaj metadata.json (optional)
        metadata = self.load_metadata()
        if not metadata:
            self.logger.warning("metadata.json not found, using config only")
            metadata = {}

        # Kreiraj ModelInfo
        model_info = ModelInfo(
            # Tokenization (iz config)
            vocab_size=config.tokenization.vocab_size,
            max_length=config.tokenization.max_length,

            # Model architecture (iz config)
            embedding_dim=config.model.embedding_dim,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            ff_dim=config.model.ff_dim,
            dropout_rate=config.model.dropout_rate,
            pooling_strategy=config.model.pooling_strategy,
            label_smoothing=config.model.label_smoothing,

            # Metadata (iz metadata.json sa fallbacks)
            num_classes=metadata.get('num_classes', len(self.load_class_map())),
            experiment_name=metadata.get('experiment_name', config.experiment_name),
            timestamp=metadata.get('timestamp', config.timestamp),
            test_accuracy=metadata.get('test_accuracy'),
            test_top3_accuracy=metadata.get('test_top3_accuracy'),

            # Putanje
            checkpoint_dir=self.checkpoint_dir
        )

        self.logger.info(f"Loaded model info: {model_info.experiment_name}")
        return model_info

    def load_model(self) -> AdClassifier:
        """
        Loads and returns the trained classifier.

        The model is loaded with its trained weights. Architecture parameters
        are inferred from the saved model file.

        Returns:
            Loaded AdClassifier instance ready for inference

        Raises:
            RuntimeError: If model loading fails
        """
        model_path = self.checkpoint_dir / self.model_filename

        try:
            # Load config to get model architecture parameters
            config = self.load_config()

            # Create classifier instance with architecture params
            # If config is not available, create with default params (will be overridden by loaded weights)
            if config and hasattr(config, 'model'):
                classifier = AdClassifier(
                    vocab_size=config.tokenization.vocab_size,
                    num_classes=1,  # Will be overridden when loading weights
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
                # Fallback: create minimal classifier and let load_model override
                classifier = AdClassifier(vocab_size=1, num_classes=1)
                self.logger.warning("Creating classifier with minimal params (will load from file)")

            # Load the trained weights
            classifier.load_model(str(model_path))
            self.logger.info(f"Loaded model from: {model_path}")

            return classifier

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def load_tokenizer(self) -> WordPieceTokenizer:
        """
        Loads and returns the trained tokenizer.

        Tries to load from:
        1. External tokenizer_path (if provided in constructor)
        2. tokenizer.json in checkpoint directory
        3. Tokenizer cache directory (if specified in config)

        Returns:
            Loaded WordPieceTokenizer instance

        Raises:
            RuntimeError: If tokenizer loading fails
        """
        # Load metadata to get max_length
        metadata = self.load_metadata()
        max_length = metadata.get('max_length', 100) if metadata else 100

        # Try multiple locations
        tokenizer_paths = []

        # 1. External path provided in constructor
        if self.tokenizer_path:
            tokenizer_paths.append(Path(self.tokenizer_path))

        # 2. Checkpoint directory
        tokenizer_paths.append(self.checkpoint_dir / 'tokenizer.json')

        # 3. Look for tokenizer in cache if config available
        config = self.load_config()
        if config and hasattr(config, 'tokenization'):
            cache_dir = Path(config.tokenization.tokenizer_cache_dir)
            if cache_dir.exists():
                # Find any tokenizer json in cache
                cache_tokenizers = list(cache_dir.glob('tokenizer_*.json'))
                tokenizer_paths.extend(cache_tokenizers)

        # Try to load from first available path
        for tokenizer_path in tokenizer_paths:
            if tokenizer_path.exists():
                try:
                    tokenizer = WordPieceTokenizer(max_length=max_length)
                    tokenizer.load(str(tokenizer_path))
                    self.logger.info(f"Loaded tokenizer from: {tokenizer_path}")

                    # Log tokenizer stats
                    self.logger.info(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
                    self.logger.info(f"Tokenizer max length: {tokenizer.max_length}")

                    return tokenizer

                except Exception as e:
                    self.logger.warning(f"Failed to load tokenizer from {tokenizer_path}: {e}")
                    continue

        # If we get here, no tokenizer was found
        self.logger.error("Failed to load tokenizer from any location")
        raise RuntimeError(
            f"Tokenizer loading failed. Tried {len(tokenizer_paths)} locations. "
            f"Please provide tokenizer.json in checkpoint or cache directory."
        )

    def load_class_map(self) -> Dict[int, str]:
        """
        Loads class ID to class name mapping.

        Returns:
            Dictionary mapping class IDs (int) to class names (str)

        Raises:
            RuntimeError: If class map loading fails
            ValueError: If class map format is invalid
        """
        class_map_path = self.checkpoint_dir / 'class_map.json'

        try:
            with open(class_map_path, 'r', encoding='utf-8') as f:
                class_map_raw = json.load(f)

            # class_map_raw has format: {"group_id": index}
            # We need to invert it to: {index: "group_id"}
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
        """
        Loads training metadata if available.

        Returns:
            Metadata dictionary or empty dict if not found
        """
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
