import hashlib
import json
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime

from ..utils.logging import LoggerMixin
from ..utils.config.data_config import DataConfig
from ..utils.config.tokenization_config import TokenizationConfig
from ..utils.config.balancing_config import BalancingConfig
from ..data.preprocess import get_preprocessing_hash


class TokenizedDatasetCache(LoggerMixin):
    def __init__(
        self,
        cache_dir: str = "cache/tokenized_datasets",
        verbose: bool = True
    ):
        self.cache_dir = Path(cache_dir)
        self.verbose = verbose

    def compute_cache_key(
        self,
        data_config: DataConfig,
        tokenization_config: TokenizationConfig,
        split_config: Dict[str, Any],
        balancing_config: Optional[BalancingConfig] = None
    ) -> str:
        hash_components = {
            'data_hash': get_preprocessing_hash(data_config),
            'vocab_size': tokenization_config.vocab_size,
            'max_length': tokenization_config.max_length,
            'min_frequency': tokenization_config.min_frequency,
            'val_size': split_config.get('val_size', 0.15),
            'test_size': split_config.get('test_size', 0.15),
            'random_state': split_config.get('random_state', 42),
        }

        if balancing_config and balancing_config.strategy != "none":
            hash_components['balancing'] = {
                'strategy': balancing_config.strategy,
                'target_threshold': balancing_config.target_threshold,
                'target_threshold_small': balancing_config.target_threshold_small,
                'increase_factor': balancing_config.increase_factor,
                'decrease_factor': balancing_config.decrease_factor,
            }

        hash_str = str(sorted(hash_components.items()))
        cache_key = hashlib.md5(hash_str.encode()).hexdigest()[:8]

        if self.verbose:
            self.logger.debug(f"Computed cache key: {cache_key}")
            self.logger.debug(f"  Data hash: {hash_components['data_hash']}")
            self.logger.debug(f"  Tokenization: vocab={hash_components['vocab_size']}, "
                            f"max_len={hash_components['max_length']}, "
                            f"min_freq={hash_components['min_frequency']}")
            self.logger.debug(f"  Split: val={hash_components['val_size']}, "
                            f"test={hash_components['test_size']}, "
                            f"seed={hash_components['random_state']}")

        return cache_key

    def get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / cache_key

    def exists(self, cache_key: str) -> bool:
        cache_path = self.get_cache_path(cache_key)

        if not cache_path.exists():
            return False

        required_files = [
            'X_train.npy', 'X_val.npy', 'X_test.npy',
            'y_train.npy', 'y_val.npy', 'y_test.npy',
            'metadata.json'
        ]

        all_exist = all((cache_path / f).exists() for f in required_files)

        if all_exist and self.verbose:
            self.logger.debug(f"Valid cache found for key: {cache_key}")

        return all_exist

    def _convert_numpy_types(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {self._convert_numpy_types(k): self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def save(
        self,
        cache_key: str,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        metadata: Dict[str, Any]
    ) -> None:
        cache_path = self.get_cache_path(cache_key)

        try:
            cache_path.mkdir(parents=True, exist_ok=True)

            np.save(cache_path / 'X_train.npy', X_train)
            np.save(cache_path / 'X_val.npy', X_val)
            np.save(cache_path / 'X_test.npy', X_test)
            np.save(cache_path / 'y_train.npy', y_train)
            np.save(cache_path / 'y_val.npy', y_val)
            np.save(cache_path / 'y_test.npy', y_test)

            metadata['cache_key'] = cache_key
            metadata['created_at'] = datetime.now().isoformat()
            metadata['shapes'] = {
                'X_train': list(X_train.shape),
                'X_val': list(X_val.shape),
                'X_test': list(X_test.shape),
                'y_train': list(y_train.shape),
                'y_val': list(y_val.shape),
                'y_test': list(y_test.shape),
            }
            metadata['dtypes'] = {
                'X': str(X_train.dtype),
                'y': str(y_train.dtype)
            }

            metadata_serializable = self._convert_numpy_types(metadata)
            with open(cache_path / 'metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata_serializable, f, indent=2, ensure_ascii=False)

            if self.verbose:
                self.logger.info(f"✓ Saved tokenized dataset cache: {cache_key}")
                self.logger.info(f"  Location: {cache_path}")
                self.logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        except PermissionError as e:
            self.logger.error(f"Permission denied when saving cache: {e}")
            self.logger.error("Continuing without caching...")
            if cache_path.exists():
                shutil.rmtree(cache_path, ignore_errors=True)
        except OSError as e:
            self.logger.error(f"OS error when saving cache: {e}")
            self.logger.error("Continuing without caching...")
            if cache_path.exists():
                shutil.rmtree(cache_path, ignore_errors=True)

    def load(
        self,
        cache_key: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        if not self.exists(cache_key):
            raise FileNotFoundError(f"Cache not found for key: {cache_key}")

        cache_path = self.get_cache_path(cache_key)

        try:
            X_train = np.load(cache_path / 'X_train.npy')
            X_val = np.load(cache_path / 'X_val.npy')
            X_test = np.load(cache_path / 'X_test.npy')
            y_train = np.load(cache_path / 'y_train.npy')
            y_val = np.load(cache_path / 'y_val.npy')
            y_test = np.load(cache_path / 'y_test.npy')

            with open(cache_path / 'metadata.json', 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            expected_shapes = metadata.get('shapes', {})
            if expected_shapes:
                assert list(X_train.shape) == expected_shapes.get('X_train', []), \
                    f"X_train shape mismatch: {X_train.shape} != {expected_shapes['X_train']}"

            if self.verbose:
                self.logger.info(f"✓ Loaded tokenized datasets from cache: {cache_key}")
                self.logger.info(f"  Location: {cache_path}")
                self.logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
                self.logger.info(f"  Cached on: {metadata.get('created_at', 'unknown')}")

            return X_train, X_val, X_test, y_train, y_val, y_test, metadata

        except Exception as e:
            self.logger.error(f"Error loading cache: {e}")
            self.logger.warning(f"Cache corrupted for key {cache_key}, will re-tokenize")
            if cache_path.exists():
                shutil.rmtree(cache_path, ignore_errors=True)
            raise ValueError(f"Corrupted cache: {e}")

    def clear_cache(self, cache_key: Optional[str] = None) -> None:
        if cache_key:
            cache_path = self.get_cache_path(cache_key)
            if cache_path.exists():
                shutil.rmtree(cache_path)
                if self.verbose:
                    self.logger.info(f"Cleared cache: {cache_key}")
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                if self.verbose:
                    self.logger.info("Cleared all tokenized dataset cache")

    def list_cached_datasets(self) -> List[Dict[str, Any]]:
        cached = []
        if not self.cache_dir.exists():
            return cached

        for cache_path in self.cache_dir.iterdir():
            if cache_path.is_dir():
                metadata_path = cache_path / 'metadata.json'
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        cached.append({
                            'cache_key': cache_path.name,
                            'created_at': metadata.get('created_at'),
                            'num_classes': metadata.get('num_classes'),
                            'train_samples': metadata.get('train_samples'),
                            'val_samples': metadata.get('val_samples'),
                            'test_samples': metadata.get('test_samples'),
                            'vocab_size': metadata.get('vocab_size'),
                            'max_length': metadata.get('max_length'),
                        })
                    except Exception as e:
                        self.logger.warning(f"Could not read metadata from {cache_path.name}: {e}")

        return cached
