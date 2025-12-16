import re
import hashlib
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import List, Dict, Union, Optional
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from tokenizers.models import WordPiece

from ...utils.logging import LoggerMixin
from ...utils.serialization import SerializationManager, ensure_dir


class EnhancedTokenizer(LoggerMixin):

    def __init__(
        self,
        vocab_size: int = 15000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
        max_length: int = 100,
        cache_dir: str = "cache/tokenizers",
        verbose: bool = True
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or ["[PAD]", "[CLS]", "[UNK]", "[SEP]", "[MASK]"]
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        self.verbose = verbose

        self.special_tokens.extend([str(i) for i in range(10)])

        self.tokenizer = None
        self._is_trained = False

    def _get_cache_key(self, texts: pd.Series) -> str:
        config_str = f"{self.vocab_size}_{self.min_frequency}_{len(self.special_tokens)}_{self.max_length}"

        text_sample = texts.head(1000).str.cat(sep=" ")
        text_hash = hashlib.md5(text_sample.encode('utf-8')).hexdigest()[:8]

        return f"tokenizer_{config_str}_{text_hash}"

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _preprocess_text_for_tokenization(self, text: str) -> str:
        text = re.sub(r'(\d+)', r' \1 ', text)
        text = re.sub(r'(\d)', r' \1 ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def train(
        self,
        texts: pd.Series,
        use_cache: bool = True,
        save_cache: bool = True
    ) -> None:
        cache_key = self._get_cache_key(texts)
        cache_path = self._get_cache_path(cache_key)

        if use_cache and cache_path.exists():
            try:
                self.tokenizer = Tokenizer.from_file(str(cache_path))
                self._is_trained = True
                if self.verbose:
                    self.logger.info(f"Loaded cached tokenizer from {cache_path}")
                return
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Failed to load cached tokenizer: {e}")

        if self.verbose:
            self.logger.info(f"Training new tokenizer with vocab_size={self.vocab_size}")

        processed_texts = texts.apply(self._preprocess_text_for_tokenization)

        self.tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()

        trainer = WordPieceTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            show_progress=self.verbose,
            special_tokens=self.special_tokens
        )

        self.tokenizer.train_from_iterator(processed_texts, trainer=trainer)
        self._is_trained = True

        if save_cache:
            ensure_dir(self.cache_dir)
            self.tokenizer.save(str(cache_path))
            if self.verbose:
                self.logger.info(f"Saved tokenizer to cache: {cache_path}")

        if self.verbose:
            self.logger.info("Tokenizer training completed")

    def encode_text(self, text: str, add_special_tokens: bool = True) -> List[str]:
        if not self._is_trained:
            raise RuntimeError("Tokenizer must be trained before encoding")

        processed_text = self._preprocess_text_for_tokenization(text)
        tokens = self.tokenizer.encode(processed_text).tokens

        if add_special_tokens:
            tokens = ["[CLS]"] + tokens + ["[SEP]"]

        return tokens

    def encode_to_ids(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self.encode_text(text, add_special_tokens)
        return [self.tokenizer.token_to_id(token) for token in tokens]

    def encode_batch(
        self,
        texts: Union[List[str], pd.Series],
        add_special_tokens: bool = True,
        pad_to_max_length: bool = True
    ) -> Union[List[List[int]], tf.Tensor]:
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        encoded = [self.encode_to_ids(text, add_special_tokens) for text in texts]

        if pad_to_max_length:
            pad_id = self.tokenizer.token_to_id("[PAD]")
            padded = tf.keras.preprocessing.sequence.pad_sequences(
                encoded,
                maxlen=self.max_length,
                padding='post',
                value=pad_id
            )
            return padded

        return encoded

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        if not self._is_trained:
            raise RuntimeError("Tokenizer must be trained before decoding")

        tokens = [self.tokenizer.id_to_token(id) for id in token_ids if id is not None]

        if skip_special_tokens:
            special_tokens_set = set(self.special_tokens)
            tokens = [token for token in tokens if token not in special_tokens_set]

        return " ".join(tokens)

    def get_vocab_size(self) -> int:
        if not self._is_trained:
            return 0
        return self.tokenizer.get_vocab_size()

    def get_vocab(self) -> Dict[str, int]:
        if not self._is_trained:
            return {}
        return self.tokenizer.get_vocab()

    def analyze_tokenization(self, texts: pd.Series, sample_size: int = 100) -> Dict:
        if not self._is_trained:
            raise RuntimeError("Tokenizer must be trained before analysis")

        sample_texts = texts.head(sample_size)

        analysis = {
            'avg_tokens_per_text': 0,
            'max_tokens': 0,
            'min_tokens': float('inf'),
            'texts_exceeding_max_length': 0,
            'token_length_distribution': {},
            'most_common_tokens': {},
            'vocab_size': self.get_vocab_size()
        }

        token_lengths = []
        all_tokens = []

        for text in sample_texts:
            tokens = self.encode_text(text, add_special_tokens=True)
            token_length = len(tokens)

            token_lengths.append(token_length)
            all_tokens.extend(tokens)

            analysis['max_tokens'] = max(analysis['max_tokens'], token_length)
            analysis['min_tokens'] = min(analysis['min_tokens'], token_length)

            if token_length > self.max_length:
                analysis['texts_exceeding_max_length'] += 1

        analysis['avg_tokens_per_text'] = sum(token_lengths) / len(token_lengths)

        length_counts = pd.Series(token_lengths).value_counts()
        analysis['token_length_distribution'] = length_counts.head(10).to_dict()

        token_counts = pd.Series(all_tokens).value_counts()
        analysis['most_common_tokens'] = token_counts.head(20).to_dict()

        return analysis

    def save(self, file_path: Union[str, Path]) -> None:
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained tokenizer")

        file_path = Path(file_path)
        ensure_dir(file_path.parent)

        self.tokenizer.save(str(file_path))
        if self.verbose:
            self.logger.info(f"Saved tokenizer to {file_path}")

    def load(self, file_path: Union[str, Path]) -> None:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {file_path}")

        self.tokenizer = Tokenizer.from_file(str(file_path))
        self._is_trained = True

        if self.verbose:
            self.logger.info(f"Loaded tokenizer from {file_path}")
