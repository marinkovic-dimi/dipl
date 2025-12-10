import pandas as pd
from collections import Counter
from typing import List, Dict, Set, Optional, Tuple
from ..utils.logging import LoggerMixin


class VocabularyBuilder(LoggerMixin):
    """Builder for vocabulary analysis and management."""

    def __init__(self, min_frequency: int = 2, max_vocab_size: Optional[int] = None):
        """
        Initialize vocabulary builder.

        Args:
            min_frequency: Minimum frequency for words to be included
            max_vocab_size: Maximum vocabulary size (None for unlimited)
        """
        self.min_frequency = min_frequency
        self.max_vocab_size = max_vocab_size

    def build_word_vocabulary(
        self,
        texts: pd.Series,
        special_tokens: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Build word-level vocabulary from texts.

        Args:
            texts: Series of texts
            special_tokens: Special tokens to include

        Returns:
            Vocabulary mapping (word -> id)
        """
        special_tokens = special_tokens or []

        self.logger.info("Building word vocabulary...")

        all_words = []
        for text in texts:
            if pd.notna(text):
                words = str(text).split()
                all_words.extend(words)

        word_counts = Counter(all_words)
        self.logger.info(f"Found {len(word_counts)} unique words")

        filtered_words = {
            word: count for word, count in word_counts.items()
            if count >= self.min_frequency
        }
        self.logger.info(f"After frequency filtering: {len(filtered_words)} words")

        sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)

        if self.max_vocab_size:
            vocab_limit = self.max_vocab_size - len(special_tokens)
            sorted_words = sorted_words[:vocab_limit]
            self.logger.info(f"After size limiting: {len(sorted_words)} words")

        vocab = {}

        for i, token in enumerate(special_tokens):
            vocab[token] = i

        start_id = len(special_tokens)
        for i, (word, _) in enumerate(sorted_words):
            vocab[word] = start_id + i

        self.logger.info(f"Final vocabulary size: {len(vocab)}")
        return vocab

    def analyze_vocabulary_coverage(
        self,
        texts: pd.Series,
        vocabulary: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Analyze vocabulary coverage on texts.

        Args:
            texts: Series of texts to analyze
            vocabulary: Vocabulary mapping

        Returns:
            Coverage statistics
        """
        vocab_words = set(vocabulary.keys())
        total_words = 0
        covered_words = 0
        oov_words = set()

        for text in texts:
            if pd.notna(text):
                words = str(text).split()
                total_words += len(words)

                for word in words:
                    if word in vocab_words:
                        covered_words += 1
                    else:
                        oov_words.add(word)

        coverage_rate = covered_words / total_words if total_words > 0 else 0

        return {
            'coverage_rate': coverage_rate,
            'total_words': total_words,
            'covered_words': covered_words,
            'oov_words': len(oov_words),
            'vocabulary_size': len(vocabulary)
        }

    def get_vocabulary_statistics(
        self,
        texts: pd.Series,
        vocabulary: Optional[Dict[str, int]] = None
    ) -> Dict:
        """
        Get comprehensive vocabulary statistics.

        Args:
            texts: Series of texts
            vocabulary: Optional vocabulary for coverage analysis

        Returns:
            Statistics dictionary
        """
        all_words = []
        text_lengths = []

        for text in texts:
            if pd.notna(text):
                words = str(text).split()
                all_words.extend(words)
                text_lengths.append(len(words))

        word_counts = Counter(all_words)

        stats = {
            'total_texts': len(texts),
            'total_words': len(all_words),
            'unique_words': len(word_counts),
            'avg_text_length': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            'max_text_length': max(text_lengths) if text_lengths else 0,
            'min_text_length': min(text_lengths) if text_lengths else 0
        }

        frequency_distribution = Counter(word_counts.values())
        stats['frequency_distribution'] = dict(sorted(frequency_distribution.items()))

        stats['most_common_words'] = dict(word_counts.most_common(20))

        for threshold in [1, 2, 5, 10]:
            words_above_threshold = sum(1 for count in word_counts.values() if count >= threshold)
            stats[f'words_freq_>={threshold}'] = words_above_threshold

        if vocabulary:
            coverage_stats = self.analyze_vocabulary_coverage(texts, vocabulary)
            stats.update(coverage_stats)

        return stats

    def find_optimal_vocab_size(
        self,
        texts: pd.Series,
        vocab_sizes: List[int],
        target_coverage: float = 0.95
    ) -> Tuple[int, Dict]:
        """
        Find optimal vocabulary size for target coverage.

        Args:
            texts: Series of texts
            vocab_sizes: List of vocabulary sizes to test
            target_coverage: Target coverage rate

        Returns:
            Tuple of (optimal_vocab_size, coverage_results)
        """
        self.logger.info(f"Finding optimal vocabulary size for {target_coverage:.1%} coverage")

        coverage_results = {}

        for vocab_size in sorted(vocab_sizes):
            temp_builder = VocabularyBuilder(
                min_frequency=self.min_frequency,
                max_vocab_size=vocab_size
            )
            vocab = temp_builder.build_word_vocabulary(texts)

            coverage_stats = self.analyze_vocabulary_coverage(texts, vocab)
            coverage_results[vocab_size] = coverage_stats

            self.logger.info(f"Vocab size {vocab_size}: {coverage_stats['coverage_rate']:.3f} coverage")

            if coverage_stats['coverage_rate'] >= target_coverage:
                self.logger.info(f"Target coverage reached with vocab size: {vocab_size}")
                return vocab_size, coverage_results

        best_vocab_size = max(vocab_sizes)
        self.logger.info(f"Target coverage not reached. Best vocab size: {best_vocab_size}")
        return best_vocab_size, coverage_results

    def export_vocabulary(
        self,
        vocabulary: Dict[str, int],
        output_path: str,
        format: str = "json"
    ) -> None:
        """
        Export vocabulary to file.

        Args:
            vocabulary: Vocabulary mapping
            output_path: Output file path
            format: Export format ('json', 'txt')
        """
        from ..utils.serialization import SerializationManager

        if format == "json":
            SerializationManager.save_json(vocabulary, output_path)
        elif format == "txt":
            sorted_vocab = sorted(vocabulary.items(), key=lambda x: x[1])
            with open(output_path, 'w', encoding='utf-8') as f:
                for word, word_id in sorted_vocab:
                    f.write(f"{word_id}\t{word}\n")
        else:
            raise ValueError(f"Unsupported export format: {format}")

        self.logger.info(f"Vocabulary exported to {output_path}")

    def load_vocabulary(self, input_path: str, format: str = "json") -> Dict[str, int]:
        """
        Load vocabulary from file.

        Args:
            input_path: Input file path
            format: File format ('json', 'txt')

        Returns:
            Vocabulary mapping
        """
        from ..utils.serialization import SerializationManager

        if format == "json":
            vocabulary = SerializationManager.load_json(input_path)
        elif format == "txt":
            vocabulary = {}
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        word_id, word = parts
                        vocabulary[word] = int(word_id)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Vocabulary loaded from {input_path}")
        return vocabulary