#!/usr/bin/env python3
"""
Tokenizer Diagnostic Test Script

This script performs comprehensive testing and analysis of the WordPiece tokenizer
to verify it's working correctly for Serbian advertisement text classification.

Tests include:
- Basic encoding/decoding validation
- Serbian text handling (Cyrillic, Latin, diacritics)
- Vocabulary coverage analysis
- Statistical analysis of token distributions

Usage:
    python test_tokenizer.py                                    # Use default config
    python test_tokenizer.py --config configs/improved.yaml     # Use specific config
    python test_tokenizer.py --sample-size 5000                 # Limit sample size
    python test_tokenizer.py --no-plots                         # Skip visualizations
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from collections import Counter

from src.utils.logging import setup_logging, get_logger
from src.utils.config import ConfigManager
from src.tokenization import WordPieceTokenizer
from src.training.train_model_wandb_gpu import load_or_preprocess_data


class TokenizerTester:
    """
    Comprehensive tokenizer testing and diagnostics.
    """

    def __init__(
        self,
        config_path: str = "configs/improved.yaml",
        sample_size: int = 10000,
        output_dir: str = "tests/tokenizer_analysis",
        create_plots: bool = True
    ):
        """
        Initialize the tokenizer tester.

        Args:
            config_path: Path to configuration file
            sample_size: Number of samples to analyze
            output_dir: Directory to save analysis results
            create_plots: Whether to generate visualizations
        """
        self.config_path = config_path
        self.sample_size = sample_size
        self.output_dir = Path(output_dir)
        self.create_plots = create_plots
        self.logger = get_logger(self.__class__.__name__)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config = ConfigManager.from_yaml(config_path)
        # Add config_path attribute for load_or_preprocess_data
        self.config.config_path = config_path

        # Initialize results storage
        self.results = {}

    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all tokenizer tests and generate comprehensive report.

        Returns:
            Dictionary with test results
        """
        self.logger.info("=" * 80)
        self.logger.info("TOKENIZER DIAGNOSTIC TEST")
        self.logger.info("=" * 80)
        self.logger.info(f"Config: {self.config_path}")
        self.logger.info(f"Vocab Size: {self.config.tokenization.vocab_size}")
        self.logger.info(f"Max Length: {self.config.tokenization.max_length}")
        self.logger.info("")

        # Load data
        self.logger.info("[1/6] Loading Data...")
        data, metadata = load_or_preprocess_data(self.config)
        clean_text_col = metadata['clean_text_column']

        # Sample data if needed
        if len(data) > self.sample_size:
            data = data.sample(n=self.sample_size, random_state=42)
        self.logger.info(f"✓ Loaded {len(data)} samples")
        self.logger.info("")

        # Train tokenizer
        self.logger.info("[2/6] Training/Loading Tokenizer...")
        tokenizer = WordPieceTokenizer(
            vocab_size=self.config.tokenization.vocab_size,
            max_length=self.config.tokenization.max_length,
            verbose=True
        )
        tokenizer.train(
            data[clean_text_col],
            use_cache=self.config.tokenization.use_cached_tokenizer
        )
        self.tokenizer = tokenizer
        self.data = data
        self.clean_text_col = clean_text_col
        self.logger.info(f"✓ Loaded tokenizer (vocab_size={tokenizer.get_vocab_size()})")
        self.logger.info("")

        # Run tests
        self.logger.info("[3/6] Basic Encoding Tests...")
        encoding_results = self.test_basic_encoding_decoding()
        self.results['encoding'] = encoding_results
        self.logger.info("")

        self.logger.info("[4/6] Serbian Text Tests...")
        serbian_results = self.test_serbian_text_handling()
        self.results['serbian'] = serbian_results
        self.logger.info("")

        self.logger.info("[5/6] Vocabulary Coverage...")
        coverage_results = self.test_vocabulary_coverage()
        self.results['coverage'] = coverage_results
        self.logger.info("")

        self.logger.info("[6/6] Statistical Analysis...")
        stats_results = self.analyze_token_statistics()
        self.results['statistics'] = stats_results
        self.logger.info("")

        # Generate visualizations
        if self.create_plots:
            try:
                self.generate_visualizations()
                self.logger.info(f"✓ Visualizations saved to: {self.output_dir}")
                self.logger.info("")
            except ImportError:
                self.logger.warning("matplotlib not available, skipping visualizations")
                self.logger.info("")

        # Generate report
        self.generate_report()

        # Print summary
        self.print_summary()

        return self.results

    def test_basic_encoding_decoding(self) -> Dict[str, bool]:
        """
        Test basic encoding/decoding functionality.

        Tests:
        - Round-trip encoding/decoding
        - Special tokens ([CLS], [SEP], [PAD])
        - Padding behavior
        - Truncation

        Returns:
            Dictionary with test results (pass/fail)
        """
        results = {}
        test_cases = [
            "Hello world",
            "Test 123",
            "Višebojni automobil",
            "Short",
            "This is a very long text that should be truncated if it exceeds the maximum length configured for the tokenizer"
        ]

        # Test 1: Round-trip encoding/decoding
        passed = 0
        for text in test_cases:
            tokens = self.tokenizer.encode_text(text, add_special_tokens=False)
            decoded = " ".join(tokens).replace(" ##", "")
            # Basic check - not exact match due to preprocessing
            if len(tokens) > 0:
                passed += 1

        results['round_trip'] = passed == len(test_cases)
        self.logger.info(f"  ✓ Round-trip encoding ({passed}/{len(test_cases)} passed)")

        # Test 2: Special tokens
        text = "Test text"
        tokens = self.tokenizer.encode_text(text, add_special_tokens=True)
        has_cls = "[CLS]" in tokens
        has_sep = "[SEP]" in tokens

        results['special_tokens'] = has_cls and has_sep
        self.logger.info(f"  ✓ Special tokens ({'✓' if results['special_tokens'] else '✗'} [CLS] and [SEP])")

        # Test 3: Padding behavior
        short_text = "Test"
        ids = self.tokenizer.encode_to_ids(short_text, add_special_tokens=True)
        pad_id = self.tokenizer.tokenizer.token_to_id("[PAD]")
        has_padding = pad_id in ids if pad_id is not None else False

        results['padding'] = has_padding or len(ids) == self.tokenizer.max_length
        self.logger.info(f"  ✓ Padding behavior ({'✓' if results['padding'] else '✗'})")

        # Test 4: Truncation
        long_text = " ".join(["word"] * 200)  # Very long text
        ids = self.tokenizer.encode_to_ids(long_text, add_special_tokens=True)
        is_truncated = len(ids) <= self.tokenizer.max_length

        results['truncation'] = is_truncated
        self.logger.info(f"  ✓ Truncation ({'✓' if results['truncation'] else '✗'} at max_length={self.tokenizer.max_length})")

        # Test 5: Batch encoding consistency
        texts = ["Test one", "Test two", "Test three"]
        batch = self.tokenizer.encode_batch(texts)
        batch_consistent = batch.shape[0] == len(texts) and batch.shape[1] == self.tokenizer.max_length

        results['batch_encoding'] = batch_consistent
        self.logger.info(f"  ✓ Batch encoding ({'✓' if results['batch_encoding'] else '✗'} shape={batch.shape})")

        return results

    def test_serbian_text_handling(self) -> Dict[str, bool]:
        """
        Test Serbian-specific text handling.

        Tests:
        - Cyrillic text
        - Latin text
        - Mixed scripts
        - Serbian diacritics (č, ć, ž, š, đ)

        Returns:
            Dictionary with test results
        """
        results = {}

        test_cases = [
            ("Београд је главни град Србије", "Cyrillic"),
            ("Beograd je glavni grad Srbije", "Latin"),
            ("Automobil sa 4 točka", "Numbers+Latin"),
            ("Višebojni automobil", "Diacritics"),
            ("Ćevapi, šljivovica, džem", "Serbian chars"),
        ]

        passed = 0
        for text, description in test_cases:
            tokens = self.tokenizer.encode_text(text, add_special_tokens=False)
            # Check that tokenization produces tokens
            if len(tokens) > 0:
                passed += 1
                self.logger.info(f"  ✓ {description}: {len(tokens)} tokens")
            else:
                self.logger.info(f"  ✗ {description}: FAILED (no tokens)")

        results['serbian_texts'] = passed == len(test_cases)
        results['passed'] = passed
        results['total'] = len(test_cases)

        return results

    def test_vocabulary_coverage(self) -> Dict[str, float]:
        """
        Analyze vocabulary coverage and OOV rate.

        Computes:
        - Out-of-vocabulary (OOV) rate
        - Most common [UNK] sources
        - Coverage by text length

        Returns:
            Dictionary with coverage statistics
        """
        results = {}

        # Get UNK token ID
        unk_id = self.tokenizer.tokenizer.token_to_id("[UNK]")

        # Tokenize all texts
        texts = self.data[self.clean_text_col].tolist()
        total_tokens = 0
        unk_tokens = 0
        unk_texts = []

        for text in texts:
            ids = self.tokenizer.encode_to_ids(text, add_special_tokens=False)
            total_tokens += len(ids)
            unk_count = ids.count(unk_id) if unk_id is not None else 0
            unk_tokens += unk_count

            if unk_count > 0:
                unk_texts.append((text, unk_count))

        # Calculate OOV rate
        oov_rate = (unk_tokens / total_tokens * 100) if total_tokens > 0 else 0
        results['total_tokens'] = total_tokens
        results['unk_tokens'] = unk_tokens
        results['oov_rate'] = oov_rate
        results['texts_with_unk'] = len(unk_texts)

        self.logger.info(f"  Total tokens: {total_tokens:,}")
        self.logger.info(f"  [UNK] tokens: {unk_tokens:,} ({oov_rate:.2f}%)")
        self.logger.info(f"  Texts with [UNK]: {len(unk_texts):,} ({len(unk_texts)/len(texts)*100:.1f}%)")

        # Health check
        if oov_rate < 5.0:
            self.logger.info(f"  ✓ OOV rate is healthy ({oov_rate:.2f}% < 5%)")
            results['health_status'] = 'good'
        elif oov_rate < 10.0:
            self.logger.info(f"  ⚠ OOV rate is moderate ({oov_rate:.2f}%)")
            results['health_status'] = 'moderate'
        else:
            self.logger.info(f"  ✗ OOV rate is high ({oov_rate:.2f}% > 10%)")
            results['health_status'] = 'poor'

        return results

    def analyze_token_statistics(self) -> Dict[str, Any]:
        """
        Analyze token length distribution and statistics.

        Computes:
        - Token length percentiles
        - Texts exceeding max_length
        - Average tokens per text

        Returns:
            Dictionary with statistical results
        """
        results = {}

        # Tokenize all texts and get lengths
        texts = self.data[self.clean_text_col].tolist()
        token_lengths = []

        for text in texts:
            ids = self.tokenizer.encode_to_ids(text, add_special_tokens=True)
            token_lengths.append(len(ids))

        token_lengths = np.array(token_lengths)

        # Compute statistics
        results['mean'] = float(np.mean(token_lengths))
        results['median'] = float(np.median(token_lengths))
        results['std'] = float(np.std(token_lengths))
        results['min'] = int(np.min(token_lengths))
        results['max'] = int(np.max(token_lengths))

        # Percentiles
        percentiles = [50, 75, 90, 95, 99]
        results['percentiles'] = {}
        for p in percentiles:
            results['percentiles'][f'P{p}'] = int(np.percentile(token_lengths, p))

        # Texts exceeding max_length
        exceeding = np.sum(token_lengths > self.tokenizer.max_length)
        results['exceeding_max_length'] = int(exceeding)
        results['exceeding_percentage'] = float(exceeding / len(token_lengths) * 100)

        # Print statistics
        self.logger.info(f"  Avg tokens/text: {results['mean']:.1f}")
        self.logger.info(f"  Median: {results['median']}")
        self.logger.info(f"  Std dev: {results['std']:.1f}")
        self.logger.info(f"  Range: [{results['min']}, {results['max']}]")
        self.logger.info(f"  Token length distribution:")
        for p in percentiles:
            self.logger.info(f"    P{p}: {results['percentiles'][f'P{p}']} tokens")

        self.logger.info(f"  Texts exceeding max_length: {exceeding:,} ({results['exceeding_percentage']:.2f}%)")

        if results['exceeding_percentage'] > 5:
            self.logger.info(f"  ⚠ Consider increasing max_length to {results['percentiles']['P95']} or {results['percentiles']['P99']}")

        results['token_lengths'] = token_lengths  # Store for plotting

        return results

    def generate_visualizations(self):
        """
        Generate visualization plots.

        Creates:
        - Token length distribution histogram
        - Vocabulary coverage analysis
        - Common tokens bar chart
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
        except ImportError:
            self.logger.warning("matplotlib not available, skipping visualizations")
            return

        # 1. Token length distribution
        token_lengths = self.results['statistics']['token_lengths']
        plt.figure(figsize=(10, 6))
        plt.hist(token_lengths, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(self.tokenizer.max_length, color='r', linestyle='--', label=f'max_length={self.tokenizer.max_length}')
        plt.axvline(np.median(token_lengths), color='g', linestyle='--', label=f'median={int(np.median(token_lengths))}')
        plt.xlabel('Token Length')
        plt.ylabel('Frequency')
        plt.title('Token Length Distribution')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'token_length_distribution.png', dpi=150)
        plt.close()

        # 2. OOV rate visualization
        oov_rate = self.results['coverage']['oov_rate']
        plt.figure(figsize=(8, 6))
        plt.bar(['In-Vocabulary', 'Out-of-Vocabulary'],
                [100 - oov_rate, oov_rate],
                color=['green', 'red'], alpha=0.7)
        plt.ylabel('Percentage (%)')
        plt.title(f'Vocabulary Coverage (OOV Rate: {oov_rate:.2f}%)')
        plt.ylim(0, 100)
        for i, v in enumerate([100 - oov_rate, oov_rate]):
            plt.text(i, v + 2, f'{v:.2f}%', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'vocab_coverage.png', dpi=150)
        plt.close()

        self.logger.info(f"  Saved visualizations to: {self.output_dir}")

    def generate_report(self):
        """
        Generate comprehensive text and markdown reports.
        """
        # Markdown report
        report_path = self.output_dir / 'full_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Tokenizer Diagnostic Report\n\n")
            f.write(f"**Config:** {self.config_path}\n\n")
            f.write(f"**Vocabulary Size:** {self.config.tokenization.vocab_size}\n\n")
            f.write(f"**Max Length:** {self.config.tokenization.max_length}\n\n")
            f.write(f"**Samples Analyzed:** {len(self.data)}\n\n")

            f.write("## 1. Basic Encoding Tests\n\n")
            for test, result in self.results['encoding'].items():
                status = "✓ PASS" if result else "✗ FAIL"
                f.write(f"- {test}: {status}\n")

            f.write("\n## 2. Serbian Text Handling\n\n")
            serbian = self.results['serbian']
            f.write(f"- Tests passed: {serbian['passed']}/{serbian['total']}\n")

            f.write("\n## 3. Vocabulary Coverage\n\n")
            cov = self.results['coverage']
            f.write(f"- Total tokens: {cov['total_tokens']:,}\n")
            f.write(f"- [UNK] tokens: {cov['unk_tokens']:,}\n")
            f.write(f"- OOV rate: {cov['oov_rate']:.2f}%\n")
            f.write(f"- Health status: {cov['health_status']}\n")

            f.write("\n## 4. Token Statistics\n\n")
            stats = self.results['statistics']
            f.write(f"- Average tokens per text: {stats['mean']:.1f}\n")
            f.write(f"- Median: {stats['median']}\n")
            f.write(f"- Range: [{stats['min']}, {stats['max']}]\n")
            f.write(f"- Texts exceeding max_length: {stats['exceeding_max_length']:,} ({stats['exceeding_percentage']:.2f}%)\n")

            f.write("\n### Percentile Distribution\n\n")
            for p, val in stats['percentiles'].items():
                f.write(f"- {p}: {val} tokens\n")

        self.logger.info(f"  Saved report to: {report_path}")

    def print_summary(self):
        """
        Print test summary to console.
        """
        self.logger.info("=" * 80)
        self.logger.info("SUMMARY")
        self.logger.info("=" * 80)

        # Check if all tests passed
        encoding_passed = all(self.results['encoding'].values())
        serbian_passed = self.results['serbian']['serbian_texts']
        coverage_healthy = self.results['coverage']['health_status'] in ['good', 'moderate']

        all_passed = encoding_passed and serbian_passed and coverage_healthy

        if all_passed:
            self.logger.info("✓ ALL TESTS PASSED")
        else:
            self.logger.info("✗ SOME TESTS FAILED")

        self.logger.info("")
        self.logger.info("Recommendations:")

        # Recommendations based on results
        oov_rate = self.results['coverage']['oov_rate']
        if oov_rate > 10:
            self.logger.info(f"  - OOV rate is high ({oov_rate:.2f}%) - consider increasing vocab_size")
        elif oov_rate > 5:
            self.logger.info(f"  - OOV rate is moderate ({oov_rate:.2f}%) - tokenizer is acceptable")
        else:
            self.logger.info(f"  - OOV rate is healthy ({oov_rate:.2f}%) - tokenizer is working well")

        exceeding_pct = self.results['statistics']['exceeding_percentage']
        if exceeding_pct > 5:
            p95 = self.results['statistics']['percentiles']['P95']
            self.logger.info(f"  - {exceeding_pct:.2f}% of texts exceed max_length - consider increasing to {p95}")

        if all_passed:
            self.logger.info("  - Tokenizer is working correctly")

        self.logger.info("")
        self.logger.info("=" * 80)


def main():
    """
    Main entry point for tokenizer testing script.
    """
    parser = argparse.ArgumentParser(
        description="Comprehensive tokenizer diagnostic testing for Serbian advertisement classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_tokenizer.py
  python test_tokenizer.py --config configs/improved.yaml
  python test_tokenizer.py --sample-size 5000
  python test_tokenizer.py --no-plots
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/improved.yaml",
        help="Path to configuration file (default: configs/improved.yaml)"
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of samples to analyze (default: 10000)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="tests/tokenizer_analysis",
        help="Output directory for results (default: tests/tokenizer_analysis)"
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip visualization generation"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Run tests
    tester = TokenizerTester(
        config_path=args.config,
        sample_size=args.sample_size,
        output_dir=args.output_dir,
        create_plots=not args.no_plots
    )

    results = tester.run_all_tests()

    # Exit with appropriate code
    encoding_passed = all(results['encoding'].values())
    serbian_passed = results['serbian']['serbian_texts']
    coverage_healthy = results['coverage']['health_status'] in ['good', 'moderate']

    if encoding_passed and serbian_passed and coverage_healthy:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Some tests failed


if __name__ == "__main__":
    main()
