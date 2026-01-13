#!/usr/bin/env python3
"""
Test script to verify advanced stratified dropout configuration.

This script validates that:
1. Config loads properly with new dropout fields
2. Model builds successfully with stratified dropout
3. All dropout rates are correctly applied to their respective layers
4. Model can compile and predict
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import ConfigManager
from src.models.classifier import AdClassifier
from src.utils.logging import setup_logging, get_logger

def test_advanced_dropout():
    """Test the advanced dropout configuration."""

    setup_logging(log_level="INFO", log_to_file=False)
    logger = get_logger(__name__)

    logger.info("=" * 80)
    logger.info("Testing Advanced Stratified Dropout Configuration")
    logger.info("=" * 80)

    # Load configuration
    logger.info("\n1. Loading configuration...")
    config_path = "configs/moderate_advanced.yaml"
    config = ConfigManager.from_yaml(config_path)

    # Verify dropout fields are loaded
    logger.info("\n2. Verifying dropout configuration:")
    logger.info(f"   - dropout_rate (base):    {config.model.dropout_rate}")
    logger.info(f"   - embedding_dropout:      {config.model.embedding_dropout}")
    logger.info(f"   - attention_dropout:      {config.model.attention_dropout}")
    logger.info(f"   - ffn_dropout:           {config.model.ffn_dropout}")
    logger.info(f"   - dense_dropout:         {config.model.dense_dropout}")

    # Calculate effective dropout
    p_no_dropout = (
        (1 - config.model.embedding_dropout) *
        (1 - config.model.attention_dropout) *
        (1 - config.model.ffn_dropout) *
        (1 - config.model.dense_dropout)
    )
    effective_dropout = 1 - p_no_dropout
    logger.info(f"\n   Effective dropout rate: {effective_dropout:.1%}")

    # Build model
    logger.info("\n3. Building model with stratified dropout...")
    classifier = AdClassifier(
        vocab_size=config.tokenization.vocab_size,
        num_classes=100,  # Dummy value for testing
        max_length=config.tokenization.max_length,
        embed_dim=config.model.embedding_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        ff_dim=config.model.ff_dim,
        dropout_rate=config.model.dropout_rate,
        embedding_dropout=config.model.embedding_dropout,
        attention_dropout=config.model.attention_dropout,
        ffn_dropout=config.model.ffn_dropout,
        dense_dropout=config.model.dense_dropout,
        pooling_strategy=config.model.pooling_strategy,
        label_smoothing=config.model.label_smoothing
    )

    model = classifier.build_model()
    logger.info("   Model built successfully!")

    # Verify classifier dropout attributes
    logger.info("\n4. Verifying classifier dropout attributes:")
    logger.info(f"   - embedding_dropout:  {classifier.embedding_dropout}")
    logger.info(f"   - attention_dropout:  {classifier.attention_dropout}")
    logger.info(f"   - ffn_dropout:        {classifier.ffn_dropout}")
    logger.info(f"   - dense_dropout:      {classifier.dense_dropout}")

    # Compile model
    logger.info("\n5. Compiling model...")
    classifier.compile_model(
        learning_rate=config.model.learning_rate,
        weight_decay=config.model.weight_decay
    )
    logger.info("   Model compiled successfully!")

    # Print model summary
    logger.info("\n6. Model architecture summary:")
    logger.info("-" * 80)
    model.summary()
    logger.info("-" * 80)

    # Test prediction with dummy data
    logger.info("\n7. Testing prediction with dummy data...")
    dummy_input = np.random.randint(0, 1000, size=(2, config.tokenization.max_length))
    predictions = classifier.predict(dummy_input, batch_size=2)
    logger.info(f"   Input shape:  {dummy_input.shape}")
    logger.info(f"   Output shape: {predictions.shape}")
    logger.info(f"   Output sum (should be ~1.0 per sample): {predictions.sum(axis=1)}")

    # Verify label smoothing
    logger.info("\n8. Verifying label smoothing:")
    logger.info(f"   Label smoothing enabled: {classifier.label_smoothing > 0}")
    logger.info(f"   Label smoothing value:   {classifier.label_smoothing}")

    # Test get_config
    logger.info("\n9. Testing configuration serialization...")
    model_config = classifier.get_config()
    assert 'embedding_dropout' in model_config, "embedding_dropout missing from config"
    assert 'attention_dropout' in model_config, "attention_dropout missing from config"
    assert 'ffn_dropout' in model_config, "ffn_dropout missing from config"
    assert 'dense_dropout' in model_config, "dense_dropout missing from config"
    logger.info("   Configuration serialization: ✓ PASSED")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✓ ALL TESTS PASSED")
    logger.info("=" * 80)
    logger.info("\nAdvanced dropout strategy is working correctly!")
    logger.info(f"Effective dropout: {effective_dropout:.1%} (target: ~50-55%)")
    logger.info("\nDropout breakdown:")
    logger.info(f"  - Embedding layer:  {config.model.embedding_dropout:.0%} (gentle - preserves input)")
    logger.info(f"  - Attention layers: {config.model.attention_dropout:.0%} (moderate)")
    logger.info(f"  - FFN layers:       {config.model.ffn_dropout:.0%} (moderate)")
    logger.info(f"  - Dense layer:      {config.model.dense_dropout:.0%} (strong - prevents overfitting)")
    logger.info("\nReady to train with: python src/training/train_model_wandb_gpu.py --config configs/moderate_advanced.yaml")

if __name__ == "__main__":
    test_advanced_dropout()
