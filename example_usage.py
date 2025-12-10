import pandas as pd
from pathlib import Path
from src.utils import ConfigManager, setup_logging
from src.data import (
    create_data_loader,
    SerbianTextPreprocessor,
    create_default_filters,
    StratifiedDataSplitter
)
from src.tokenization import WordPieceTokenizer

def main():
    """Demonstrate the improved architecture."""

    config_manager = ConfigManager()
    config = config_manager.create_default_config()

    logger = setup_logging(
        log_level=config.training.log_level,
        experiment_name=config.experiment_name
    )

    logger.info("Starting improved AI classifier pipeline demonstration")
    logger.info("Step 1: Loading data")

    sample_data = pd.DataFrame({
        'text': [
            'Prodajem iPhone 12 128GB',
            'Apartman za izdavanje',
            'Samsung Galaxy S21',
            'Kuća na prodaju',
            'BMW X5 2020',
            'MacBook Pro 13 inch',
            'Stan u centru grada',
            'Audi A4 dizel',
            'Nokia 3310 classic',
            'Garsonjera izdavanje',
            'Tesla Model 3',
            'iPad Air novi',
            'Villa sa dvorištem',
            'Mercedes C klasa',
            'Samsung TV 55 inch',
            'Dvosoban stan centar',
            'Volkswagen Golf 7',
            'iPhone 13 Pro Max',
            'Trosoban stan',
            'Opel Astra'
        ],
        'group_id': [1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    })

    logger.info(f"Loaded sample data: {len(sample_data)} rows")
    logger.info("Step 2: Text preprocessing")

    preprocessor = SerbianTextPreprocessor(
        transliterate_cyrillic=True,
        lowercase=True,
        remove_stop_words=True,
        verbose=True
    )

    processed_data = preprocessor.preprocess_dataframe(
        sample_data,
        text_column='text',
        output_column='clean_text'
    )

    logger.info("Text preprocessing completed")
    logger.info("Step 3: Data filtering")

    filters = create_default_filters(
        class_column='group_id',
        text_column='clean_text',
        min_samples_per_class=2,  # Lower threshold for demo
        remove_ostalo=False  # Skip ostalo filtering for demo
    )

    filtered_data = filters.filter(processed_data)

    logger.info(f"Filtered data: {len(filtered_data)} rows")
    logger.info("Step 4: Data splitting")

    splitter = StratifiedDataSplitter(
        class_column='group_id',
        val_size=0.2,  # Reasonable for demo
        test_size=0.15,
        verbose=True
    )

    train_data, val_data, test_data = splitter.split(filtered_data)

    logger.info("Step 5: Tokenization")

    tokenizer = WordPieceTokenizer(
        vocab_size=1000,  # Smaller for demo
        max_length=50,
        verbose=True
    )

    tokenizer.train(train_data['clean_text'], use_cache=False)

    sample_text = "prodajem samsung galaxy s21"
    tokens = tokenizer.encode_text(sample_text)
    token_ids = tokenizer.encode_to_ids(sample_text)

    logger.info(f"Sample tokenization:")
    logger.info(f"  Text: '{sample_text}'")
    logger.info(f"  Tokens: {tokens}")
    logger.info(f"  Token IDs: {token_ids}")

    analysis = tokenizer.analyze_tokenization(train_data['clean_text'])
    logger.info("Tokenization analysis:")
    for key, value in analysis.items():
        if isinstance(value, dict):
            logger.info(f"  {key}: {dict(list(value.items())[:5])}...")  
        else:
            logger.info(f"  {key}: {value}")

    logger.info("Pipeline demonstration completed successfully!")
    logger.info("Key improvements demonstrated:")
    logger.info("  - Modular, object-oriented design")
    logger.info("  - Serbian language support with Cyrillic/Latin handling")
    logger.info("  - Flexible configuration management")
    logger.info("  - Enhanced tokenization with caching")
    logger.info("  - Comprehensive logging")
    logger.info("  - Better error handling and validation")


if __name__ == "__main__":
    main()