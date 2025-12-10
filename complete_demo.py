import pandas as pd
import numpy as np
from pathlib import Path

from src.utils import ConfigManager, setup_logging
from src.data import (
    SerbianTextPreprocessor,
    create_default_filters,
    StratifiedDataSplitter
)
from src.tokenization import WordPieceTokenizer
from src.models import create_classifier_model, calculate_class_weights

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


def create_sample_data(n_samples=100):
    """Create sample Serbian ad data."""
    templates = {
        1: [  # Elektronika
            'Prodajem iPhone {} {}GB',
            'Samsung Galaxy {} kao nov',
            'MacBook Pro {} inch',
            'iPad Air {}',
            'Nokia {} telefon',
            'Laptop Lenovo {}',
            'Samsung TV {} inch',
            'PlayStation {} konzola',
        ],
        2: [  # Nekretnine
            'Stan {} m2 u centru',
            'Kuća {} m2 sa dvorištem',
            'Apartman {} m2 izdavanje',
            'Garsonjera {} m2',
            'Trosoban stan {}',
            'Vila sa {} m2',
            'Poslovni prostor {} m2',
        ],
        3: [  # Automobili
            'BMW {} dizel',
            'Mercedes {} model',
            'Audi A{} quattro',
            'Volkswagen Golf {}',
            'Opel Astra {}',
            'Tesla Model {}',
            'Ford Focus {}',
        ]
    }

    data = []
    for _ in range(n_samples):
        class_id = np.random.choice([1, 2, 3])
        template = np.random.choice(templates[class_id])

        if class_id == 1:
            text = template.format(np.random.choice(['11', '12', '13', '14']),
                                 np.random.choice([64, 128, 256]))
        elif class_id == 2:
            text = template.format(np.random.randint(30, 150))
        else:
            text = template.format(np.random.choice(['X5', 'C', '4', '7', '3']))

        data.append({'text': text, 'group_id': class_id})

    return pd.DataFrame(data)


def main():
    """Run complete demo pipeline."""

    # Setup
    logger = setup_logging(log_level='INFO', experiment_name='complete_demo')
    logger.info("="*60)
    logger.info("COMPLETE DEMO: Serbian Ad Classifier")
    logger.info("="*60)

    # STEP 1: Data Preparation
    logger.info("\n[1/5] DATA PREPARATION")
    logger.info("-" * 60)

    data = create_sample_data(n_samples=200)
    logger.info(f"Created {len(data)} sample ads")
    logger.info(f"Class distribution:\n{data['group_id'].value_counts()}")

    # STEP 2: Text Preprocessing
    logger.info("\n[2/5] TEXT PREPROCESSING")
    logger.info("-" * 60)

    preprocessor = SerbianTextPreprocessor(
        transliterate_cyrillic=True,
        lowercase=True,
        remove_stop_words=False,  # Keep all words for better demo
        verbose=False
    )

    data = preprocessor.preprocess_dataframe(data, 'text', 'clean_text')
    logger.info(f"Example:\n  Original: {data['text'].iloc[0]}\n  Cleaned:  {data['clean_text'].iloc[0]}")

    # Filtering and splitting
    filters = create_default_filters('group_id', 'clean_text', min_samples_per_class=5, remove_ostalo=False)
    data = filters.filter(data)

    splitter = StratifiedDataSplitter('group_id', val_size=0.15, test_size=0.15, verbose=False)
    train_data, val_data, test_data = splitter.split(data)

    logger.info(f"Split sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # STEP 3: Tokenization
    logger.info("\n[3/5] TOKENIZATION")
    logger.info("-" * 60)

    tokenizer = WordPieceTokenizer(vocab_size=500, max_length=50, verbose=False)
    tokenizer.train(train_data['clean_text'], use_cache=False)

    logger.info(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    logger.info(f"Max sequence length: {tokenizer.max_length}")

    # Tokenize all splits
    X_train = tokenizer.encode_batch(train_data['clean_text'].tolist())
    X_val = tokenizer.encode_batch(val_data['clean_text'].tolist())
    X_test = tokenizer.encode_batch(test_data['clean_text'].tolist())

    y_train = train_data['group_id'].values - 1  # Convert to 0-indexed
    y_val = val_data['group_id'].values - 1
    y_test = test_data['group_id'].values - 1

    logger.info(f"Tokenized data shapes: X_train={X_train.shape}, y_train={y_train.shape}")

    # STEP 4: Model Training
    logger.info("\n[4/5] MODEL TRAINING")
    logger.info("-" * 60)

    num_classes = len(np.unique(y_train))

    # Create model with smaller architecture for demo
    classifier = create_classifier_model(
        vocab_size=tokenizer.get_vocab_size(),
        num_classes=num_classes,
        config={
            'max_length': 50,
            'embed_dim': 128,
            'num_heads': 4,
            'num_layers': 2,
            'ff_dim': 256,
            'dropout_rate': 0.1,
            'pooling_strategy': 'cls',
            'label_smoothing': 0.1
        }
    )

    # Build and compile
    classifier.build_model()
    classifier.compile_model(learning_rate=1e-3, optimizer='adamw')

    logger.info("Model architecture:")
    logger.info(classifier.get_model_summary())

    # Calculate class weights
    class_counts = {i: np.sum(y_train == i) for i in range(num_classes)}
    class_weights = calculate_class_weights(class_counts, strategy='balanced')
    classifier.set_class_weights(class_weights)

    logger.info(f"Training with class weights: {class_weights}")

    # Train
    logger.info("\nTraining model...")
    history = classifier.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=16,
        class_weight=class_weights,
        verbose=1
    )

    # STEP 5: Evaluation
    logger.info("\n[5/5] EVALUATION")
    logger.info("-" * 60)

    # Evaluate on test set
    test_results = classifier.model.evaluate(X_test, y_test, verbose=0)

    logger.info(f"Test Loss: {test_results[0]:.4f}")
    logger.info(f"Test Accuracy: {test_results[1]:.4f}")
    logger.info(f"Test Top-3 Accuracy: {test_results[2]:.4f}")

    # Predictions on sample texts
    logger.info("\nSample Predictions:")
    logger.info("-" * 60)

    test_texts = [
        "Prodajem iPhone 13 128GB nov",
        "Stan 45 m2 u centru grada",
        "BMW X5 dizel automatik"
    ]
    class_names = {0: 'Elektronika', 1: 'Nekretnine', 2: 'Automobili'}

    for text in test_texts:
        clean = preprocessor.preprocess_text(text)
        encoded = tokenizer.encode_batch([clean])[0]  # Get padded token IDs

        predictions = classifier.predict(np.array([encoded]))
        top_class = np.argmax(predictions[0])
        confidence = predictions[0][top_class]

        logger.info(f"\nText: '{text}'")
        logger.info(f"Predicted: {class_names[top_class]} (confidence: {confidence:.2%})")
        logger.info(f"All probabilities: {dict(zip(class_names.values(), predictions[0]))}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("DEMO COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info("\nPipeline steps demonstrated:")
    logger.info("  ✓ Data preparation with Serbian text support")
    logger.info("  ✓ Text preprocessing and filtering")
    logger.info("  ✓ WordPiece tokenization")
    logger.info("  ✓ Transformer model training")
    logger.info("  ✓ Model evaluation and predictions")


if __name__ == "__main__":
    main()
